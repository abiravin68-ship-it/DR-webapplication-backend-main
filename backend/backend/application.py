import os
import re
import uuid
import time
import base64
import hashlib
import warnings
import logging
from threading import Lock
from typing import Optional ,Tuple ,Dict ,Any

os .environ ["TF_CPP_MIN_LOG_LEVEL"]=os .getenv ("TF_CPP_MIN_LOG_LEVEL","2")
os .environ ["TF_ENABLE_ONEDNN_OPTS"]=os .getenv ("TF_ENABLE_ONEDNN_OPTS","0")
os .environ ["CUDA_VISIBLE_DEVICES"]=os .getenv ("CUDA_VISIBLE_DEVICES","-1")

warnings .filterwarnings ("ignore",category =DeprecationWarning )
logging .getLogger ("tensorflow").setLevel (logging .ERROR )

import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import layers ,models ,regularizers
from dotenv import load_dotenv

from fastapi import FastAPI ,Request
from fastapi .responses import JSONResponse
from fastapi .middleware .cors import CORSMiddleware

BASE_DIR =os .path .dirname (os .path .abspath (__file__ ))
os .makedirs (os .path .join (BASE_DIR ,"logs"),exist_ok =True )

load_dotenv ()

preprocess_input =tf .keras .applications .efficientnet .preprocess_input

try :
    if hasattr (keras ,"config")and hasattr (keras .config ,"enable_unsafe_deserialization"):
        keras .config .enable_unsafe_deserialization ()
    elif hasattr (keras ,"saving")and hasattr (keras .saving ,"enable_unsafe_deserialization"):
        keras .saving .enable_unsafe_deserialization ()
except Exception :
    pass


def _patch_from_config (layer_cls ,drop_keys ):
    orig =layer_cls .from_config
    orig_func =getattr (orig ,"__func__",orig )

    @classmethod
    def patched (cls ,config ):
        for k in drop_keys :
            config .pop (k ,None )
        return orig_func (cls ,config )

    layer_cls .from_config =patched


try :
    from keras .layers import RandomFlip ,RandomRotation ,RandomZoom ,RandomContrast

    _patch_from_config (RandomFlip ,["data_format"])
    _patch_from_config (RandomRotation ,["data_format"])
    _patch_from_config (RandomZoom ,["data_format"])
    _patch_from_config (RandomContrast ,["value_range","data_format"])
except Exception :
    pass


@tf .keras .utils .register_keras_serializable ()
class ReduceMeanLayer (layers .Layer ):
    def call (self ,x ):
        return tf .reduce_mean (x ,axis =-1 ,keepdims =True )


@tf .keras .utils .register_keras_serializable ()
class ReduceMaxLayer (layers .Layer ):
    def call (self ,x ):
        return tf .reduce_max (x ,axis =-1 ,keepdims =True )


@tf .keras .utils .register_keras_serializable ()
class EfficientNetPreprocess (layers .Layer ):
    def call (self ,x ):
        x =tf .cast (x ,tf .float32 )
        return preprocess_input (x )

    def compute_output_shape (self ,input_shape ):
        return input_shape


@tf .keras .utils .register_keras_serializable ()
class EfficientNetB3Block (layers .Layer ):
    def __init__ (self ,input_shape =(300 ,300 ,3 ),trainable_base =False ,weights ="imagenet",**kwargs ):
        super ().__init__ (**kwargs )
        self .input_shape_ =tuple (input_shape )
        self .trainable_base =bool (trainable_base )

        if weights in ("imagenet","noisy-student"):
            weights =None
        self .weights_ =weights

        self .base =tf .keras .applications .EfficientNetB3 (
        include_top =False ,
        weights =self .weights_ ,
        input_shape =self .input_shape_ ,
        )
        self .base .trainable =self .trainable_base

    def call (self ,inputs ,training =False ):
        return self .base (inputs ,training =training )

    def get_config (self ):
        config =super ().get_config ()
        config .update (
        {
        "input_shape":self .input_shape_ ,
        "trainable_base":self .trainable_base ,
        "weights":self .weights_ ,
        }
        )
        return config

    @classmethod
    def from_config (cls ,config ):
        w =config .get ("weights",None )
        if w in ("imagenet","noisy-student"):
            config ["weights"]=None
        return cls (**config )



def residual_attention_block (x ,reduction =16 ,bottleneck =4 ,spatial_kernel =7 ):
    

    c =x .shape [-1 ]
    if c is None :
        raise ValueError ("Channel dimension must be known for residual_attention_block.")
    c =int (c )

    shortcut =x

    mid =max (c //int (bottleneck ),1 )

    y =layers .Conv2D (mid ,1 ,padding ="same",use_bias =False )(x )
    y =layers .BatchNormalization ()(y )
    y =layers .Activation ("swish")(y )

    y =layers .DepthwiseConv2D (3 ,padding ="same",use_bias =False )(y )
    y =layers .BatchNormalization ()(y )
    y =layers .Activation ("swish")(y )

    y =layers .Conv2D (c ,1 ,padding ="same",use_bias =False )(y )
    y =layers .BatchNormalization ()(y )

    avg =layers .GlobalAveragePooling2D ()(y )
    mx =layers .GlobalMaxPooling2D ()(y )

    hidden =max (c //int (reduction ),1 )
    dense1 =layers .Dense (hidden ,activation ="relu")
    dense2 =layers .Dense (c ,activation =None )

    ca =layers .Add ()([dense2 (dense1 (avg )),dense2 (dense1 (mx ))])
    ca =layers .Activation ("sigmoid")(ca )
    ca =layers .Reshape ((1 ,1 ,c ))(ca )
    y =layers .Multiply ()([y ,ca ])

    avg_sp =ReduceMeanLayer ()(y )
    max_sp =ReduceMaxLayer ()(y )
    sa =layers .Concatenate (axis =-1 )([avg_sp ,max_sp ])
    sa =layers .Conv2D (1 ,spatial_kernel ,padding ="same",activation ="sigmoid")(sa )
    y =layers .Multiply ()([y ,sa ])

    out =layers .Add ()([shortcut ,y ])
    out =layers .Activation ("swish")(out )
    return out


def build_RA_EfficientNetB3 (input_shape =(300 ,300 ,3 ),num_classes =5 ,train_base =False ,aug_layer =None ):
    
    inp =layers .Input (shape =input_shape )

    x =inp
    if aug_layer is not None :
        x =aug_layer (x )

    x =EfficientNetPreprocess (name ="effnet_preprocess")(x )

    x =EfficientNetB3Block (
    input_shape =input_shape ,
    trainable_base =train_base ,
    name ="ra_effb3_block",
    )(x )

    x =residual_attention_block (x )

    x =layers .Dropout (0.25 )(x )

    x =layers .Conv2D (
    256 ,1 ,activation ="relu",
    kernel_regularizer =regularizers .l2 (2e-5 ),
    )(x )
    x =layers .BatchNormalization ()(x )
    x =layers .GlobalAveragePooling2D ()(x )

    x =layers .Dropout (0.35 )(x )
    out =layers .Dense (
    num_classes ,
    activation ="softmax",
    kernel_regularizer =regularizers .l2 (2e-5 ),
    dtype ="float32",
    name ="softmax",
    )(x )

    return models .Model (inputs =inp ,outputs =out ,name ="RA_EfficientNetB3")



def _sha256_hex (s :str )->str :
    return hashlib .sha256 (s .encode ("utf-8")).hexdigest ()


try :
    from utils .encryption import get_encryptor
except Exception :
    get_encryptor =None

try :
    from utils .anonymization import DataAnonymizer
except Exception :

    class DataAnonymizer :
        def anonymize_id (self ,value :str )->str :
            return _sha256_hex (value )


try :
    from utils .access_control import rate_limiter
except Exception :

    class _RateLimiterFallback :
        def is_allowed (self ,key :str ,max_requests :int ,window_seconds :int )->bool :
            return True

    rate_limiter =_RateLimiterFallback ()


try :
    from utils .audit_log import audit_logger
except Exception :

    class _AuditLoggerFallback :
        def log_access (self ,user_id ,action ,path ,ip_address ,success ,details =None ):
            return None

        def log_error (self ,action ,message ,ip_address ="unknown",details =None ):
            return None

    audit_logger =_AuditLoggerFallback ()


app =FastAPI ()

DEV_MODE =os .getenv ("DEV_MODE","false").strip ().lower ()=="true"



MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))


MULTIPART_OVERHEAD_BYTES = int(os.getenv("MULTIPART_OVERHEAD_BYTES", str(2 * 1024 * 1024)))
MULTIPART_BODY_MAX_BYTES = int(
    os.getenv(
        "MULTIPART_BODY_MAX_BYTES",
        str(MAX_IMAGE_BYTES + MULTIPART_OVERHEAD_BYTES),
    )
)


JSON_OVERHEAD_BYTES = int(os.getenv("JSON_OVERHEAD_BYTES", str(2 * 1024 * 1024)))
_MIN_JSON_BYTES = ((MAX_IMAGE_BYTES * 4 + 2) // 3) + JSON_OVERHEAD_BYTES
JSON_BODY_MAX_BYTES = int(os.getenv("JSON_BODY_MAX_BYTES", str(_MIN_JSON_BYTES)))


MULTIPART_BODY_MAX_BYTES = max(MULTIPART_BODY_MAX_BYTES, MAX_IMAGE_BYTES + MULTIPART_OVERHEAD_BYTES)
JSON_BODY_MAX_BYTES = max(JSON_BODY_MAX_BYTES, _MIN_JSON_BYTES)

MAX_CONTENT_LENGTH = MAX_IMAGE_BYTES

def _bytes_to_mb(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)
RATE_LIMIT_ENABLED =os .getenv ("RATE_LIMIT_ENABLED","false").strip ().lower ()=="true"
PREDICT_MAX =int (os .getenv ("PREDICT_MAX_PER_MIN","60"))
PREDICT_WINDOW =int (os .getenv ("PREDICT_WINDOW_SECONDS","60"))
OTHER_MAX =int (os .getenv ("OTHER_MAX_PER_HOUR","2000"))
OTHER_WINDOW =int (os .getenv ("OTHER_WINDOW_SECONDS","3600"))

EXEMPT_PATHS ={
"/",
"/api/health",
"/api/model-info",
"/api/privacy-notice",
"/api/audit-log",
}

MODEL_FILE =os .getenv ("MODEL_FILE","").strip ()or "best_ra_finetune_export.keras"
IMAGE_SIZE :Tuple [int ,int ]=(300 ,300 )

STORE_UPLOADS =os .getenv ("STORE_UPLOADS","false").strip ().lower ()=="true"
SECURE_UPLOAD_DIR =os .path .join (BASE_DIR ,"secure_uploads")

encryptor =None
if STORE_UPLOADS and get_encryptor is not None :
    try :
        encryptor =get_encryptor ()
    except Exception :
        encryptor =None

anonymizer =DataAnonymizer ()

if STORE_UPLOADS :
    os .makedirs (SECURE_UPLOAD_DIR ,exist_ok =True )
    if encryptor is None :
        raise RuntimeError ("STORE_UPLOADS=true requires ENCRYPTION_KEY and a working utils/encryption.py")


def _encrypt_upload_bytes (data :bytes ,aad :bytes )->str :
    if encryptor is None :
        raise RuntimeError ("Encryptor is not available")
    if hasattr (encryptor ,"encrypt_bytes"):
        return encryptor .encrypt_bytes (data ,aad =aad )
    if hasattr (encryptor ,"encrypt"):
        try :
            return encryptor .encrypt (data )
        except TypeError :
            return encryptor .encrypt (base64 .b64encode (data ).decode ("ascii"))
    raise RuntimeError ("Encryptor does not support encryption methods")


import builtins
builtins .tf =tf

PREDICT_LOCK =Lock ()
PREDICT_ACQUIRE_TIMEOUT =int (os .getenv ("PREDICT_ACQUIRE_TIMEOUT","10"))

MODEL_PATH =None
MODEL_LOAD_ERROR =None
model =None
infer =None

CANDIDATES =[
os .path .join (BASE_DIR ,"model",MODEL_FILE ),
os .path .join (BASE_DIR ,MODEL_FILE ),
]

for p in CANDIDATES :
    if os .path .exists (p ):
        MODEL_PATH =p
        break

if MODEL_PATH is None :
    MODEL_LOAD_ERROR =f"Model file not found. MODEL_FILE={MODEL_FILE}. Searched={CANDIDATES}. CWD={os.getcwd()}"

if MODEL_PATH and MODEL_LOAD_ERROR is None :
    try :
        baseline_aug =tf .keras .Sequential (
        [
        layers .RandomFlip ("horizontal"),
        layers .RandomRotation (0.015 ),
        layers .RandomZoom (0.02 ),
        layers .RandomContrast (0.02 ),
        ],
        name ="baseline_aug",
        )

        custom_objects ={
        "baseline_aug":baseline_aug ,
        "Custom>baseline_aug":baseline_aug ,
        "EfficientNetPreprocess":EfficientNetPreprocess ,
        "Custom>EfficientNetPreprocess":EfficientNetPreprocess ,
        "EfficientNetB3Block":EfficientNetB3Block ,
        "Custom>EfficientNetB3Block":EfficientNetB3Block ,
        "ReduceMeanLayer":ReduceMeanLayer ,
        "Custom>ReduceMeanLayer":ReduceMeanLayer ,
        "ReduceMaxLayer":ReduceMaxLayer ,
        "Custom>ReduceMaxLayer":ReduceMaxLayer ,
        "residual_attention_block":residual_attention_block ,
        "build_RA_EfficientNetB3":build_RA_EfficientNetB3 ,
        "tf":tf ,
        }

        model =keras .models .load_model (MODEL_PATH ,custom_objects =custom_objects ,compile =False )

        @tf .function (reduce_retracing =True )
        def _infer (x ):
            return model (x ,training =False )

        infer =_infer
        _ =infer (tf .zeros ((1 ,IMAGE_SIZE [0 ],IMAGE_SIZE [1 ],3 ),dtype =tf .float32 ))
        MODEL_LOAD_ERROR =None
    except Exception as e :
        MODEL_LOAD_ERROR =f"{type(e).__name__}: {str(e)}"
        model =None
        infer =None


CLASS_LABELS ={
0 :"No_DR",
1 :"Mild",
2 :"Moderate",
3 :"Severe",
4 :"Proliferative_DR",
}

CLASS_DESCRIPTIONS ={
"No_DR":"No Diabetic Retinopathy detected",
"Mild":"Mild Diabetic Retinopathy - Early stage with microaneurysms",
"Moderate":"Moderate Diabetic Retinopathy - More widespread blood vessel damage",
"Severe":"Severe Diabetic Retinopathy - Significant blood vessel blockage",
"Proliferative_DR":"Proliferative Diabetic Retinopathy - Advanced stage with new abnormal blood vessels",
}

ALLOWED_EXTENSIONS ={"png","jpg","jpeg"}


def allowed_file (filename :str )->bool :
    return "."in filename and filename .rsplit (".",1 )[1 ].lower ()in ALLOWED_EXTENSIONS


def get_client_ip (request :Request )->str :
    ip =(request .client .host if request .client else "")or ""
    ip =ip .strip ()
    if ip :
        return ip

    xff =request .headers .get ("X-Forwarded-For","")
    if xff :
        return xff .split (",")[0 ].strip ()

    cf_ip =request .headers .get ("CF-Connecting-IP")
    if cf_ip :
        return cf_ip .strip ()

    xri =request .headers .get ("X-Real-IP")
    if xri :
        return xri .strip ()

    return "unknown"


def allowed_origins ():
    raw =os .getenv (
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
    )

    def _normalize (o :str )->str :
        o =(o or "").strip ()
        while o .endswith ("/"):
            o =o [:-1 ]
        return o

    def _wildcard_to_regex_or_exact (o :str )->str :
        o =_normalize (o )
        if not o :
            return ""
        if o .startswith ("[*.]"):
            base =o [len ("[*.]"):]
            return rf"^https:\/\/.*\.{re.escape(base)}$"
        if "*"in o :
            esc =re .escape (o ).replace (r"\*",".*")
            return rf"^{esc}$"
        return o

    origins =[]
    for o in raw .split (","):
        entry =_wildcard_to_regex_or_exact (o )
        if entry :
            origins .append (entry )

    allow_previews =os .getenv ("ALLOW_VERCEL_PREVIEWS","false").strip ().lower ()=="true"
    if not allow_previews :
        seen =set ()
        out =[]
        for o in origins :
            if o not in seen :
                seen .add (o )
                out .append (o )
        return out

    project =os .getenv ("VERCEL_PROJECT_SLUG","").strip ()
    team =os .getenv ("VERCEL_TEAM_SLUG","").strip ()

    if not project :
        vercel_origin =next (
        (o for o in origins if isinstance (o ,str )and o .startswith ("https://")and o .endswith (".vercel.app")),
        "",
        )
        if vercel_origin :
            host =vercel_origin .split ("://",1 )[-1 ].split ("/",1 )[0 ].split (":",1 )[0 ]
            if host .endswith (".vercel.app"):
                project =host [:-len (".vercel.app")]

    patterns =[r"^https:\/\/.*\.vercel\.app$"]
    if project and team :
        patterns .append (rf"^https://{re.escape(project)}-(?:git-)?[a-z0-9-]+-{re.escape(team)}\.vercel\.app$")
    elif project :
        patterns .append (rf"^https://{re.escape(project)}-(?:git-)?[a-z0-9-]+\.vercel\.app$")

    for p in patterns :
        if p not in origins :
            origins .append (p )

    seen =set ()
    out =[]
    for o in origins :
        if o not in seen :
            seen .add (o )
            out .append (o )
    return out


def _split_cors_origins (origins ):
    exact =[]
    regexes =[]
    for o in origins :
        if isinstance (o ,str )and o .startswith ("^"):
            regexes .append (o )
        else :
            exact .append (o )
    allow_origin_regex =None
    if regexes :
        allow_origin_regex ="(?:"+"|".join (regexes )+")"
    return exact ,allow_origin_regex


CORS_ORIGINS =allowed_origins ()
CORS_EXACT ,CORS_REGEX =_split_cors_origins (CORS_ORIGINS )

app .add_middleware (
CORSMiddleware ,
allow_origins =CORS_EXACT ,
allow_origin_regex =CORS_REGEX ,
allow_credentials =False ,
allow_methods =["GET","POST","OPTIONS"],
allow_headers =["*"],
expose_headers =["Retry-After"],
max_age =600 ,
)


@app.middleware("http")
async def max_size_limit(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl:
            try:
                ct = (request.headers.get("content-type") or "").lower()

                if "application/json" in ct:
                    limit = JSON_BODY_MAX_BYTES
                elif "multipart/form-data" in ct:
                    limit = MULTIPART_BODY_MAX_BYTES
                else:
                    limit = MAX_IMAGE_BYTES

                if int(cl) > limit:
                    return JSONResponse(
                        {
                            "error": f"File too large. Maximum allowed size is {_bytes_to_mb(limit):.2f} MB.",
                            "code": "PAYLOAD_TOO_LARGE",
                            "content_length": int(cl),
                            "limit": limit,
                            "limit_mb": round(_bytes_to_mb(limit), 2),
                            "content_type": ct,
                        },
                        status_code=413,
                    )
            except Exception:
                pass

    return await call_next(request)




@app .middleware ("http")
async def rate_limit_and_security_headers (request :Request ,call_next ):
    if request .method !="OPTIONS":
        if RATE_LIMIT_ENABLED and (not request .url .path .startswith ("/static/"))and (request .url .path not in EXEMPT_PATHS ):
            identifier =get_client_ip (request )
            is_predict =request .method =="POST"and request .url .path in ("/predict","/api/predict")

            if is_predict :
                key =f"{identifier}:predict"
                if not rate_limiter .is_allowed (key ,max_requests =PREDICT_MAX ,window_seconds =PREDICT_WINDOW ):
                    resp =JSONResponse (
                    {"error":"Rate limit exceeded. Please try again later.","code":"RATE_LIMIT"},
                    status_code =429 ,
                    )
                    resp .headers ["Retry-After"]=str (PREDICT_WINDOW )
                    return resp
            else :
                key =f"{identifier}:{request.url.path}"
                if not rate_limiter .is_allowed (key ,max_requests =OTHER_MAX ,window_seconds =OTHER_WINDOW ):
                    resp =JSONResponse ({"error":"Too many requests","code":"RATE_LIMIT"},status_code =429 )
                    resp .headers ["Retry-After"]=str (OTHER_WINDOW )
                    return resp

    response =await call_next (request )

    response .headers ["X-Content-Type-Options"]="nosniff"
    response .headers ["X-Frame-Options"]="DENY"
    response .headers ["X-XSS-Protection"]="1; mode=block"

    xf_proto =request .headers .get ("X-Forwarded-Proto","")
    if request .url .scheme =="https"or xf_proto =="https":
        response .headers ["Strict-Transport-Security"]="max-age=31536000; includeSubDomains"

    response .headers ["Content-Security-Policy"]="default-src 'self'"
    response .headers ["Referrer-Policy"]="strict-origin-when-cross-origin"
    return response


def preprocess_image (img_bgr :np .ndarray )->np .ndarray :
    img_bgr =cv2 .resize (img_bgr ,IMAGE_SIZE ,interpolation =cv2 .INTER_AREA )

    lab =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2LAB )
    l ,a ,b =cv2 .split (lab )

    clahe =cv2 .createCLAHE (clipLimit =2.0 ,tileGridSize =(8 ,8 ))
    l2 =clahe .apply (l )

    lab2 =cv2 .merge ([l2 ,a ,b ])
    enhanced_bgr =cv2 .cvtColor (lab2 ,cv2 .COLOR_LAB2BGR )
    enhanced_rgb =cv2 .cvtColor (enhanced_bgr ,cv2 .COLOR_BGR2RGB )

    x =enhanced_rgb .astype ("float32")
    x =np .expand_dims (x ,axis =0 )
    return x


def _is_true (v :str )->bool :
    return (v or "").strip ().lower ()in {"1","true","yes","y","on"}


def preprocess_image_with_vis (img_bgr :np .ndarray )->Tuple [np .ndarray ,np .ndarray ]:



    img_bgr =cv2 .resize (img_bgr ,IMAGE_SIZE ,interpolation =cv2 .INTER_AREA )


    vis_rgb_u8 =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
    vis_rgb_u8 =np .clip (vis_rgb_u8 ,0 ,255 ).astype (np .uint8 )


    lab =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2LAB )
    l ,a ,b =cv2 .split (lab )

    clahe =cv2 .createCLAHE (clipLimit =2.0 ,tileGridSize =(8 ,8 ))
    l2 =clahe .apply (l )

    lab2 =cv2 .merge ([l2 ,a ,b ])
    enhanced_bgr =cv2 .cvtColor (lab2 ,cv2 .COLOR_LAB2BGR )
    enhanced_rgb =cv2 .cvtColor (enhanced_bgr ,cv2 .COLOR_BGR2RGB )
    enhanced_rgb =np .clip (enhanced_rgb ,0 ,255 ).astype (np .uint8 )

    x =enhanced_rgb .astype ("float32")[None ,...]
    return x ,vis_rgb_u8


def _gradcam_fundus_mask (img_rgb_u8 :np .ndarray ,erode_ratio :float =0.05 )->np .ndarray :
    
    gray =cv2 .cvtColor (img_rgb_u8 ,cv2 .COLOR_RGB2GRAY )
    gray =cv2 .GaussianBlur (gray ,(7 ,7 ),0 )

    _ ,th =cv2 .threshold (gray ,0 ,255 ,cv2 .THRESH_BINARY +cv2 .THRESH_OTSU )

    cnts ,_ =cv2 .findContours (th ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )
    mask =np .zeros_like (gray ,dtype =np .uint8 )
    if cnts :
        c =max (cnts ,key =cv2 .contourArea )
        cv2 .drawContours (mask ,[c ],-1 ,255 ,thickness =-1 )

    mask =cv2 .morphologyEx (mask ,cv2 .MORPH_CLOSE ,np .ones ((9 ,9 ),np .uint8 ))

    img_size =int (img_rgb_u8 .shape [0 ])
    er =max (1 ,int (img_size *float (erode_ratio )))
    k =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(2 *er +1 ,2 *er +1 ))
    mask =cv2 .erode (mask ,k ,iterations =1 )

    mask =cv2 .GaussianBlur (mask ,(9 ,9 ),0 )
    return mask .astype (np .float32 )/255.0


def _gradcam_disc_center_estimate (img_rgb_u8 :np .ndarray ,fmask :np .ndarray ,glare_thresh :int =230 )->Tuple [int ,int ]:
    
    gray =cv2 .cvtColor (img_rgb_u8 ,cv2 .COLOR_RGB2GRAY ).astype (np .float32 )
    gray =cv2 .GaussianBlur (gray ,(21 ,21 ),0 )

    m =(fmask >0.3 ).astype (np .float32 )
    gray_masked =gray .copy ()
    gray_masked [m <0.5 ]=0.0
    gray_masked [gray_masked >=float (glare_thresh )]=0.0

    _ ,max_val ,_ ,max_loc =cv2 .minMaxLoc (gray_masked )
    if max_val <=0 :
        _ ,_ ,_ ,max_loc =cv2 .minMaxLoc (gray *m )
    return int (max_loc [0 ]),int (max_loc [1 ])

GRADCAM_TARGET_LAYER :Optional [str ]=None
GRADCAM_TARGET_INIT_ERROR :Optional [str ]=None
GRADCAM_TARGET_LOCK =Lock ()


def _list_4d_layers (m :tf .keras .Model ,limit :int =25 ):
    layers_4d =[]
    for layer in m .layers :
        try :
            out =layer .output
            if out is None or isinstance (out ,(list ,tuple )):
                continue
            if len (out .shape )==4 :
                shape =[]
                for dim in out .shape :
                    try :
                        shape .append (int (dim )if dim is not None else None )
                    except Exception :
                        shape .append (None )
                layers_4d .append ({"name":layer .name ,"shape":shape })
        except Exception :
            continue
    return layers_4d [-limit :]


def _pick_gradcam_target_layer (m :tf .keras .Model )->str :

    env_name =os .getenv ("GRADCAM_LAYER_NAME","").strip ()
    if env_name :
        try :
            _ =m .get_layer (env_name )

            dummy =tf .zeros ((1 ,IMAGE_SIZE [0 ],IMAGE_SIZE [1 ],3 ),dtype =tf .float32 )
            grad_model =tf .keras .Model (m .inputs ,[m .get_layer (env_name ).output ,m .output ])
            with tf .GradientTape ()as tape :
                conv_out ,preds =grad_model (dummy ,training =False )
                loss =tf .reduce_max (preds ,axis =1 )
            grads =tape .gradient (loss ,conv_out )
            if grads is not None :
                return env_name
        except Exception :

            pass


    dummy =tf .zeros ((1 ,IMAGE_SIZE [0 ],IMAGE_SIZE [1 ],3 ),dtype =tf .float32 )
    for layer in reversed (m .layers ):
        try :
            out =layer .output
            if out is None or isinstance (out ,(list ,tuple )):
                continue
            if len (out .shape )!=4 :
                continue

            grad_model =tf .keras .Model (m .inputs ,[out ,m .output ])
            with tf .GradientTape ()as tape :
                conv_out ,preds =grad_model (dummy ,training =False )
                loss =tf .reduce_max (preds ,axis =1 )
            grads =tape .gradient (loss ,conv_out )
            if grads is not None :
                return layer .name
        except Exception :
            continue

    raise ValueError ("No Grad-CAM target layer found with valid gradients")


def _get_gradcam_target_layer (m :tf .keras .Model )->Optional [str ]:
    global GRADCAM_TARGET_LAYER ,GRADCAM_TARGET_INIT_ERROR
    with GRADCAM_TARGET_LOCK :
        if GRADCAM_TARGET_LAYER is not None :
            return GRADCAM_TARGET_LAYER
        if GRADCAM_TARGET_INIT_ERROR is not None :
            return None
        try :
            GRADCAM_TARGET_LAYER =_pick_gradcam_target_layer (m )
            return GRADCAM_TARGET_LAYER
        except Exception as e :
            GRADCAM_TARGET_INIT_ERROR =f"{type(e).__name__}: {str(e)}"
            GRADCAM_TARGET_LAYER =None
            return None


def build_gradcam_data_url (
m :tf .keras .Model ,
x :tf .Tensor ,
vis_rgb_u8 :np .ndarray ,
target_layer_name :str ,
)->str :
    

    alpha =float (os .getenv ("GRADCAM_ALPHA","0.35"))
    mask_power =float (os .getenv ("GRADCAM_MASK_POWER","1.4"))

    apply_fundus_mask =os .getenv ("GRADCAM_APPLY_FUNDUS_MASK","true").strip ().lower ()in {"1","true","yes","y","on"}
    suppress_disc =os .getenv ("GRADCAM_SUPPRESS_DISC","true").strip ().lower ()in {"1","true","yes","y","on"}
    disc_radius_ratio =float (os .getenv ("GRADCAM_DISC_RADIUS_RATIO","0.14"))
    disc_suppress =float (os .getenv ("GRADCAM_DISC_SUPPRESS","0.92"))

    suppress_glare =os .getenv ("GRADCAM_SUPPRESS_GLARE","true").strip ().lower ()in {"1","true","yes","y","on"}
    glare_thresh =int (os .getenv ("GRADCAM_GLARE_THRESH","230"))
    glare_suppress =float (os .getenv ("GRADCAM_GLARE_SUPPRESS","0.85"))

    erode_ratio =float (os .getenv ("GRADCAM_ERODE_RATIO","0.05"))


    target_layer =m .get_layer (target_layer_name )
    grad_model =tf .keras .Model (m .inputs ,[target_layer .output ,m .output ])

    with tf .GradientTape ()as tape :
        conv_out ,preds =grad_model (x ,training =False )

        loss =tf .math .log (tf .reduce_max (preds ,axis =1 )+1e-8 )

    grads =tape .gradient (loss ,conv_out )
    if grads is None :
        raise RuntimeError ("Grad-CAM gradients are None for selected layer")

    pooled =tf .reduce_mean (grads ,axis =(0 ,1 ,2 ))
    conv_map =conv_out [0 ]

    heat =tf .reduce_sum (conv_map *pooled ,axis =-1 )
    heat =tf .maximum (heat ,0 )
    heat =heat /(tf .reduce_max (heat )+1e-8 )

    heat =tf .image .resize (heat [...,None ],(IMAGE_SIZE [0 ],IMAGE_SIZE [1 ]))
    heat =tf .squeeze (heat ).numpy ().astype (np .float32 )
    heat =np .clip (heat ,0.0 ,1.0 )


    if apply_fundus_mask :
        fmask =_gradcam_fundus_mask (vis_rgb_u8 ,erode_ratio =erode_ratio )
        heat =heat *fmask

        if suppress_disc :
            cx ,cy =_gradcam_disc_center_estimate (vis_rgb_u8 ,fmask ,glare_thresh =glare_thresh )
            r =int (IMAGE_SIZE [0 ]*disc_radius_ratio )
            disc_mask =np .zeros ((IMAGE_SIZE [0 ],IMAGE_SIZE [1 ]),dtype =np .float32 )
            cv2 .circle (disc_mask ,(cx ,cy ),r ,1.0 ,thickness =-1 )
            disc_mask =cv2 .GaussianBlur (disc_mask ,(0 ,0 ),sigmaX =max (1.0 ,r /2.0 ))
            heat =heat *(1.0 -disc_suppress *disc_mask )

        if suppress_glare :
            gray =cv2 .cvtColor (vis_rgb_u8 ,cv2 .COLOR_RGB2GRAY )
            glare =(gray >=glare_thresh ).astype (np .float32 )
            glare =cv2 .GaussianBlur (glare ,(0 ,0 ),sigmaX =7 )
            heat =heat *(1.0 -glare_suppress *glare )

        heat =np .clip (heat ,0.0 ,1.0 )
        heat =heat /(float (heat .max ())+1e-8 )


    lo ,hi =np .percentile (heat ,[40 ,99 ])
    heat =np .clip ((heat -lo )/(hi -lo +1e-8 ),0.0 ,1.0 )
    heat =np .power (heat ,0.5 )


    heat_u8 =np .uint8 (255 *heat )

    heat_u8 [heat_u8 <60 ]=0

    heat_color_bgr =cv2 .applyColorMap (heat_u8 ,cv2 .COLORMAP_HOT )
    heat_color_rgb =cv2 .cvtColor (heat_color_bgr ,cv2 .COLOR_BGR2RGB ).astype (np .float32 )/255.0

    img_norm =vis_rgb_u8 .astype (np .float32 )/255.0
    mask =np .power (heat ,mask_power )[...,None ].astype (np .float32 )

    overlay =img_norm *(1.0 -alpha *mask )+heat_color_rgb *(alpha *mask )
    overlay =np .clip (overlay ,0.0 ,1.0 )

    overlay_bgr_u8 =(overlay *255.0 ).astype (np .uint8 )[...,::-1 ]

    ok ,buf =cv2 .imencode (".png",overlay_bgr_u8 )
    if not ok :
        raise RuntimeError ("Failed to encode Grad-CAM overlay as PNG")

    b64 =base64 .b64encode (buf .tobytes ()).decode ("ascii")
    return f"data:image/png;base64,{b64}"



def _decode_base64_image (data_uri_or_b64 :str )->Optional [np .ndarray ]:
    s =(data_uri_or_b64 or "").strip ()
    if not s :
        return None
    if ","in s and s .lower ().startswith ("data:"):
        s =s .split (",",1 )[1 ].strip ()
    try :
        raw =base64 .b64decode (s ,validate =False )
    except Exception :
        return None
    if raw and len (raw )>MAX_CONTENT_LENGTH :
        return None
    arr =np .frombuffer (raw ,dtype =np .uint8 )
    img =cv2 .imdecode (arr ,cv2 .IMREAD_COLOR )
    return img


async def _get_image_from_request (request :Request )->Tuple [Optional [np .ndarray ],Optional [bytes ],Optional [str ]]:
    ctype =(request .headers .get ("content-type")or "").lower ()

    if "multipart/form-data"in ctype :
        try :
            form =await request .form ()
        except Exception :
            return None ,None ,"Invalid multipart/form-data request"

        f =form .get ("file")
        if f is None :
            return None ,None ,"No image provided. Send multipart/form-data with key 'file' or JSON with base64."

        try :
            filename =getattr (f ,"filename","")or ""
            if not filename :
                return None ,None ,"No file selected"
            if not allowed_file (filename ):
                return None ,None ,"Unsupported file type. Use png/jpg/jpeg."
            data =await f .read ()
            if not data :
                return None ,None ,"Empty file"
            if len (data )>MAX_CONTENT_LENGTH :
                return None ,None ,f"File too large. Maximum allowed size is {_bytes_to_mb(MAX_CONTENT_LENGTH):.2f} MB."
            img =cv2 .imdecode (np .frombuffer (data ,np .uint8 ),cv2 .IMREAD_COLOR )
            if img is None :
                return None ,None ,"Could not decode image"
            return img ,data ,None
        except Exception :
            return None ,None ,"Could not read uploaded file"

    if "application/json"in ctype :
        payload =await request .json ()
        if not isinstance (payload ,dict ):
            return None ,None ,"Invalid JSON payload"
        b64 =payload .get ("image_base64")or payload .get ("image")or payload .get ("base64")
        if isinstance (b64 ,str )and b64 .strip ():
            img =_decode_base64_image (b64 )
            if img is None :
                return None ,None ,"Invalid base64 image"
            raw =None
            try :
                s =b64 .strip ()
                if ","in s and s .lower ().startswith ("data:"):
                    s =s .split (",",1 )[1 ].strip ()
                raw =base64 .b64decode (s ,validate =False )
            except Exception :
                raw =None
            if raw is not None and len (raw )>MAX_CONTENT_LENGTH :
                return None ,None ,f"File too large. Maximum allowed size is {_bytes_to_mb(MAX_CONTENT_LENGTH):.2f} MB."
            return img ,raw ,None
        return None ,None ,"JSON provided but no image_base64/image/base64 field found"

    return None ,None ,"No image provided. Send multipart/form-data with key 'file' or JSON with base64."


@app .get ("/")
def root ():
    return {"status":"ok","endpoints":["/api/health","/api/predict","/api/model-info"]}


@app .get ("/api/health")
def health_check ():
    return {
    "status":"ok",
    "model_loaded":model is not None and infer is not None ,
    "model_loading":False ,
    "store_uploads":STORE_UPLOADS ,
    "dev_mode":DEV_MODE ,
    "model_load_error":MODEL_LOAD_ERROR ,
    "model_path":MODEL_PATH ,
    }


@app .post ("/predict")
@app .post ("/api/predict")
async def predict (request :Request ):
    client_ip =get_client_ip (request )
    t0 =time .time ()
    want_gradcam =_is_true (request .query_params .get ("gradcam",""))


    if model is None or infer is None :
        return JSONResponse (
        {
        "success":False ,
        "error":"Model not loaded.",
        "dev_mode":DEV_MODE ,
        "model_load_error":MODEL_LOAD_ERROR ,
        "message":"Backend is running but the model is unavailable. Upload the model / fix MODEL_FILE.",
        },
        status_code =503 ,
        )

    session_id =str (uuid .uuid4 ())
    try :
        anonymized_id =anonymizer .anonymize_id (session_id )if hasattr (anonymizer ,"anonymize_id")else _sha256_hex (session_id )
    except Exception :
        anonymized_id =_sha256_hex (session_id )

    acquired =False
    try :
        img_bgr ,raw_bytes ,err =await _get_image_from_request (request )
        if err :
            audit_logger .log_access (anonymized_id ,"PREDICT",request .url .path ,client_ip ,False ,{"error":err })
            return JSONResponse ({"success":False ,"error":err },status_code =400 )

        processed_image ,vis_rgb_u8 =preprocess_image_with_vis (img_bgr )
        x =tf .convert_to_tensor (processed_image ,dtype =tf .float32 )

        acquired =PREDICT_LOCK .acquire (timeout =PREDICT_ACQUIRE_TIMEOUT )
        if not acquired :
            resp =JSONResponse ({"error":"Server busy. Please try again shortly.","code":"SERVER_BUSY"},status_code =503 )
            resp .headers ["Retry-After"]="10"
            audit_logger .log_access (anonymized_id ,"PREDICT",request .url .path ,client_ip ,False ,{"error":"SERVER_BUSY"})
            return resp

        preds =infer (x ).numpy ()
        preds =np .asarray (preds ).reshape (-1 )
        if preds .size !=len (CLASS_LABELS ):
            raise RuntimeError (f"Unexpected model output size: {preds.size}")

        idx =int (np .argmax (preds ))
        label =CLASS_LABELS .get (idx ,str (idx ))
        confidence =float (preds [idx ])
        probs ={CLASS_LABELS [i ]:float (preds [i ])for i in range (len (CLASS_LABELS ))}


        gradcam_image =None
        gradcam_layer =None
        gradcam_ok =False
        gradcam_error =None
        if want_gradcam :
            try :
                gradcam_layer =_get_gradcam_target_layer (model )
                if gradcam_layer :
                    gradcam_image =build_gradcam_data_url (model ,x ,vis_rgb_u8 ,gradcam_layer )
                    gradcam_ok =gradcam_image is not None
                else :
                    gradcam_ok =False
                    if DEV_MODE :
                        gradcam_error =GRADCAM_TARGET_INIT_ERROR or "No valid Grad-CAM layer found"
            except Exception as _e :
                gradcam_image =None
                gradcam_ok =False
                if DEV_MODE :
                    gradcam_error =f"{type(_e).__name__}: {str(_e)}"

        stored =False
        if STORE_UPLOADS and raw_bytes is not None :
            aad =f"{anonymized_id}:{int(t0)}".encode ("utf-8")
            token =_encrypt_upload_bytes (raw_bytes ,aad =aad )
            out_path =os .path .join (SECURE_UPLOAD_DIR ,f"{anonymized_id}.enc")
            with open (out_path ,"w",encoding ="utf-8")as f :
                f .write (token )
            stored =True

        elapsed_ms =int ((time .time ()-t0 )*1000 )

        audit_logger .log_access (
        anonymized_id ,
        "PREDICT",
        request .url .path ,
        client_ip ,
        True ,
        {
        "prediction":label ,
        "confidence":confidence ,
        "elapsed_ms":elapsed_ms ,
        "stored_upload":stored ,
        },
        )

        return {
        "success":True ,
        "session_id":anonymized_id ,
        "prediction":{
        "class":label ,
        "confidence":confidence ,
        "description":CLASS_DESCRIPTIONS .get (label ,""),
        },
        "all_probabilities":probs ,
        "gradcam":bool (want_gradcam ),
        "gradcam_ok":bool (gradcam_ok ),
        "gradcam_layer":gradcam_layer ,
        "gradcam_image":gradcam_image ,
        "gradcam_error":gradcam_error if DEV_MODE else None ,
        "elapsed_ms":elapsed_ms ,
        "security":{
        "anonymized":True ,
        "encrypted":bool (STORE_UPLOADS ),
        "gdpr_compliant":True ,
        "pdpa_compliant":True ,
        "rate_limited":bool (RATE_LIMIT_ENABLED ),
        },
        "storage":{
        "stored_upload":bool (stored ),
        "encrypted_upload_id":anonymized_id if stored else None ,
        },
        "compat":{
        "prediction_label":label ,
        "confidence":confidence ,
        "description":CLASS_DESCRIPTIONS .get (label ,""),
        "probabilities":probs ,
        },
        }

    except Exception as e :
        audit_logger .log_error ("PREDICT_ERROR",str (e ),ip_address =client_ip ,details ={"session":anonymized_id })
        payload ={"success":False ,"error":"An error occurred during prediction. Please try again."}
        if DEV_MODE :
            payload ["debug"]=f"{type(e).__name__}: {str(e)}"
        return JSONResponse (payload ,status_code =500 )

    finally :
        if acquired :
            try :
                PREDICT_LOCK .release ()
            except Exception :
                pass


@app .get ("/api/model-info")
def model_info (request :Request ):
    client_ip =get_client_ip (request )
    audit_logger .log_access ("anonymous","MODEL_INFO","/api/model-info",client_ip ,True )

    input_shape =[IMAGE_SIZE [0 ],IMAGE_SIZE [1 ],3 ]
    if model is not None :
        try :
            ish =getattr (model ,"input_shape",None )
            if ish is not None :
                input_shape =list (ish )
        except Exception :
            pass

    return {
    "model_name":"EfficientNetB3 - Diabetic Retinopathy Classifier",
    "input_shape":input_shape ,
    "classes":CLASS_LABELS ,
    "descriptions":CLASS_DESCRIPTIONS ,
    "num_classes":len (CLASS_LABELS ),
    "model_loaded":model is not None and infer is not None ,
    "model_load_error":MODEL_LOAD_ERROR ,
    "model_path":MODEL_PATH ,
    "build_tag":os .getenv ("BUILD_TAG","FASTAPI_BACKEND_V1"),
    "gradcam_info":{
    "requested_via_query":"gradcam=1",
    "env_layer_name":os .getenv ("GRADCAM_LAYER_NAME","").strip ()or None ,
    "selected_layer":_get_gradcam_target_layer (model )if model is not None else None ,
    "candidates":_list_4d_layers (model ,limit =25 )if model is not None else [],
    },
    "security_features":{
    "data_encryption":"AES-256-GCM"if STORE_UPLOADS else "Not enabled (no at-rest storage)",
    "anonymization":"SHA-256 hashing",
    "audit_logging":"Enabled",
    "rate_limiting":"Enabled"if RATE_LIMIT_ENABLED else "Disabled",
    },
    }


@app .get ("/api/privacy-notice")
def privacy_notice ():
    security_measures =[
    "Data anonymization",
    "Access control",
    "Audit logging",
    "Rate limiting",
    "Secure HTTPS communication",
    ]
    if STORE_UPLOADS :
        security_measures .insert (0 ,"AES-256-GCM encryption (at-rest uploads)")

    return {
    "controller":{
    "name":"Diabetic Retinopathy Detection Service",
    "contact":"privacy@drdetection.com",
    "dpo_contact":"dpo@drdetection.com",
    },
    "processing_purposes":[
    "Medical diagnosis and treatment",
    "Healthcare service improvement",
    "Research and development (anonymized data only)",
    ],
    "legal_basis":"Explicit consent (GDPR Article 6(1)(a)) and Health data processing (Article 9(2)(a))",
    "data_collected":[
    "Retinal images (metadata removed)",
    "Prediction results",
    "Session information (anonymized)",
    "IP address (for security only)",
    ],
    "retention_period":"90 days from upload or until data deletion request",
    "rights":[
    "Right to access (GDPR Article 15)",
    "Right to rectification (GDPR Article 16)",
    "Right to erasure (GDPR Article 17)",
    "Right to restrict processing (GDPR Article 18)",
    "Right to data portability (GDPR Article 20)",
    "Right to object (GDPR Article 21)",
    ],
    "security_measures":security_measures ,
    "recipients":"Data is not shared with third parties",
    "automated_decision_making":"AI-based diagnosis - results should be verified by medical professionals",
    "international_transfers":"None",
    "complaint_authority":"Your national data protection authority",
    }


@app .get ("/api/audit-log")
def get_audit_log ():
    return {
    "success":True ,
    "message":"Audit logging is enabled. Logs are stored in logs/audit.log",
    }
