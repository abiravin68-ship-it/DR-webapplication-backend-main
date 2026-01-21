from application import app  # noqa: F401

# ASGI app (FastAPI). Use Uvicorn on Render:
#   uvicorn application:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"
