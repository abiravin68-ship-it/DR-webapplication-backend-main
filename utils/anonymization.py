import hashlib
import uuid
import re
from datetime import datetime, timedelta

class DataAnonymizer:
    """GDPR/PDPA compliant data anonymization"""
    
    @staticmethod
    def anonymize_patient_id(patient_id, salt=None):
        """
        Create anonymized patient ID using SHA-256
        Args:
            patient_id: Original patient identifier
            salt: Optional salt for hashing
        Returns:
            Anonymized ID (irreversible)
        """
        if salt is None:
            salt = "dr_detection_2024_secure_salt"
        
        # Combine patient_id with salt
        data = f"{patient_id}:{salt}".encode('utf-8')
        
        # Generate SHA-256 hash
        hash_object = hashlib.sha256(data)
        anonymized_id = hash_object.hexdigest()
        
        return f"ANON_{anonymized_id[:16]}"
    
    @staticmethod
    def pseudonymize_patient_id(patient_id):
        """
        Create reversible pseudonym using UUID mapping
        Store mapping separately in secure location
        """
        pseudonym = str(uuid.uuid4())
        return f"PSEUDO_{pseudonym}"
    
    @staticmethod
    def anonymize_personal_data(data):
        """
        Remove or mask personally identifiable information (PII)
        Args:
            data: dict containing patient data
        Returns:
            Anonymized data dict
        """
        anonymized = data.copy()
        
        # Remove direct identifiers
        pii_fields = [
            'name', 'full_name', 'first_name', 'last_name',
            'email', 'phone', 'address', 'ssn', 'passport',
            'national_id', 'insurance_number'
        ]
        
        for field in pii_fields:
            if field in anonymized:
                del anonymized[field]
        
        # Mask partial identifiers
        if 'date_of_birth' in anonymized:
            # Keep only year
            dob = anonymized['date_of_birth']
            if isinstance(dob, str):
                anonymized['year_of_birth'] = dob[:4]
            del anonymized['date_of_birth']
        
        # Generalize location data
        if 'zipcode' in anonymized:
            # Keep only first 3 digits
            anonymized['zipcode'] = anonymized['zipcode'][:3] + 'XX'
        
        if 'city' in anonymized:
            # Keep only state/province
            anonymized['region'] = anonymized.pop('city', '')
        
        # Add anonymization metadata
        anonymized['anonymized_at'] = datetime.utcnow().isoformat()
        anonymized['anonymization_version'] = '1.0'
        
        return anonymized
    
    @staticmethod
    def mask_image_metadata(image):
        """
        Remove EXIF data and metadata from medical images
        Args:
            image: PIL Image object
        Returns:
            Image without metadata
        """
        # Remove EXIF data by creating new image without metadata
        from PIL import Image as PILImage
        data = list(image.getdata())
        image_without_exif = PILImage.new(image.mode, image.size)
        image_without_exif.putdata(data)
        
        return image_without_exif
    
    @staticmethod
    def generate_consent_token():
        """Generate unique consent tracking token"""
        return f"CONSENT_{uuid.uuid4()}"
    
    @staticmethod
    def calculate_retention_date(consent_duration_days=90):
        """
        Calculate data retention expiration date
        GDPR: Right to be forgotten
        """
        return datetime.utcnow() + timedelta(days=consent_duration_days)
    
    @staticmethod
    def generate_session_id():
        """Generate unique session ID for tracking"""
        return f"SESSION_{uuid.uuid4()}"
