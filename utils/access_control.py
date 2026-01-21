from functools import wraps
from flask import request, jsonify
import jwt
from datetime import datetime, timedelta
import os

class AccessControl:
    """Role-Based Access Control (RBAC)"""
    
    # Define roles and permissions
    ROLES = {
        'admin': ['read', 'write', 'delete', 'manage_users', 'view_audit'],
        'doctor': ['read', 'write', 'diagnose'],
        'nurse': ['read', 'upload'],
        'researcher': ['read_anonymized'],
        'patient': ['read_own'],
        'public': ['predict']  # Public access for predictions
    }
    
    SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key-change-in-production-please!')
    
    @staticmethod
    def generate_token(user_id, role='public', expires_in=3600):
        """
        Generate JWT access token
        Args:
            user_id: User identifier
            role: User role
            expires_in: Token expiration in seconds
        """
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, AccessControl.SECRET_KEY, algorithm='HS256')
        return token
    
    @staticmethod
    def verify_token(token):
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, AccessControl.SECRET_KEY, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}
    
    @staticmethod
    def require_auth(required_permission=None):
        """
        Decorator to require authentication
        Usage: @require_auth('read')
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Get token from header
                auth_header = request.headers.get('Authorization')
                
                if not auth_header:
                    # For public endpoints, allow without token
                    if required_permission == 'predict':
                        request.user = {'user_id': 'anonymous', 'role': 'public'}
                        return f(*args, **kwargs)
                    return jsonify({'error': 'No authorization header'}), 401
                
                try:
                    # Extract token
                    token = auth_header.split(' ')[1]  # "Bearer <token>"
                    
                    # Verify token
                    payload = AccessControl.verify_token(token)
                    
                    if 'error' in payload:
                        return jsonify({'error': payload['error']}), 401
                    
                    # Check permission
                    user_role = payload.get('role')
                    
                    if required_permission:
                        user_permissions = AccessControl.ROLES.get(user_role, [])
                        
                        if required_permission not in user_permissions:
                            return jsonify({'error': 'Insufficient permissions'}), 403
                    
                    # Add user info to request
                    request.user = payload
                    
                    return f(*args, **kwargs)
                    
                except Exception as e:
                    return jsonify({'error': f'Authorization failed: {str(e)}'}), 401
            
            return decorated_function
        return decorator

# Rate limiting for API security
class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, identifier, max_requests=100, window_seconds=3600):
        """
        Check if request is allowed
        Args:
            identifier: IP address or user ID
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        now = datetime.utcnow()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if (now - req_time).total_seconds() < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def clear(self):
        """Clear all rate limit data"""
        self.requests = {}

rate_limiter = RateLimiter()
