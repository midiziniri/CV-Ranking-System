import os
import re
import logging
import shutil
from datetime import datetime
from functools import wraps
from flask import session, request, jsonify, flash, redirect, url_for
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename

# Configure logger
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass

class FileHandlingError(Exception):
    """Custom file handling error"""
    pass

def validate_object_id(obj_id_str):
    """Validate MongoDB ObjectId format"""
    try:
        ObjectId(obj_id_str)
        return True
    except:
        return False

def validate_email(email):
    """Validate email format using regex"""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None

def validate_password(password):
    """
    Validate password strength
    Returns: (is_valid: bool, message: str)
    """
    if not password or not isinstance(password, str):
        return False, "Password is required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"

def validate_filename(filename, allowed_extensions):
    """Validate filename and extension"""
    if not filename or not isinstance(filename, str):
        return False, "No filename provided"
    
    if '.' not in filename:
        return False, "File must have an extension"
    
    extension = filename.rsplit('.', 1)[1].lower()
    if extension not in allowed_extensions:
        return False, f"Only {', '.join(allowed_extensions)} files are allowed"
    
    return True, "Valid filename"

def sanitize_filename(filename):
    """Create a safe filename with timestamp"""
    if not filename:
        return None
    
    secured = secure_filename(filename)
    if not secured:
        return None
    
    # Add timestamp to prevent conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    return timestamp + secured

def validate_job_posting_data(data):
    """Validate job posting form data"""
    errors = []
    
    job_profile = data.get('job_profile', '').strip()
    company = data.get('company', '').strip()
    last_date = data.get('last_date', '').strip()
    salary = data.get('salary', '').strip()
    
    if not job_profile or len(job_profile) < 2:
        errors.append("Job profile must be at least 2 characters long")
    
    if not company or len(company) < 2:
        errors.append("Company name must be at least 2 characters long")
    
    if not last_date:
        errors.append("Application deadline is required")
    else:
        try:
            last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
            if last_date_obj.date() <= datetime.now().date():
                errors.append("Application deadline must be in the future")
        except ValueError:
            errors.append("Invalid date format")
    
    if salary and not re.match(r'^[\d,\-\s$€£¥]+$', salary):
        errors.append("Invalid salary format")
    
    return errors

def require_login(f):
    """Decorator to require user login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page", "error")
            return redirect(url_for('loginpage'))
        return f(*args, **kwargs)
    return decorated_function

def require_role(required_role):
    """Decorator to require specific user role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash("Please log in to access this page", "error")
                return redirect(url_for('loginpage'))
            
            user_role = session.get('role', 'applicant')
            if user_role != required_role:
                flash(f"Access denied. {required_role.title()} privileges required.", "error")
                return redirect(url_for('index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def handle_api_errors(f):
    """Decorator for API error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}", exc_info=True)
            return jsonify({"success": False, "error": "Internal server error"}), 500
    return decorated_function

def safe_file_operation(operation, *args, **kwargs):
    """Safely perform file operations with error handling"""
    try:
        return operation(*args, **kwargs)
    except IOError as e:
        logger.error(f"File I/O error: {e}")
        raise FileHandlingError(f"File operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected file error: {e}")
        raise FileHandlingError(f"Unexpected file error: {e}")

def cleanup_old_files(directory_path, keep_files=None):
    """Clean up old files in a directory"""
    keep_files = keep_files or []
    
    try:
        if not os.path.exists(directory_path):
            return
        
        for filename in os.listdir(directory_path):
            if filename not in keep_files:
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Removed old directory: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up directory {directory_path}: {e}")

def extract_file_text(file_path):
    """Extract text from PDF or DOCX files"""
    try:
        if not os.path.exists(file_path):
            raise FileHandlingError("File does not exist")
        
        file_extension = file_path.rsplit('.', 1)[1].lower()
        text = ""
        
        if file_extension == 'pdf':
            import fitz
            with fitz.open(file_path) as doc:
                text = " ".join([page.get_text() for page in doc])
                
        elif file_extension == 'docx':
            import docx2txt
            text = docx2txt.process(file_path)
            if isinstance(text, list):
                text = ' '.join([line.replace('\t', ' ') for line in text if line])
                
        else:
            raise FileHandlingError(f"Unsupported file format: {file_extension}")
        
        # Validate extracted text
        if not text or len(text.strip()) < 10:
            raise FileHandlingError("Extracted text is too short or empty")
            
        return text.strip()
        
    except ImportError as e:
        logger.error(f"Missing required library for file extraction: {e}")
        raise FileHandlingError("Required library not available for file processing")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        raise FileHandlingError(f"Failed to extract text: {e}")

def log_user_action(action, user_id=None, details=None):
    """Log user actions for audit trail"""
    user_id = user_id or session.get('user_id', 'anonymous')
    details = details or {}
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'action': action,
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', ''),
        'details': details
    }
    
    logger.info(f"User action: {log_entry}")
    return log_entry

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_info(file_path):
    """Get comprehensive file information"""
    try:
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.rsplit('.', 1)[1].lower() if '.' in file_path else None
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None

class DatabaseManager:
    """Helper class for safe database operations"""
    
    @staticmethod
    def safe_find_one(collection, filter_dict, projection=None):
        """Safely find one document"""
        try:
            return collection.find_one(filter_dict, projection)
        except Exception as e:
            logger.error(f"Database find_one error: {e}")
            return None
    
    @staticmethod
    def safe_insert_one(collection, document):
        """Safely insert one document"""
        try:
            result = collection.insert_one(document)
            logger.info(f"Document inserted with ID: {result.inserted_id}")
            return result
        except Exception as e:
            logger.error(f"Database insert_one error: {e}")
            raise
    
    @staticmethod
    def safe_update_one(collection, filter_dict, update_dict):
        """Safely update one document"""
        try:
            result = collection.update_one(filter_dict, update_dict)
            logger.info(f"Documents modified: {result.modified_count}")
            return result
        except Exception as e:
            logger.error(f"Database update_one error: {e}")
            raise
    
    @staticmethod
    def safe_delete_one(collection, filter_dict):
        """Safely delete one document"""
        try:
            result = collection.delete_one(filter_dict)
            logger.info(f"Documents deleted: {result.deleted_count}")
            return result
        except Exception as e:
            logger.error(f"Database delete_one error: {e}")
            raise

def create_response(success=True, message="", data=None, status_code=200):
    """Create standardized API response"""
    response = {
        "success": success,
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    return jsonify(response), status_code