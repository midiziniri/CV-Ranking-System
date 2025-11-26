"""Job_post.py"""

from flask import Blueprint, render_template, request, url_for, redirect, session, jsonify, flash
from werkzeug.utils import secure_filename
import os, fitz, logging, re, json, time
from bson.objectid import ObjectId
import docx2txt
from database import mongo
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# CRITICAL: Import matching functions at module level
from Matching import ProductionMatching

# Import enhanced ranking system (with error handling)


from notifications import create_application_notification
try:
    from notifications import create_application_notification
    NOTIFICATIONS_AVAILABLE = True
    logging.info("✅ Notifications module loaded")
except ImportError as e:
    NOTIFICATIONS_AVAILABLE = False
    logging.warning(f"⚠️ Notifications module not available: {e}")
    def create_application_notification(*args, **kwargs):
        pass 
# Then later in apply_job_enhanced, the import is already at module level:


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

job_post = Blueprint("Job_post", __name__, static_folder="static", template_folder="templates")
UF = "static/Job_Description"
LOGO_FOLDER = "static/company_logos"
os.makedirs(UF, exist_ok=True)
os.makedirs(LOGO_FOLDER, exist_ok=True)

JOBS = mongo.db.JOBS
Applied_EMP = mongo.db.Applied_EMP
resumeFetchedData = mongo.db.resumeFetchedData
IRS_USERS = mongo.db.IRS_USERS

def allowedExtension(filename):
    """Enhanced file validation"""
    if not filename or '.' not in filename:
        return False
    if filename.count('.') > 1:  # Security: prevent double extensions
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ['docx', 'pdf']

def allowedLogo(filename):
    """Logo file validation"""
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ['png', 'jpg', 'jpeg', 'gif', 'webp']

def validate_job_data(job_profile, company, last_date, salary):
    """Enhanced job data validation"""
    errors = []
    
    if not job_profile or len(job_profile.strip()) < 2:
        errors.append("Job profile must be at least 2 characters long")
    elif len(job_profile.strip()) > 200:
        errors.append("Job profile too long (max 200 characters)")
    
    if not company or len(company.strip()) < 2:
        errors.append("Company name must be at least 2 characters long")
    elif len(company.strip()) > 100:
        errors.append("Company name too long (max 100 characters)")
    
    if not last_date:
        errors.append("Last date is required")
    else:
        try:
            last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
            if last_date_obj.date() <= datetime.now().date():
                errors.append("Last date must be in the future")
            if (last_date_obj.date() - datetime.now().date()).days > 730:
                errors.append("Last date cannot be more than 2 years in the future")
        except ValueError:
            errors.append("Invalid date format")
    
    if salary and len(salary) > 50:
        errors.append("Salary description too long")
    
    return errors

def cleanup_old_files(job_id, new_filename=None):
    """Enhanced file cleanup"""
    try:
        job_path = os.path.join(UF, str(job_id))
        if os.path.exists(job_path):
            for file in os.listdir(job_path):
                if new_filename and file != new_filename:
                    old_file_path = os.path.join(job_path, file)
                    try:
                        os.remove(old_file_path)
                        logger.info(f"Removed old file: {old_file_path}")
                    except OSError as e:
                        logger.error(f"Failed to remove file {old_file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up files for job {job_id}: {e}")

def extractData(file_path, ext):
    """Enhanced text extraction with validation"""
    text = ""
    try:
        if ext == "docx":
            text_lines = docx2txt.process(file_path)
            if text_lines:
                text = [line.replace('\t', ' ') for line in text_lines.split('\n') if line.strip()]
                text = ' '.join(text)
                
        elif ext == "pdf":
            with fitz.open(file_path) as doc:
                text_parts = []
                for page_num, page in enumerate(doc):
                    if page_num > 50:  # Limit pages
                        logger.warning(f"PDF has more than 50 pages, truncating at page {page_num}")
                        break
                    page_text = page.get_text()
                    if page_text:
                        text_parts.append(page_text)
                text = " ".join(text_parts)
        
        # Clean and validate
        if text:
            text = re.sub(r'\s+', ' ', text.strip())
            if len(text) < 50:
                raise ValueError("Document content too short (minimum 50 characters)")
            if len(text) > 100000:
                logger.warning(f"Large document detected, truncating")
                text = text[:100000] + "... [truncated]"
            if not re.search(r'[a-zA-Z]', text):
                raise ValueError("Document contains no readable text")
        else:
            raise ValueError("No text extracted from document")
            
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {e}")
        raise ValueError(f"Failed to extract text: {str(e)}")
    
    return text

@job_post.route("/")
def home():
    return "<h1>Job Posting Module</h1>"

class WeightConfig:
    """Weight configuration with all your original templates"""
    
    @staticmethod
    def get_role_templates():
        return {
            "technical": {
                "name": "Technical Roles",
                "description": "Software developers, engineers, technical specialists",
                "weights": {"skills": 50, "experience": 30, "education": 15, "certifications": 5}
            },
            "managerial": {
                "name": "Management Roles", 
                "description": "Team leads, managers, directors",
                "weights": {"skills": 25, "experience": 45, "education": 20, "certifications": 10}
            },
            "entry_level": {
                "name": "Entry Level Roles",
                "description": "Recent graduates, interns, junior positions", 
                "weights": {"skills": 35, "experience": 15, "education": 40, "certifications": 10}
            },
            "specialized": {
                "name": "Specialized Roles",
                "description": "Roles requiring specific certifications",
                "weights": {"skills": 30, "experience": 25, "education": 20, "certifications": 25}
            },
            "healthcare": {
                "name": "Healthcare Roles",
                "description": "Medical professionals, healthcare workers",
                "weights": {"skills": 20, "experience": 30, "education": 30, "certifications": 20}
            },
            "custom": {
                "name": "Custom Configuration",
                "description": "Define your own weights",
                "weights": {"skills": 40, "experience": 30, "education": 15, "certifications": 15}
            }
        }
    
    @staticmethod
    def validate_weights(weights_dict):
        required_components = ["skills", "experience", "education", "certifications"]
        
        if not all(comp in weights_dict for comp in required_components):
            missing = [comp for comp in required_components if comp not in weights_dict]
            return False, f"Missing components: {', '.join(missing)}"
        
        try:
            values = []
            for comp in required_components:
                val = float(weights_dict[comp])
                if val < 0 or val > 100:
                    return False, f"{comp} must be between 0 and 100"
                values.append(val)
        except (ValueError, TypeError):
            return False, "All weights must be numeric"
        
        total = sum(values)
        if abs(total - 100) > 0.1:
            return False, f"Weights must sum to 100, got {total:.1f}"
        
        return True, "Valid weights"
    
    @staticmethod
    def normalize_weights(weights_dict):
        try:
            total = sum(float(v) for v in weights_dict.values())
            if total == 0:
                return {"skills": 40, "experience": 30, "education": 15, "certifications": 15}
            normalized = {k: round((float(v) / total) * 100, 1) for k, v in weights_dict.items()}
            # Ensure exactly 100
            actual_total = sum(normalized.values())
            if abs(actual_total - 100) > 0.1:
                largest_key = max(normalized.keys(), key=lambda k: normalized[k])
                normalized[largest_key] = round(normalized[largest_key] + (100 - actual_total), 1)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing weights: {e}")
            return {"skills": 40, "experience": 30, "education": 15, "certifications": 15}

@job_post.route("/Company_Job_Posting")
def Company_Job_Posting():
    """HR dashboard - show only jobs created by current HR user"""
    try:
        # Security check
        if 'user_id' not in session or session.get('role') != 'employer':
            flash("Access denied. Employer login required.", "error")
            return redirect(url_for('loginpage'))
        
        current_hr_user_id = session['user_id']
        
        auto_close_expired_jobs()
        
        logger.info(f"Company_Job_Posting - HR User: {current_hr_user_id}")
        
        # CRITICAL FIX: Filter jobs by created_by
        job_filter = {"created_by": ObjectId(current_hr_user_id)}
        
        fetched_jobs = JOBS.find(job_filter, {
            "_id": 1, "Job_Profile": 1, "CompanyName": 1, "CreatedAt": 1,
            "Job_description_file_name": 1, "LastDate": 1, "Salary": 1,
            "CompanyLogo": 1, "Status": 1, "ScoringWeights": 1, "RoleTemplate": 1,
            "created_by": 1  # Include for verification
        }).sort([("CreatedAt", -1)])

        jobs = {}
        cnt = 0
        
        for job_doc in fetched_jobs:
            try:
                # Verify ownership (extra security)
                if str(job_doc.get('created_by')) != current_hr_user_id:
                    logger.warning(f"Job {job_doc['_id']} has wrong created_by, skipping")
                    continue
                
                last_date = job_doc.get('LastDate')
                if isinstance(last_date, datetime):
                    last_date_str = last_date.strftime("%Y-%m-%d")
                else:
                    last_date_str = str(last_date) if last_date else "N/A"

                jobs[cnt] = {
                    "job_id": str(job_doc['_id']),
                    "Job_Profile": job_doc.get('Job_Profile', 'N/A'),
                    "CompanyName": job_doc.get('CompanyName', 'N/A'),
                    "CreatedAt": job_doc.get('CreatedAt', datetime.now()),
                    "Job_description_file_name": job_doc.get('Job_description_file_name', 'N/A'),
                    "LastDate": last_date_str,
                    "Salary": job_doc.get('Salary', 'N/A'),
                    "CompanyLogo": job_doc.get('CompanyLogo'),
                    "Status": job_doc.get("Status", "Open"),
                    "ScoringWeights": job_doc.get('ScoringWeights', {}),
                    "RoleTemplate": job_doc.get('RoleTemplate', 'custom')
                }
                cnt += 1
            except Exception as e:
                logger.error(f"Error processing job {job_doc.get('_id')}: {e}")
                continue
        
        logger.info(f"Showing {cnt} jobs for HR user {current_hr_user_id}")
        
        # If no jobs found, show helpful message
        if cnt == 0:
            logger.info("No jobs found for this HR user")
            flash("You haven't posted any jobs yet. Create your first job posting!", "info")

        role_templates = WeightConfig.get_role_templates()
        
        return render_template(
            "Company_Job_Posting.html", 
            len=len(jobs), 
            data=jobs,
            role_templates=role_templates
        )
        
    except Exception as e:
        logger.error(f"Error in Company_Job_Posting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        flash("Error loading job postings", "error")
        return render_template("Company_Job_Posting.html", 
                             len=0, 
                             data={},
                             role_templates=WeightConfig.get_role_templates(),
                             errorMsg="Error loading jobs")



@job_post.route("/add_job", methods=["POST"])
def ADD_JOB():
    try:
        logger.info("Starting job upload process")

        # Get form data
        file = request.files.get('jd')
        job_profile = request.form.get('jp', '').strip()
        company = request.form.get('company', '').strip()
        last_date = request.form.get('last_date', '').strip()
        salary = request.form.get('salary', '').strip()
        logo_file = request.files.get('company_logo')

        # Validate input
        validation_errors = validate_job_data(job_profile, company, last_date, salary)
        if validation_errors:
            return render_template("Company_Job_Posting.html", errorMsg="; ".join(validation_errors))

        # Enhanced file validation
        if not file or file.filename == '':
            raise ValueError("Job Description file is required")
        if not allowedExtension(file.filename):
            raise ValueError("Invalid file format. Only PDF and DOCX allowed")
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise ValueError("File too large. Maximum size is 10MB")
        if file_size < 100:
            raise ValueError("File appears empty or corrupted")

        # Save file
        filename = secure_filename(file.filename)
        jd_id = ObjectId()
        path = os.path.join(UF, str(jd_id))
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        
        try:
            file.save(file_path)
        except Exception as e:
            raise ValueError("Failed to save uploaded file")

        # Extract content
        file_extension = filename.rsplit('.', 1)[1].lower()
        try:
            fetchedData = extractData(file_path, file_extension)
        except Exception as e:
            # Clean up on failure
            try:
                os.remove(file_path)
                os.rmdir(path)
            except:
                pass
            raise ValueError(f"Failed to extract content: {str(e)}")

        # Handle logo
        logo_filename = None
        if logo_file and logo_file.filename and allowedLogo(logo_file.filename):
            try:
                logo_file.seek(0, os.SEEK_END)
                logo_size = logo_file.tell()
                logo_file.seek(0)
                
                if logo_size <= 2 * 1024 * 1024:  # 2MB limit
                    logo_filename = f"{jd_id}_{secure_filename(logo_file.filename)}"
                    logo_path = os.path.join(LOGO_FOLDER, logo_filename)
                    logo_file.save(logo_path)
                    logger.info(f"Logo saved: {logo_filename}")
            except Exception as e:
                logger.warning(f"Logo upload failed: {e}")

        # Process weights (YOUR CUSTOMIZATION STILL WORKS!)
        weights_json = request.form.get('scoring_weights')
        role_template = request.form.get('role_template', 'custom')

        if weights_json:
            try:
                custom_weights = json.loads(weights_json)
                is_valid, message = WeightConfig.validate_weights(custom_weights)
                if not is_valid:
                    logger.warning(f"Invalid weights: {message}, using template")
                    template = WeightConfig.get_role_templates().get(role_template, 
                             WeightConfig.get_role_templates()['custom'])
                    custom_weights = template['weights']
                else:
                    custom_weights = WeightConfig.normalize_weights(custom_weights)
            except json.JSONDecodeError:
                template = WeightConfig.get_role_templates().get(role_template,
                         WeightConfig.get_role_templates()['custom'])
                custom_weights = template['weights']
        else:
            template = WeightConfig.get_role_templates().get(role_template,
                     WeightConfig.get_role_templates()['custom'])
            custom_weights = template['weights']

        # Create job document
        job_document = {
            "_id": jd_id,
            "Job_Profile": job_profile,
            "Job_Description": fetchedData,
            "CompanyName": company,
            "LastDate": datetime.strptime(last_date, "%Y-%m-%d"),
            "CreatedAt": datetime.now(),
            "Job_description_file_name": filename,
            "Salary": salary,
            "CompanyLogo": logo_filename,
            "Status": "Open",
            "ScoringWeights": custom_weights,
            "RoleTemplate": role_template,
            "FileSize": file_size,
            "TextLength": len(fetchedData),
            "Version": "2.0",
            "created_by": ObjectId(session['user_id'])  # ← ADD THIS CRITICAL LINE
        }

        # Save to database
        JOBS.insert_one(job_document)

        # Store file binary data
        try:
            with open(file_path, "rb") as f:
                jd_data = f.read()
            JOBS.update_one({"_id": jd_id}, {"$set": {"FileData": jd_data}})
        except Exception as e:
            logger.error(f"Failed to store file binary: {e}")

        logger.info(f"Job added successfully with ID: {jd_id}")
        flash("Job posted successfully!", "success")
        return redirect('/HR1/Company_Job_Posting')

    except Exception as e:
        logger.error(f"Error in ADD_JOB: {e}")
        error_msg = str(e) if str(e) else "An unexpected error occurred"
        flash(f"Error: {error_msg}", "error")
        return render_template("Company_Job_Posting.html", errorMsg=error_msg)
    

@job_post.route("/apply_job_enhanced", methods=["POST"])
def APPLY_JOB_ENHANCED():
    """
    Enhanced job application with detailed logging
    You can see exactly what's happening at each step
    """
    
    start_time = time.time()
    
    try:
        print("\n" + "="*80)
        print("🚀 APPLICATION PROCESSING STARTED")
        print("="*80)
        
        # ===== STEP 1: Validate Inputs =====
        job_id = request.form.get('job_id', '').strip()
        resume_id = request.form.get('resume_id', '').strip()
        user_id = session.get('user_id')
        user_name = session.get('user_name')
        
        print(f"\n📋 INPUT VALIDATION:")
        print(f"   Job ID: {job_id}")
        print(f"   Resume ID: {resume_id}")
        print(f"   User ID: {user_id}")
        print(f"   User Name: {user_name}")
        
        if not all([job_id, resume_id, user_id, user_name]):
            print("❌ VALIDATION FAILED: Missing required fields")
            return jsonify({"StatusCode": 400, "Message": "Missing required fields"})

        # Validate ObjectIds
        try:
            job_obj_id = ObjectId(job_id)
            resume_obj_id = ObjectId(resume_id)
            user_obj_id = ObjectId(user_id)
            print("✅ ObjectId validation passed")
        except:
            print("❌ VALIDATION FAILED: Invalid ObjectId format")
            return jsonify({"StatusCode": 400, "Message": "Invalid ID format"})

        # ===== STEP 2: Get Job Details =====
        print(f"\n📄 FETCHING JOB DETAILS...")
        job = JOBS.find_one({"_id": job_obj_id})
        if not job:
            print("❌ Job not found in database")
            return jsonify({"StatusCode": 404, "Message": "Job not found"})
        
        print(f"✅ Job found: {job.get('Job_Profile', 'Unknown Title')}")
        print(f"   Status: {job.get('Status', 'Unknown')}")
        print(f"   Created by: {job.get('created_by', 'Unknown')}")
        
        if job.get('Status') != 'Open':
            print("❌ Job is not open for applications")
            return jsonify({"StatusCode": 400, "Message": "Job no longer accepting applications"})

        # Get custom weights
        custom_weights = job.get('ScoringWeights', {
            "skills": 40, "experience": 30, "education": 15, "certifications": 15
        })
        
        print(f"\n⚖️  SCORING WEIGHTS:")
        print(f"   Skills: {custom_weights.get('skills', 40)}%")
        print(f"   Experience: {custom_weights.get('experience', 30)}%")
        print(f"   Education: {custom_weights.get('education', 15)}%")
        print(f"   Certifications: {custom_weights.get('certifications', 15)}%")

        # ===== STEP 3: Traditional Matching (Matching.py) =====
        print("\n" + "="*80)
        print("📊 TRADITIONAL MATCHING (Matching.py)")
        print("="*80)
        
        from Matching import ProductionMatching
        
        traditional_start = time.time()
        traditional_result = ProductionMatching(
            job_id=job_id,
            resume_id=resume_id,
            skill_weight=custom_weights.get('skills', 40),
            experience_weight=custom_weights.get('experience', 30),
            education_weight=custom_weights.get('education', 15),
            certification_weight=custom_weights.get('certifications', 15)
        )
        traditional_time = (time.time() - traditional_start) * 1000
        
        # Extract scores
        traditional_score = float(traditional_result.get('overall_score', 0))
        component_scores = traditional_result.get('component_scores', {})
        
        print(f"\n✅ TRADITIONAL MATCHING COMPLETE ({traditional_time:.2f}ms)")
        print(f"   Overall Score: {traditional_score:.2f}%")
        print(f"\n   Component Breakdown:")
        print(f"   ├─ Skills: {component_scores.get('skills', 0):.2f}%")
        print(f"   ├─ Experience: {component_scores.get('experience', 0):.2f}%")
        print(f"   ├─ Education: {component_scores.get('education', 0):.2f}%")
        print(f"   └─ Certifications: {component_scores.get('certifications', 0):.2f}%")
        
        # Show detailed analysis if available
        if 'detailed_analysis' in traditional_result:
            print(f"\n   Detailed Analysis:")
            for analysis in traditional_result['detailed_analysis']:
                print(f"   • {analysis.get('component', 'Unknown').title()}: {analysis.get('score', 0):.2f}%")
                if analysis.get('strengths'):
                    print(f"     Strengths: {', '.join(analysis['strengths'])}")
                if analysis.get('weaknesses'):
                    print(f"     Weaknesses: {', '.join(analysis['weaknesses'])}")

       


        # ===== STEP 4: Semantic Matching (sentence_transformer_ranker.py) =====
        print("\n" + "="*80)
        print("🤖 SEMANTIC MATCHING (AI-Powered)")
        print("="*80)
        
        semantic_score = 0.0
        semantic_similarity_raw = 0.0
        semantic_available = False
        job_text_extracted = ""
        resume_text_extracted = ""
        
        try:
            from sentence_transformer_ranker import get_semantic_ranker
            
            ranker = get_semantic_ranker()
            
            if ranker.is_available():
                print(f"✅ Semantic ranker loaded: {ranker.model_name}")
                
                # Get resume document
                resume_doc = resumeFetchedData.find_one({"_id": resume_obj_id})
                
                if resume_doc:
                    print(f"✅ Resume document found: {resume_doc.get('Name', 'Unknown')}")
                    
                    semantic_start = time.time()
                    
                    # Extract job requirements
                    print(f"\n📝 EXTRACTING JOB REQUIREMENTS...")
                    job_text_extracted = ranker.extract_job_requirements(job)
                    print(f"   Extracted {len(job_text_extracted)} characters from job description")
                    print(f"\n   Job Text Preview (first 500 chars):")
                    print(f"   {'-'*76}")
                    print(f"   {job_text_extracted[:500]}...")
                    print(f"   {'-'*76}")
                    
                    # Extract resume content
                    print(f"\n📄 EXTRACTING RESUME CONTENT...")
                    resume_text_extracted = ranker.extract_resume_content(resume_doc)
                    print(f"   Extracted {len(resume_text_extracted)} characters from resume")
                    print(f"\n   Resume Text Preview (first 500 chars):")
                    print(f"   {'-'*76}")
                    print(f"   {resume_text_extracted[:500]}...")
                    print(f"   {'-'*76}")
                    
                    # Calculate semantic similarity
                    print(f"\n🧠 CALCULATING SEMANTIC SIMILARITY...")
                    print(f"   Converting texts to embeddings...")
                    semantic_similarity_raw = ranker.calculate_semantic_similarity(
                        job_text_extracted, 
                        resume_text_extracted
                    )
                    
                    if semantic_similarity_raw is None:
                        semantic_similarity_raw = 0.0
                        print(f"   ⚠️  Similarity calculation returned None, using 0.0")
                    
                    # Convert to percentage
                    semantic_score = semantic_similarity_raw * 100
                    semantic_available = True
                    
                    semantic_time = (time.time() - semantic_start) * 1000
                    
                    print(f"\n✅ SEMANTIC MATCHING COMPLETE ({semantic_time:.2f}ms)")
                    print(f"   Raw Similarity: {semantic_similarity_raw:.4f} (0-1 scale)")
                    print(f"   Percentage: {semantic_score:.2f}%")
                    
                    # Interpretation
                    if semantic_score >= 80:
                        interpretation = "Excellent semantic match!"
                    elif semantic_score >= 70:
                        interpretation = "Strong semantic alignment"
                    elif semantic_score >= 60:
                        interpretation = "Good semantic similarity"
                    elif semantic_score >= 50:
                        interpretation = "Moderate semantic match"
                    else:
                        interpretation = "Limited semantic alignment"
                    
                    print(f"   Interpretation: {interpretation}")
                    
                else:
                    print("❌ Resume document not found in database")
            else:
                print("⚠️  Semantic ranker not available (model not loaded)")
                
        except ImportError as e:
            print(f"⚠️  Semantic ranker module not installed: {e}")
        except Exception as e:
            print(f"❌ Semantic scoring failed: {e}")
            import traceback
            print(f"\n   Full Error Trace:")
            print(f"   {'-'*76}")
            traceback.print_exc()
            print(f"   {'-'*76}")

        # ===== STEP 5: Calculate Combined Score =====
        print("\n" + "="*80)
        print("🔄 CALCULATING COMBINED SCORE")
        print("="*80)
        
        TRADITIONAL_WEIGHT = 0.40  # 60%
        SEMANTIC_WEIGHT = 0.60    # 40%
        
        print(f"\n   Scoring Method:")
        if semantic_available and semantic_score > 0:
            print(f"   ✅ Hybrid Scoring (Traditional + Semantic)")
            print(f"\n   Calculation:")
            print(f"   Combined = (Traditional × {TRADITIONAL_WEIGHT}) + (Semantic × {SEMANTIC_WEIGHT})")
            print(f"   Combined = ({traditional_score:.2f} × {TRADITIONAL_WEIGHT}) + ({semantic_score:.2f} × {SEMANTIC_WEIGHT})")
            
            combined_score = (traditional_score * TRADITIONAL_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT)
            
            print(f"   Combined = {traditional_score * TRADITIONAL_WEIGHT:.2f} + {semantic_score * SEMANTIC_WEIGHT:.2f}")
            print(f"   Combined = {combined_score:.2f}%")
            
            scoring_method = "hybrid"
            
            # Show improvement/difference
            diff = combined_score - traditional_score
            if diff > 0:
                print(f"\n   📈 Semantic AI improved score by {diff:.2f} points!")
            elif diff < 0:
                print(f"\n   📉 Semantic AI lowered score by {abs(diff):.2f} points")
            else:
                print(f"\n   ➡️  Semantic AI confirmed traditional score")
                
        else:
            print(f"   ℹ️  Traditional Only (Semantic not available)")
            combined_score = traditional_score
            scoring_method = "traditional_only"
            print(f"\n   Combined Score = {combined_score:.2f}% (same as traditional)")

        # ===== STEP 6: Check Existing Application =====
        print(f"\n📌 CHECKING FOR EXISTING APPLICATION...")
        existing = Applied_EMP.find_one({
            "job_id": job_obj_id,
            "user_id": user_obj_id
        })
        
        if existing:
            print(f"   ⚠️  Found existing application (will update)")
            print(f"   Previous score: {existing.get('Matching_percentage', 'N/A')}")
        else:
            print(f"   ✅ New application (will create)")

        # ===== STEP 7: Store Application Data =====
        print(f"\n💾 SAVING TO DATABASE...")
        
        application_data = {
            "job_id": job_obj_id,
            "user_id": user_obj_id,
            "resume_id": resume_obj_id,
            "User_name": user_name,
            
            # Scoring data
            "Matching_percentage": {
                "overall_score": traditional_score,
                "component_scores": component_scores  # ✅ Store breakdown here
                
            },
            "semantic_similarity": semantic_similarity_raw if semantic_available else None,
            "combined_score": combined_score,
            "scoring_method": scoring_method,
            
            # Detailed breakdown
            "component_scores": component_scores,
            "semantic_available": semantic_available,
            
            "weights_used": {
                "traditional": TRADITIONAL_WEIGHT * 100,
                "semantic": SEMANTIC_WEIGHT * 100 if semantic_available else 0,
                "component_weights": custom_weights

            
            },

            
            
            # Metadata
            "applied_at": datetime.now(),
            "status": "pending",
            "system_version": "3.0_semantic_verified",
            "enhanced_version": "sentence_transformer_v2" if semantic_available else None,
            
            # Debug info (optional - for troubleshooting)
            "debug_info": {
                "job_text_length": len(job_text_extracted),
                "resume_text_length": len(resume_text_extracted),
                "traditional_time_ms": round(traditional_time, 2),
                "semantic_time_ms": round((semantic_time if semantic_available else 0), 2)
            }
        }

        try:
            if existing:
                Applied_EMP.update_one(
                    {"_id": existing["_id"]},
                    {"$set": application_data}
                )
                action = "updated"
                print(f"   ✅ Application updated successfully")
            else:
                Applied_EMP.insert_one(application_data)
                action = "created"
                print(f"   ✅ Application created successfully")
                
                # Create notification if available
                if NOTIFICATIONS_AVAILABLE:
                    try:
                        create_application_notification(
                            applicant_id=user_id,
                            hr_id=str(job.get('created_by')),
                            job_id=job_id,
                            job_title=job.get('Job_Profile', 'Unknown Job'),
                            applicant_name=user_name
                        )
                        print(f"   ✅ Notification sent to HR")
                    except Exception as e:
                        print(f"   ⚠️  Notification failed: {e}")

            # ===== STEP 8: Build Response =====
            total_time = round((time.time() - start_time) * 1000, 2)
            
            print("\n" + "="*80)
            print("✅ APPLICATION PROCESSING COMPLETE")
            print("="*80)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   Traditional Score: {traditional_score:.2f}%")
            if semantic_available:
                print(f"   Semantic Score: {semantic_score:.2f}%")
            print(f"   Combined Score: {combined_score:.2f}%")
            print(f"   Processing Time: {total_time:.2f}ms")
            print(f"   Action: {action.upper()}")
            print("\n" + "="*80 + "\n")
            
            response_data = {
                "StatusCode": 200,
                "Message": f"Application {action} successfully",
                
                # Scores
                "TraditionalScore": round(traditional_score, 2),
                "CombinedScore": round(combined_score, 2),
                "ScoringMethod": scoring_method,
                
                # Component breakdown
                "ComponentScores": {
                    "Skills": round(component_scores.get('skills', 0), 2),
                    "Experience": round(component_scores.get('experience', 0), 2),
                    "Education": round(component_scores.get('education', 0), 2),
                    "Certifications": round(component_scores.get('certifications', 0), 2)
                },
                
                # Performance
                "ProcessingTime": f"{total_time}ms",
                "Timestamp": datetime.now().isoformat()
            }
            
            # Add semantic data if available
            if semantic_available:
                response_data["SemanticScore"] = round(semantic_score, 2)
                response_data["SemanticSimilarity"] = round(semantic_similarity_raw, 4)
                response_data["AIEnhanced"] = True
                response_data["ScoringWeights"] = {
                    "Traditional": f"{TRADITIONAL_WEIGHT*100:.0f}%",
                    "Semantic": f"{SEMANTIC_WEIGHT*100:.0f}%"
                }
                response_data["ExtractionInfo"] = {
                    "JobTextLength": len(job_text_extracted),
                    "ResumeTextLength": len(resume_text_extracted)
                }
            else:
                response_data["AIEnhanced"] = False
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"\n❌ DATABASE ERROR:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"StatusCode": 500, "Message": "Failed to save application"})

    except Exception as e:
        total_time = round((time.time() - start_time) * 1000, 2)
        
        print("\n" + "="*80)
        print("❌ CRITICAL ERROR")
        print("="*80)
        print(f"   Error: {e}")
        print(f"   Processing Time: {total_time}ms")
        print("\n   Full Trace:")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        
        return jsonify({
            "StatusCode": 500,
            "Message": "Application processing error",
            "Error": str(e),
            "ProcessingTime": f"{total_time}ms",
            "Timestamp": datetime.now().isoformat()
        })





@job_post.route("/apply_job", methods=["POST"])
def APPLY_JOB():
    """Enhanced job application using ProductionMatching"""
    start_time = time.time()
    
    try:
        # Validate inputs
        job_id = request.form.get('job_id', '').strip()
        resume_id = request.form.get('resume_id', '').strip()
        user_id = session.get('user_id')
        user_name = session.get('user_name')

        if not all([job_id, resume_id, user_id, user_name]):
            return jsonify({"StatusCode": 400, "Message": "Missing required fields"})

        try:
            ObjectId(job_id)
            ObjectId(resume_id)
            ObjectId(user_id)
        except:
            return jsonify({"StatusCode": 400, "Message": "Invalid ID format"})

        logger.info(f"Processing application: user={user_id}, job={job_id}, resume={resume_id}")

        # Check existing application
        existing_application = Applied_EMP.find_one({
            "job_id": ObjectId(job_id),
            "user_id": ObjectId(user_id)
        })

        # Get job and weights
        job = JOBS.find_one({"_id": ObjectId(job_id)})
        if not job:
            return jsonify({"StatusCode": 404, "Message": "Job not found"})
        
        if job.get('Status') != 'Open':
            return jsonify({"StatusCode": 400, "Message": "Job no longer accepting applications"})

        # Get job weights (YOUR CUSTOMIZATION WEIGHTS!)
        job_weights = job.get('ScoringWeights', {
            "skills": 40, "experience": 30, "education": 15, "certifications": 15
        })

        logger.info(f"Using job weights: {job_weights}")

        # IMPORTANT: Use ProductionMatching (enhanced version)
        try:
            match_result = ProductionMatching(
                job_id,
                resume_id,
                skill_weight=int(job_weights.get("skills", 40)),
                experience_weight=int(job_weights.get("experience", 30)),
                education_weight=int(job_weights.get("education", 15)),
                certification_weight=int(job_weights.get("certifications", 15))
            )
            logger.info("Production matching completed successfully")
        except Exception as e:
            logger.error(f"Production matching failed: {e}")
            match_result = {
                "overall_score": 0.0,
                "error": str(e),
                "component_scores": {"skills": 0, "experience": 0, "education": 0, "certifications": 0}
            }

        # Extract results
        match_percentage = 0.0
        component_scores = {}
        weights_used = {}
        detailed_analysis = []

        if isinstance(match_result, dict):
            try:
                match_percentage = float(match_result.get("overall_score", 0))
                component_scores = match_result.get("component_scores", {})
                weights_used = match_result.get("weights_used", job_weights)
                detailed_analysis = match_result.get("detailed_analysis", [])
                
                if "error" in match_result:
                    logger.warning(f"Matching error: {match_result['error']}")
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Error extracting match data: {e}")
                match_percentage = 0.0

        # Enhanced logging
        logger.info(f"=== MATCHING RESULTS ===")
        logger.info(f"Overall Score: {match_percentage:.2f}%")
        if component_scores and weights_used:
            for component, score in component_scores.items():
                weight = weights_used.get(component, 0)
                logger.info(f"{component.capitalize()}: {score:.1f}% (weight: {weight:.1f}%)")

        # Store application
        application_data = {
            "job_id": ObjectId(job_id),
            "user_id": ObjectId(user_id),
            "resume_id": ObjectId(resume_id),
            "User_name": user_name,
            "Matching_percentage": match_percentage,
            "applied_at": datetime.now(),
            "weights_used": weights_used,
            "component_scores": component_scores,
            "detailed_analysis": detailed_analysis,
            "matching_version": "2.0",
            "job_status_at_application": job.get('Status', 'Open')
        }

        try:
            if existing_application:
                Applied_EMP.update_one(
                    {"_id": existing_application["_id"]},
                    {"$set": application_data}
                )
                logger.info(f"Updated application for user {user_id}")
                action_message = "Application updated successfully"
            else:
                Applied_EMP.insert_one(application_data)
                logger.info(f"Created new application for user {user_id}")
                action_message = "Applied successfully"

            total_time = round((time.time() - start_time) * 1000, 2)
            
            return jsonify({
                "StatusCode": 200,
                "Message": action_message,
                "MatchingScore": round(match_percentage, 2),
                "ProcessingTime": f"{total_time}ms",
                "ComponentScores": component_scores,
                "Timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Database error: {e}")
            return jsonify({"StatusCode": 500, "Message": "Failed to save application"})

    except Exception as e:
        total_time = round((time.time() - start_time) * 1000, 2)
        logger.error(f"Error in APPLY_JOB: {e}", exc_info=True)
        return jsonify({
            "StatusCode": 500,
            "Message": "Application processing error",
            "ProcessingTime": f"{total_time}ms"
        })




# Auto-close expired jobs
def auto_close_expired_jobs():
    try:
        today = datetime.now()
        result = JOBS.update_many(
            {'LastDate': {'$lt': today}, 'Status': 'Open'},
            {'$set': {'Status': 'Closed', 'AutoClosedAt': datetime.now()}}
        )
        if result.modified_count > 0:
            logger.info(f"Auto-closed {result.modified_count} expired jobs")
    except Exception as e:
        logger.error(f"Error auto-closing jobs: {e}")

# Initialize scheduler
try:
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=auto_close_expired_jobs, trigger="interval", hours=1)
    scheduler.start()
    logger.info("Job scheduler started")
except Exception as e:
    logger.error(f"Failed to start scheduler: {e}")

@job_post.route("/show_job")
def show_job():
    try:
        user_id = session.get('user_id')
        user_role = session.get('role')  # Get user role to determine filtering
        auto_close_expired_jobs()

        if not user_id:
            flash("Please log in", "error")
            return redirect(url_for('loginpage'))
        
        # Different filtering logic based on user role
        if user_role == 'employer':
            # HR users see only their own jobs
            job_filter = {"created_by": ObjectId(user_id)}
        else:
            # Applicants see all open jobs
            job_filter = {"Status": "Open"}
            
        fetched_jobs = JOBS.find(
            job_filter,
            {
                "_id": 1, "Job_Profile": 1, "CompanyName": 1, "CreatedAt": 1,
                "Job_description_file_name": 1, "LastDate": 1, "Salary": 1,
                "CompanyLogo": 1, "Status": 1, "ScoringWeights": 1, "RoleTemplate": 1
            }
        ).sort([("CreatedAt", -1)])

        jobs = {}
        applied_job_ids = set()

        # Get applied jobs for applicants only
        if user_role != 'employer':
            try:
                applied_jobs = Applied_EMP.find({"user_id": ObjectId(user_id)}, {"job_id": 1})
                applied_job_ids = set(str(app['job_id']) for app in applied_jobs)
            except Exception as e:
                logger.error(f"Error fetching applied jobs: {e}")

        cnt = 0
        for job_doc in fetched_jobs:
            try:
                last_date = job_doc.get('LastDate')
                if isinstance(last_date, datetime):
                    last_date_str = last_date.strftime("%Y-%m-%d")
                    # For applicants, skip expired jobs
                    if user_role != 'employer' and last_date < datetime.now():
                        continue
                else:
                    last_date_str = str(last_date) if last_date else "N/A"

                jobs[cnt] = {
                    "job_id": str(job_doc['_id']),
                    "Job_Profile": job_doc.get('Job_Profile', 'N/A'),
                    "CompanyName": job_doc.get('CompanyName', 'N/A'),
                    "CreatedAt": job_doc.get('CreatedAt', datetime.now()),
                    "Job_description_file_name": job_doc.get('Job_description_file_name', 'N/A'),
                    "LastDate": last_date_str,
                    "Salary": job_doc.get('Salary', 'N/A'),
                    "CompanyLogo": job_doc.get("CompanyLogo"),
                    "Status": job_doc.get("Status", "Open"),
                    "RoleTemplate": job_doc.get("RoleTemplate", "custom"),
                    "IsApplied": str(job_doc['_id']) in applied_job_ids
                }
                cnt += 1
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                continue

        return render_template("All_jobs.html", len=len(jobs), data=jobs, applied_job_ids=applied_job_ids)

    except Exception as e:
        logger.error(f"Error in show_job: {e}")
        return render_template("All_jobs.html", len=0, data={}, applied_job_ids=set(), 
                             errorMsg="Error loading jobs")

# API Endpoints
@job_post.route("/get_role_templates", methods=["GET"])
def get_role_templates():
    try:
        templates = WeightConfig.get_role_templates()
        return jsonify({"success": True, "templates": templates, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error fetching templates: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@job_post.route("/validate_weights", methods=["POST"])
def validate_weights():
    try:
        weights = request.get_json()
        if not weights:
            return jsonify({"success": False, "message": "No weight data provided"}), 400
        
        is_valid, message = WeightConfig.validate_weights(weights)
        
        if is_valid:
            normalized = WeightConfig.normalize_weights(weights)
            return jsonify({
                "success": True, 
                "weights": normalized, 
                "message": message,
                "total": sum(normalized.values())
            })
        else:
            return jsonify({"success": False, "message": message}), 400
            
    except Exception as e:
        logger.error(f"Error validating weights: {e}")
        return jsonify({"success": False, "message": str(e)}), 500



@job_post.route("/view_applied_candidates", methods=["POST", "GET"])
def view_applied_candidates():
    try:
        job_id = request.form.get('job_id', '').strip()
        
        if not job_id:
            return jsonify({"StatusCode": 400, "Message": "Job ID is required"})

        try:
            ObjectId(job_id)
        except:
            return jsonify({"StatusCode": 400, "Message": "Invalid job ID format"})

        logger.info(f"Fetching candidates for job {job_id}")

        result_data = Applied_EMP.find(
            {"job_id": ObjectId(job_id)},
            {"User_name": 1, "Matching_percentage": 1, "user_id": 1, "applied_at": 1}
        ).sort([("Matching_percentage", -1)])

        if not result_data:
            return jsonify({"StatusCode": 200, "count": 0, "data": [], "Message": "No candidates found"})

        response_list = []
        for candidate in result_data:
            try:
                raw_match = candidate.get('Matching_percentage', 0)
                
                # Normalize match percentage
                match_value = 0.0
                if isinstance(raw_match, (int, float)):
                    match_value = float(raw_match)
                elif isinstance(raw_match, dict):
                    # Try common keys
                    for key in ["overall_score", "match_percentage", "score"]:
                        if key in raw_match and isinstance(raw_match[key], (int, float)):
                            match_value = float(raw_match[key])
                            break
                else:
                    try:
                        match_value = float(str(raw_match))
                    except (ValueError, TypeError):
                        match_value = 0.0

                response_list.append({
                    "Name": candidate.get('User_name', 'Unknown'),
                    "Match": round(match_value, 2),
                    "user_id": str(candidate.get('user_id', '')),
                    "applied_at": candidate.get('applied_at', '')
                })
                
            except Exception as e:
                logger.error(f"Error processing candidate {candidate.get('user_id')}: {e}")
                continue

        logger.info(f"Found {len(response_list)} candidates for job {job_id}")
        return jsonify({"StatusCode": 200, "count": len(response_list), "data": response_list})

    except Exception as e:
        logger.error(f"Error in view_applied_candidates: {e}")
        return jsonify({"StatusCode": 500, "Message": "Error fetching candidates"})

@job_post.route("/update_job", methods=["POST"])
def update_job():
    """Update job - SECURITY: Only allow HR to update their own jobs"""
    try:
        # Security check
        if 'user_id' not in session or session.get('role') != 'employer':
            return jsonify(success=False, message="Access denied"), 403
        
        current_hr_user_id = session['user_id']
        
        job_id = request.form.get("job_id")
        if not job_id:
            raise ValueError("Missing Job ID")
        
        # CRITICAL: Verify job belongs to current HR user
        job = JOBS.find_one({
            "_id": ObjectId(job_id),
            "created_by": ObjectId(current_hr_user_id)
        })
        
        if not job:
            logger.warning(f"HR {current_hr_user_id} attempted to update job {job_id} they don't own")
            return jsonify(success=False, message="Job not found or access denied"), 403
        
        logger.info(f"Updating job {job_id} by HR {current_hr_user_id}")

        # Get form data
        file = request.files.get('jd')
        job_profile = request.form.get('jp', '').strip()
        company = request.form.get('company', '').strip()
        last_date = request.form.get('last_date', '').strip()
        salary = request.form.get('salary', '').strip()
        status = request.form.get('status', 'Open').strip()
        logo_file = request.files.get('company_logo')

        # Validate input
        validation_errors = validate_job_data(job_profile, company, last_date, salary)
        if validation_errors:
            return jsonify(success=False, message="; ".join(validation_errors)), 400

        update_fields = {
            "Job_Profile": job_profile,
            "CompanyName": company,
            "Salary": salary,
            "Status": status,
            "LastDate": datetime.strptime(last_date, "%Y-%m-%d"),
            "UpdatedAt": datetime.now(),
        }

        # Handle JD update
        if file and file.filename:
            if not allowedExtension(file.filename):
                raise ValueError("Invalid JD format. Only PDF and DOCX allowed")

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            if file_size > 10 * 1024 * 1024:
                raise ValueError("File too large. Max 10MB")
            if file_size < 100:
                raise ValueError("File appears empty or corrupted")

            filename = secure_filename(file.filename)
            path = os.path.join(UF, str(job["_id"]))
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, filename)

            file.save(file_path)

            file_extension = filename.rsplit('.', 1)[1].lower()
            try:
                fetchedData = extractData(file_path, file_extension)
            except Exception as e:
                raise ValueError(f"Failed to extract content: {str(e)}")

            update_fields.update({
                "Job_Description": fetchedData,
                "Job_description_file_name": filename,
                "FileSize": file_size,
                "TextLength": len(fetchedData),
            })

            # Update binary
            try:
                with open(file_path, "rb") as f:
                    jd_data = f.read()
                update_fields["FileData"] = jd_data
            except Exception as e:
                logger.warning(f"Failed to store JD binary: {e}")

        # Handle logo update
        if logo_file and logo_file.filename and allowedLogo(logo_file.filename):
            try:
                logo_file.seek(0, os.SEEK_END)
                logo_size = logo_file.tell()
                logo_file.seek(0)

                if logo_size <= 2 * 1024 * 1024:
                    logo_filename = f"{job_id}_{secure_filename(logo_file.filename)}"
                    logo_path = os.path.join(LOGO_FOLDER, logo_filename)
                    logo_file.save(logo_path)
                    update_fields["CompanyLogo"] = logo_filename
            except Exception as e:
                logger.warning(f"Logo upload failed: {e}")

        # Handle scoring weights
        weights_json = request.form.get('scoring_weights')
        role_template = request.form.get('role_template', 'custom')

        if weights_json:
            try:
                custom_weights = json.loads(weights_json)
                is_valid, message = WeightConfig.validate_weights(custom_weights)
                if not is_valid:
                    template = WeightConfig.get_role_templates().get(role_template,
                                WeightConfig.get_role_templates()['custom'])
                    custom_weights = template['weights']
                else:
                    custom_weights = WeightConfig.normalize_weights(custom_weights)
            except json.JSONDecodeError:
                template = WeightConfig.get_role_templates().get(role_template,
                            WeightConfig.get_role_templates()['custom'])
                custom_weights = template['weights']
        else:
            template = WeightConfig.get_role_templates().get(role_template,
                        WeightConfig.get_role_templates()['custom'])
            custom_weights = template['weights']

        update_fields["ScoringWeights"] = custom_weights
        update_fields["RoleTemplate"] = role_template

        # Update DB with additional security check
        result = JOBS.update_one(
            {
                "_id": ObjectId(job_id),
                "created_by": ObjectId(current_hr_user_id)  # Double-check ownership
            },
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            return jsonify(success=False, message="Job not found or access denied"), 403

        logger.info(f"Job {job_id} updated successfully by HR {current_hr_user_id}")
        return jsonify(success=True, message="Job updated successfully")

    except Exception as e:
        logger.error(f"Error in update_job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify(success=False, message=str(e)), 500


@job_post.route("/delete_job", methods=["POST"])
def delete_job():
    """Delete job - SECURITY: Only allow HR to delete their own jobs"""
    try:
        # Security check
        if 'user_id' not in session or session.get('role') != 'employer':
            return jsonify(success=False, message="Access denied"), 403
        
        current_hr_user_id = session['user_id']
        
        job_id = request.form.get("job_id", '').strip()
        if not job_id:
            return jsonify(success=False, message="Job ID required"), 400

        try:
            ObjectId(job_id)
        except:
            return jsonify(success=False, message="Invalid job ID"), 400

        # CRITICAL: Verify job belongs to current HR user
        job_data = JOBS.find_one({
            "_id": ObjectId(job_id),
            "created_by": ObjectId(current_hr_user_id)
        })
        
        if not job_data:
            logger.warning(f"HR {current_hr_user_id} attempted to delete job {job_id} they don't own")
            return jsonify(success=False, message="Job not found or access denied"), 403

        logger.info(f"Deleting job {job_id} by HR {current_hr_user_id}")

        # Delete with ownership verification
        result = JOBS.delete_one({
            "_id": ObjectId(job_id),
            "created_by": ObjectId(current_hr_user_id)
        })

        if result.deleted_count > 0:
            try:
                # Cleanup files
                cleanup_old_files(job_id)
                if job_data.get('CompanyLogo'):
                    logo_path = os.path.join(LOGO_FOLDER, job_data['CompanyLogo'])
                    if os.path.exists(logo_path):
                        os.remove(logo_path)
                
                # Delete applications
                Applied_EMP.delete_many({"job_id": ObjectId(job_id)})
                
                logger.info(f"Job {job_id} and related data deleted successfully")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            return jsonify(success=True, message="Job deleted successfully!")
        else:
            return jsonify(success=False, message="Job not found or already deleted"), 404

    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify(success=False, message="Error deleting job"), 500
    
@job_post.route("/fix_existing_jobs_once", methods=["GET"])
def fix_existing_jobs_once():
    """ONE-TIME route to fix existing jobs - REMOVE after running once"""
    try:
        if session.get('role') != 'employer':
            return "Access denied"
            
        # Find first HR user to assign existing jobs to
        first_hr = IRS_USERS.find_one({"Role": "employer"})
        if not first_hr:
            return "No HR users found"
        
        # Update jobs without created_by field
        result = JOBS.update_many(
            {"created_by": {"$exists": False}},
            {"$set": {"created_by": first_hr["_id"]}}
        )
        
        return f"SUCCESS: Fixed {result.modified_count} existing jobs. Now remove this route!"
        
    except Exception as e:
        return f"Error: {e}"

