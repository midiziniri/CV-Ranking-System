"""
ENHANCED Matching.py - WITH DETAILED EXTRACTION AND MATCHING LOGS
"""

import fitz, io, re, logging
from difflib import SequenceMatcher
from database import mongo
from bson.objectid import ObjectId
import nltk
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import threading
import time
from functools import wraps

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a separate logger for matching details
match_logger = logging.getLogger('MatchingDetails')
match_logger.setLevel(logging.INFO)

@dataclass
class MatchingConfig:
    exact_match_threshold: float = 0.85
    high_match_threshold: float = 0.75
    medium_match_threshold: float = 0.6
    certification_match_threshold: float = 0.75
    max_similarity_cache_size: int = 1000
    max_text_length_for_nlp: int = 100
    pdf_extraction_timeout: int = 30

class SimpleMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            'total_matches': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_processing_time': 0.0,
            'error_counts': {},
            'last_reset': datetime.now().isoformat()
        }
    
    def record_match_attempt(self):
        with self._lock:
            self._metrics['total_matches'] += 1
    
    def record_successful_match(self, processing_time: float, scores: dict = None):
        with self._lock:
            self._metrics['successful_matches'] += 1
            total = self._metrics['successful_matches']
            current_avg = self._metrics['avg_processing_time']
            self._metrics['avg_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
    
    def record_failed_match(self, error_type: str):
        with self._lock:
            self._metrics['failed_matches'] += 1
            if error_type not in self._metrics['error_counts']:
                self._metrics['error_counts'][error_type] = 0
            self._metrics['error_counts'][error_type] += 1
    
    def get_metrics(self) -> dict:
        with self._lock:
            metrics_copy = self._metrics.copy()
            total = metrics_copy['total_matches']
            if total > 0:
                metrics_copy['success_rate'] = metrics_copy['successful_matches'] / total
                metrics_copy['failure_rate'] = metrics_copy['failed_matches'] / total
            else:
                metrics_copy['success_rate'] = 0.0
                metrics_copy['failure_rate'] = 0.0
            return metrics_copy

class InputSanitizer:
    @staticmethod
    def validate_object_id(obj_id: str) -> bool:
        if not isinstance(obj_id, str) or len(obj_id) != 24:
            return False
        try:
            int(obj_id, 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_weights(weights: dict) -> dict:
        sanitized = {}
        valid_components = {'skills', 'experience', 'education', 'certifications'}
        for key, value in weights.items():
            if key in valid_components:
                try:
                    weight_val = float(value)
                    sanitized[key] = max(0.0, weight_val)
                except (ValueError, TypeError):
                    continue
        return sanitized

def simple_performance_monitor(metrics):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.record_match_attempt()
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                scores = {}
                if isinstance(result, dict) and 'component_scores' in result:
                    scores = result['component_scores']
                metrics.record_successful_match(processing_time, scores)
                return result
            except Exception as e:
                error_type = type(e).__name__
                metrics.record_failed_match(error_type)
                raise
        return wrapper
    return decorator

matching_config = MatchingConfig()
matching_metrics = SimpleMetrics()

def get_matching_metrics() -> dict:
    return matching_metrics.get_metrics()

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.warning(f"NLTK download failed: {e}")

try:
    resumeFetchedData = mongo.db.resumeFetchedData
    JOBS = mongo.db.JOBS
except Exception as e:
    logging.error(f"Database connection failed: {e}")
    resumeFetchedData = None
    JOBS = None

@dataclass
class MatchingWeights:
    skills: float
    experience: float
    education: float
    certifications: float
    
    def __post_init__(self):
        total = self.skills + self.experience + self.education + self.certifications
        if abs(total - 1.0) > 0.01:
            self.skills /= total
            self.experience /= total
            self.education /= total
            self.certifications /= total

@dataclass
class ScoreAnalysis:
    component: str
    score: float
    max_possible: float
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    matched_items: List[Dict]
    missing_items: List[str]

@dataclass
class MatchingResult:
    overall_score: float
    component_scores: Dict[str, float]
    detailed_analysis: List[ScoreAnalysis]
    hr_feedback: Dict[str, Any]
    candidate_feedback: Dict[str, Any]
    weights_used: MatchingWeights
    extraction_log: Dict[str, Any]
    matching_log: Dict[str, Any]

class IntelligentMatcher:
    def __init__(self):
        self.config = matching_config
        self._similarity_cache = {}
        self._cache_size_limit = self.config.max_similarity_cache_size
        self.extraction_log = {}
        self.matching_log = {}
    
    def match_resume_to_job(self, job_id: str, resume_id: str, 
                            custom_weights: Dict[str, int] = None) -> MatchingResult:
        try:
            if resumeFetchedData is None or JOBS is None:
                raise ConnectionError("Database connection not available")
            
            match_logger.info("="*80)
            match_logger.info("STARTING NEW MATCHING PROCESS")
            match_logger.info("="*80)
            
            job, resume_data = self._validate_inputs(job_id, resume_id)
            weights = self._process_custom_weights(custom_weights)
            
            # Extract and log JD requirements
            match_logger.info("\n" + "="*80)
            match_logger.info("📋 EXTRACTING JOB DESCRIPTION REQUIREMENTS")
            match_logger.info("="*80)
            jd_requirements = self._extract_jd_requirements(job)
            self._log_jd_extraction(jd_requirements)
            
            # Extract and log resume data
            match_logger.info("\n" + "="*80)
            match_logger.info("📄 EXTRACTING RESUME DATA")
            match_logger.info("="*80)
            resume_parsed = self._parse_resume_from_database(resume_data)
            self._log_resume_extraction(resume_parsed)
            
            # Calculate scores with detailed matching
            match_logger.info("\n" + "="*80)
            match_logger.info("🔍 PERFORMING DETAILED MATCHING")
            match_logger.info("="*80)
            detailed_scores = self._calculate_detailed_scores(jd_requirements, resume_parsed)
            
            analysis = self._generate_detailed_analysis(detailed_scores, jd_requirements, resume_parsed)
            final_score = self._calculate_weighted_score(detailed_scores, weights)
            hr_feedback = self._generate_hr_feedback(analysis, detailed_scores, weights)
            candidate_feedback = self._generate_candidate_feedback(analysis, detailed_scores)
            
            match_logger.info("\n" + "="*80)
            match_logger.info("✅ MATCHING COMPLETE")
            match_logger.info("="*80)
            match_logger.info(f"Final Score: {round(final_score * 100, 2)}%")

            component_scores = {}
            for k, v in detailed_scores.items():
                if v is None:
                    component_scores[k] = 0.0  # Display as 0 for not required
                else:
                    component_scores[k] = round(v * 100, 2)
            return MatchingResult(
                overall_score=round(final_score * 100, 2),
                component_scores=component_scores,
                detailed_analysis=analysis,
                hr_feedback=hr_feedback,
                candidate_feedback=candidate_feedback,
                weights_used=weights,
                extraction_log=self.extraction_log,
                matching_log=self.matching_log
            )
            
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_result(f"Input validation failed: {e}")
        except ConnectionError as e:
            logger.error(f"Database connection error: {e}")
            return self._create_error_result("System temporarily unavailable")
        except Exception as e:
            logger.critical(f"Unexpected error: {e}", exc_info=True)
            return self._create_error_result("System error occurred")

    def _log_jd_extraction(self, jd_requirements: Dict):
        """Log extracted JD requirements in detail"""
        self.extraction_log['jd'] = jd_requirements
        
        match_logger.info(f"\n📊 Skills Required: {len(jd_requirements.get('skills', []))}")
        for i, skill in enumerate(jd_requirements.get('skills', []), 1):
            match_logger.info(f"  {i}. {skill}")
        
        match_logger.info(f"\n🎓 Education Required: {len(jd_requirements.get('education', []))}")
        for i, edu in enumerate(jd_requirements.get('education', []), 1):
            match_logger.info(f"  {i}. {edu}")
        
        match_logger.info(f"\n📜 Certifications Required: {len(jd_requirements.get('certifications', []))}")
        for i, cert in enumerate(jd_requirements.get('certifications', []), 1):
            match_logger.info(f"  {i}. {cert}")
        
        match_logger.info(f"\n💼 Experience Required: {jd_requirements.get('experience', 0)} years")
    
    def _log_resume_extraction(self, resume_parsed: Dict):
        """Log extracted resume data in detail"""
        self.extraction_log['resume'] = resume_parsed
        
        match_logger.info(f"\n📊 Skills Found: {len(resume_parsed.get('skills', []))}")
        for i, skill in enumerate(resume_parsed.get('skills', []), 1):
            match_logger.info(f"  {i}. {skill}")
        
        match_logger.info(f"\n🎓 Education Found: {len(resume_parsed.get('education', []))}")
        for i, edu in enumerate(resume_parsed.get('education', []), 1):
            match_logger.info(f"  {i}. {edu}")
        
        match_logger.info(f"\n📜 Certifications Found: {len(resume_parsed.get('certifications', []))}")
        for i, cert in enumerate(resume_parsed.get('certifications', []), 1):
            match_logger.info(f"  {i}. {cert}")
        
        match_logger.info(f"\n💼 Experience Found: {resume_parsed.get('experience', 0)} years")

    def _validate_inputs(self, job_id: str, resume_id: str) -> Tuple[Dict, Dict]:
        sanitizer = InputSanitizer()
        if not sanitizer.validate_object_id(job_id) or not sanitizer.validate_object_id(resume_id):
            raise ValueError("Invalid ObjectId format")
        
        if resumeFetchedData is None or JOBS is None:
            raise ConnectionError("Database collections not available")
        
        job = JOBS.find_one({"_id": ObjectId(job_id)})
        resume = resumeFetchedData.find_one({"_id": ObjectId(resume_id)})
        
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        if resume is None:
            raise ValueError(f"Resume not found: {resume_id}")
        
        return job, resume
    
    def _process_custom_weights(self, custom_weights: Dict[str, int] = None) -> MatchingWeights:
        """
        FIXED: Properly process custom weights from job posting
        
        Input: {'skills': 70, 'experience': 5, 'education': 20, 'certifications': 5}
        Output: MatchingWeights(skills=0.70, experience=0.05, education=0.20, certifications=0.05)
        """
        logger.info("\n" + "─"*80)
        logger.info("⚙️  PROCESSING CUSTOM WEIGHTS IN MATCHER")
        logger.info("─"*80)
        
        if not custom_weights:
            logger.info("No custom weights provided, using defaults")
            return MatchingWeights(skills=0.50, experience=0.30, education=0.15, certifications=0.05)
        
        logger.info(f"Custom weights received: {custom_weights}")
        
        # Sanitize (this removes invalid entries)
        sanitized = InputSanitizer.sanitize_weights(custom_weights)
        
        if not sanitized:
            logger.warning("All weights invalid, using defaults")
            return MatchingWeights(skills=0.50, experience=0.30, education=0.15, certifications=0.05)
        
        logger.info(f"After sanitization: {sanitized}")
        
        # Calculate total
        total = sum(sanitized.values())
        logger.info(f"Total weight: {total}")
        
        if total == 0:
            logger.warning("Total is 0, using defaults")
            return MatchingWeights(skills=0.50, experience=0.30, education=0.15, certifications=0.05)
        
        # CRITICAL: Convert percentages to decimals (0-1 range)
        # Input: 70 (meaning 70%)
        # Output: 0.70 (decimal for calculation)
        weights = MatchingWeights(
            skills=sanitized.get('skills', 40) / total,
            experience=sanitized.get('experience', 30) / total,
            education=sanitized.get('education', 15) / total,
            certifications=sanitized.get('certifications', 15) / total
        )
        
        logger.info(f"\n✅ Final MatchingWeights object:")
        logger.info(f"  skills: {weights.skills:.3f} ({weights.skills * 100:.1f}%)")
        logger.info(f"  experience: {weights.experience:.3f} ({weights.experience * 100:.1f}%)")
        logger.info(f"  education: {weights.education:.3f} ({weights.education * 100:.1f}%)")
        logger.info(f"  certifications: {weights.certifications:.3f} ({weights.certifications * 100:.1f}%)")
        logger.info(f"  Total: {weights.skills + weights.experience + weights.education + weights.certifications:.3f}")
        logger.info("─"*80 + "\n")
        
        return weights
    
    def _calculate_weighted_score(self, scores: Dict[str, float], weights: MatchingWeights) -> float:
    
        try:
            cert_score = scores.get('certifications', None)
            
            # Case 1: Certifications are NEUTRAL (None) → Redistribute weight
            if cert_score is None:
                total_other_weights = weights.skills + weights.experience + weights.education
                
                if total_other_weights == 0:
                    match_logger.warning("⚠️  All weight was on certifications. Using default weights.")
                    adjusted_weights = {
                        'skills': 0.50,
                        'experience': 0.30,
                        'education': 0.20
                    }
                else:
                    # Redistribute certification weight proportionally
                    adjusted_weights = {
                        'skills': weights.skills / total_other_weights,
                        'experience': weights.experience / total_other_weights,
                        'education': weights.education / total_other_weights
                    }
                
                final_score = (
                    scores.get('skills', 0) * adjusted_weights['skills'] +
                    scores.get('experience', 0) * adjusted_weights['experience'] +
                    scores.get('education', 0) * adjusted_weights['education']
                )
                
                match_logger.info("\n⚖️  WEIGHT ADJUSTMENT (Certifications Not Required):")
                match_logger.info(f"   Original weights → Skills: {weights.skills:.1%}, Experience: {weights.experience:.1%}, Education: {weights.education:.1%}, Certs: {weights.certifications:.1%}")
                match_logger.info(f"   Adjusted weights → Skills: {adjusted_weights['skills']:.1%}, Experience: {adjusted_weights['experience']:.1%}, Education: {adjusted_weights['education']:.1%}")
                match_logger.info(f"   (Certification weight redistributed)")
            
            # Case 2: Certifications have a score (required OR bonus) → Include them!
            else:
                final_score = (
                    scores.get('skills', 0) * weights.skills +
                    scores.get('experience', 0) * weights.experience +
                    scores.get('education', 0) * weights.education +
                    cert_score * weights.certifications
                )
                
                cert_type = self.matching_log.get('certifications', {}).get('type', 'required')
                
                match_logger.info("\n⚖️  WEIGHTS APPLIED:")
                match_logger.info(f"   Skills: {weights.skills:.1%} × {scores.get('skills', 0):.2%} = {scores.get('skills', 0) * weights.skills:.2%}")
                match_logger.info(f"   Experience: {weights.experience:.1%} × {scores.get('experience', 0):.2%} = {scores.get('experience', 0) * weights.experience:.2%}")
                match_logger.info(f"   Education: {weights.education:.1%} × {scores.get('education', 0):.2%} = {scores.get('education', 0) * weights.education:.2%}")
                match_logger.info(f"   Certifications [{cert_type.upper()}]: {weights.certifications:.1%} × {cert_score:.2%} = {cert_score * weights.certifications:.2%}")
                
                if cert_type == 'bonus':
                    match_logger.info(f"   🎁 Bonus certs boosted score by {cert_score * weights.certifications:.2%}!")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return 0.0
            
    def _parse_resume_from_database(self, resume_data: Dict) -> Dict:
        """Parse resume data from database"""
        try:
            exp_value = resume_data.get('CALCULATED_EXPERIENCE_YEARS', 0)
            try:
                experience = float(exp_value) if exp_value is not None else 0.0
            except (ValueError, TypeError):
                logger.warning(f"Invalid experience value: {exp_value}, defaulting to 0")
                experience = 0.0
            
            # Clean and VALIDATE skills from resume
            raw_skills = resume_data.get('SKILLS', [])
            
            
            parsed_data = {
                'skills': raw_skills,
                'experience': experience,
                'certifications': self._clean_list(resume_data.get('CERTIFICATION', [])),
                'education': self._clean_list(resume_data.get('EDUCATION', []))
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume from database: {e}", exc_info=True)
            return {'skills': [], 'experience': 0.0, 'education': [], 'certifications': []}
        
    
        
    def _clean_list(self, items: List) -> List[str]:
        """Clean and deduplicate list items"""
        if not items:
            return []
        cleaned = []
        for item in items:
            if isinstance(item, str) and len(item.strip()) > 1:
                cleaned.append(item.strip())
        return list(set(cleaned))
    
    def _extract_jd_requirements(self, job: Dict) -> Dict:
        """Extract requirements from job description"""
        requirements = {'skills': [], 'education': [], 'certifications': [], 'experience': 0}
        
        try:
            jd_text = self._get_jd_text(job)
            
            requirements['skills'] = self._extract_skills_from_jd_text(jd_text)
            requirements['education'] = self._extract_education_from_jd_text(jd_text)
            requirements['certifications'] = self._extract_certifications_from_jd_text(jd_text)
            requirements['experience'] = self._extract_experience_requirement(jd_text)
            
            for key in ['skills', 'education', 'certifications']:
                requirements[key] = self._clean_requirement_list(requirements[key])
            
        except Exception as e:
            logger.error(f"Error extracting JD requirements: {e}")
        
        return requirements
    
    def _clean_requirement_list(self, items: List[str]) -> List[str]:
        """Clean requirement list"""
        if not items:
            return []
        cleaned = []
        seen = set()
        for item in items:
            if isinstance(item, str):
                item_clean = item.strip().lower()
                if (item_clean and len(item_clean) > 2 and 
                    item_clean not in seen and 
                    not self._is_jd_noise(item_clean) and
                    not self._is_generic_word(item_clean)):
                    cleaned.append(item.strip())
                    seen.add(item_clean)
        return cleaned
    
    def _is_generic_word(self, text: str) -> bool:
        """Filter generic words"""
        generic_words = {
            'experience', 'knowledge', 'skills', 'ability', 'understanding',
            'proficiency', 'expertise', 'familiar', 'working', 'strong',
            'excellent', 'good', 'basic', 'advanced', 'required', 'preferred',
            'must', 'should', 'nice', 'plus', 'bonus', 'years', 'year',
            'required skills', 'preferred skills', 'key skills', 'technical skills'
        }
        return text.lower().strip() in generic_words
    
    def _get_jd_text(self, job: Dict) -> str:
        """Extract text from job document"""
        text_fields = [
            "Job_Description", "job_description", "description",
            "Job Description", "text", "content"
        ]
        
        for field in text_fields:
            if job.get(field):
                return str(job[field]).lower()
        
        if job.get("FileData"):
            try:
                with io.BytesIO(job["FileData"]) as data:
                    doc = fitz.open(stream=data)
                    return " ".join([page.get_text() for page in doc]).lower()
            except Exception as e:
                logger.warning(f"JD PDF extraction failed: {e}")
        
        return ""
    
    def _extract_skills_from_jd_text(self, text: str) -> List[str]:
        """Extract skills from JD"""
        if not text:
            return []
        
        skills = set()
        text_lower = text.lower()
        
        # YOUR OLD PATTERNS (keep these)
        skill_section_patterns = [
            r'(?:required|desired|key|technical|core)\s+skills?[:\-]\s*([^.!?\n]{10,800})',
            r'(?:skills?\s+required)[:\-]?\s*([^.!?\n]{10,800})',
            r'(?:technical\s+requirements?)[:\-]\s*([^.!?\n]{10,800})',
        ]
        
        for pattern in skill_section_patterns:
            try:
                matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    tokens = re.split(r'[,;/\n•\-\u2022\u25CF\|\(\)]+', match)
                    for token in tokens:
                        token = token.strip()
                        if self._is_valid_skill_token(token):
                            skills.add(token)
            except Exception as e:
                logger.warning(f"Skill section pattern failed: {e}")
        
        experience_patterns = [
            r'(?:experience\s+(?:with|in|using))\s+([a-zA-Z0-9\+\#\.\s\-]{2,50})(?:\s|,|;|\.|and|or|$)',
            r'(?:proficient\s+(?:in|with))\s+([a-zA-Z0-9\+\#\.\s\-]{2,50})(?:\s|,|;|\.|and|or|$)',
            r'(?:knowledge\s+of)\s+([a-zA-Z0-9\+\#\.\s\-]{2,50})(?:\s|,|;|\.|and|or|$)',
        ]
        
        for pattern in experience_patterns:
            try:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    token = match.strip()
                    if self._is_valid_skill_token(token):
                        skills.add(token)
            except Exception as e:
                logger.warning(f"Experience pattern failed: {e}")
        
        bullet_patterns = [
            r'(?:^|\n)\s*[•\-\*\u2022\u25CF]\s*([^\n]{5,150})',
        ]
        
        for pattern in bullet_patterns:
            try:
                matches = re.findall(pattern, text, re.MULTILINE)
                for match in matches:
                    cleaned = re.sub(r'^(?:experience\s+with|knowledge\s+of|proficiency\s+in)\s+', 
                                '', match.lower())
                    if self._is_valid_skill_token(cleaned):
                        skills.add(cleaned.strip())
            except Exception as e:
                logger.warning(f"Bullet pattern failed: {e}")
        
        # ADD NEW: Extract from "Technical Skills" structured list
        tech_section = re.search(r'technical\s+skills[:\-\s]+(.+?)(?=\n\n|soft\s+skills|what\s+we|$)', 
                                text_lower, re.IGNORECASE | re.DOTALL)
        if tech_section:
            # Extract items after Required:/Preferred:/Bonus:
            items = re.findall(r'(?:required|preferred|bonus)[:\-\s]+([^\n]+)', tech_section.group(1))
            for item in items:
                tokens = re.split(r'[,;/]+', item)
                for token in tokens:
                    cleaned = token.strip()
                    if self._is_valid_skill_token(cleaned):
                        skills.add(cleaned)
        
        tech_patterns = [
            r'\b(python|java|javascript|c\+\+|sql|mysql|postgresql|mongodb|firebase)\b',
            r'\b(react|angular|vue|node\.?js|express|django|flask)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|git|github)\b',
            r'\b(power\s*bi|tableau|excel|pandas|numpy|tensorflow|pytorch)\b',
            r'\b(html|css|rest|api|json|xml)\b',
            r'\b(figma|canva|adobe\s*xd|sketch)\b',  # NEW
            r'\b(rasa|chatbot|nlp)\b',  # NEW
            r'\b(capcut|premiere\s*pro)\b',  # NEW
            r'\b(ios|android|mobile)\b',  # NEW
        ]
        
        for pattern in tech_patterns:
            try:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if self._is_valid_skill_token(match):
                        skills.add(match.strip())
            except Exception as e:
                logger.warning(f"Tech pattern failed: {e}")
        
        return list(set([s for s in skills if s]))

    
    
    def _is_valid_skill_token(self, token: str) -> bool:
        """Validate if token is a real skill"""
        if not token or not isinstance(token, str):
            return False
        
        token = token.strip().lower()
        
        if len(token) < 2 or len(token) > 60:
            return False
        
        if not re.search(r'[a-z0-9]', token):
            return False
        
        if self._is_jd_noise(token) or self._is_generic_word(token):
            return False
        
        noise_starts = ['and ', 'or ', 'the ', 'of ', 'in ', 'at ', 'for ', 'with ', 'by ']
        if any(token.startswith(ns) for ns in noise_starts):
            return False
        
        noise_ends = [' years', ' year', ' experience', ' skills', ' knowledge']
        if any(token.endswith(ne) for ne in noise_ends):
            return False
        
        return True
    
    def _extract_education_from_jd_text(self, text: str) -> List[str]:
        """Extract education requirements"""
        if not text:
            return []
        
        education = set()
        patterns = [
            r'(?:education|degree|qualification)[:\-]\s*([^.!?\n]{5,200})',
            r'(?:bachelor|master|phd|doctorate|mba|degree)\s*(?:in\s*)?([a-zA-Z\s,]{3,100})',
            r'(?:major\s+in)\s+([a-zA-Z\s,]{3,100})',
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    tokens = re.split(r'[,;/\n•\-\u2022\u25CF]+', match)
                    for token in tokens:
                        token = token.strip().lower()
                        if token and len(token) > 3 and not self._is_jd_noise(token):
                            education.add(token)
            except Exception as e:
                logger.warning(f"Education extraction failed: {e}")
        
        return list(education)
    
    def _extract_certifications_from_jd_text(self, text: str) -> List[str]:
        """Extract certification requirements"""
        if not text:
            return []
        
        certifications = set()
        patterns = [
            r'([A-Za-z\s]+)\s+or\s+([A-Za-z\s]+)\s+(?:certified|certification)',
            r'([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Certified\s+[A-Za-z]+)',
            r'Certifications?[:\-\s]+([^.!?\n]{5,200})',
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        cert_text = ' '.join(match).strip()
                    else:
                        cert_text = match.strip()
                    
                    cert_clean = re.sub(r'\s+(?:is\s+)?(?:a\s+)?plus\s*$', '', 
                                       cert_text, flags=re.IGNORECASE).strip()
                    if cert_clean and len(cert_clean) > 5:
                        certifications.add(cert_clean)
            except Exception as e:
                logger.warning(f"Certification extraction failed: {e}")
        
        return list(certifications)
    
    def _extract_experience_requirement(self, text: str) -> float:
        """Extract years of experience requirement"""
        if not text:
            return 0.0
        
        patterns = [
            r'(\d+)(?:\+)?\s*(?:years?|yrs)\s+(?:of\s+)?experience',
            r'experience[:\-]\s*(\d+)(?:\+)?\s*(?:years?|yrs)',
            r'(\d+)(?:\+)?\s*(?:years?|yrs)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return 0.0
    
    def _calculate_detailed_scores(self, jd_requirements: Dict, resume_parsed: Dict) -> Dict[str, float]:
        """Calculate scores for each component"""
        scores = {}
        
        scores['skills'] = self._calculate_skill_score(
            jd_requirements.get('skills', []), 
            resume_parsed.get('skills', [])
        )
        
        scores['experience'] = self._calculate_experience_score(
            jd_requirements.get('experience', 0), 
            resume_parsed.get('experience', 0)
        )
        
        scores['education'] = self._calculate_education_score(
            jd_requirements.get('education', []), 
            resume_parsed.get('education', [])
        )
        
        cert_score = self._calculate_certification_score(
            jd_requirements.get('certifications', []), 
            resume_parsed.get('certifications', [])
        )
        
        # Store None if certifications not required, otherwise store score
        scores['certifications'] = cert_score if cert_score is not None else None
        
        return scores

    def _calculate_skill_score(self, jd_skills: List[str], resume_skills: List[str]) -> float:
        """
        IMPROVED: More reasonable scoring with better thresholds
        
        Scoring Philosophy:
        - Exact Match = 100% (perfect)
        - Strong Word Match = 90-95% (e.g., "excel" matches "ms excel")
        - Substring Match = 80-85% (e.g., "python" in "python programming")
        - High Fuzzy Match = 70-79% (very similar, likely same skill)
        - Medium Fuzzy Match = 50-69% (somewhat related, partial credit)
        - Low/No Match = 0-49% (not found or weak match)
        
        Final score = Average quality of all matches
        """
        match_logger.info("\n" + "─"*80)
        match_logger.info("🔧 SKILL MATCHING ANALYSIS")
        match_logger.info("─"*80)
        
        if not jd_skills:
            match_logger.info("ℹ️  No skills required in JD - Perfect match!")
            return 1.0
        
        if not resume_skills:
            match_logger.info("⚠️  No skills found in resume - No match!")
            return 0.0
        
        jd_skills_norm = [self._normalize_skill(s) for s in jd_skills]
        resume_skills_norm = [self._normalize_skill(s) for s in resume_skills]
        
        total_match_score = 0.0
        matches = []
        missing = []
        
        for jd_skill in jd_skills:
            jd_skill_norm = self._normalize_skill(jd_skill)
            jd_words = set(jd_skill_norm.split())
            
            best_match_score = 0.0
            best_match_resume_skill = None
            match_type = "NO MATCH"
            
            for resume_skill in resume_skills:
                resume_skill_norm = self._normalize_skill(resume_skill)
                resume_words = set(resume_skill_norm.split())
                
                # 1. EXACT MATCH = 100%
                if jd_skill_norm == resume_skill_norm:
                    best_match_score = 1.0
                    best_match_resume_skill = resume_skill
                    match_type = "EXACT"
                    break
                
                # 2. STRONG WORD MATCH = 90-95%
                # Core skill word(s) present in both
                common_words = jd_words & resume_words
                if common_words and len(jd_words) > 0:
                    word_match_ratio = len(common_words) / len(jd_words)
                    
                    # If most key words match, it's a strong match
                    if word_match_ratio >= 0.6:  # 60%+ words match
                        score = 0.90 + (word_match_ratio * 0.10)  # 90-100%
                        if score > best_match_score:
                            best_match_score = min(0.95, score)  # Cap at 95%
                            best_match_resume_skill = resume_skill
                            match_type = "WORD_MATCH"
                    elif word_match_ratio >= 0.4:  # 40-60% words match
                        score = 0.75 + (word_match_ratio * 0.15)  # 75-84%
                        if score > best_match_score:
                            best_match_score = score
                            best_match_resume_skill = resume_skill
                            match_type = "PARTIAL_WORD"
                
                # 3. SUBSTRING MATCH = 80-85%
                if jd_skill_norm in resume_skill_norm:
                    score = 0.85  # JD skill contained in resume
                    if score > best_match_score:
                        best_match_score = score
                        best_match_resume_skill = resume_skill
                        match_type = "SUBSTRING"
                elif resume_skill_norm in jd_skill_norm:
                    score = 0.80  # Resume skill contained in JD (less ideal)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_resume_skill = resume_skill
                        match_type = "SUBSTRING"
                
                # 4. FUZZY MATCH = 70-79% (only for very similar)
                similarity = SequenceMatcher(None, jd_skill_norm, resume_skill_norm).ratio()
                if similarity >= 0.80:  # High threshold for fuzzy
                    score = similarity * 0.90  # Max 90% for fuzzy
                    if score > best_match_score:
                        best_match_score = score
                        best_match_resume_skill = resume_skill
                        match_type = "FUZZY_HIGH"
                elif similarity >= 0.60:  # Medium fuzzy match
                    score = similarity * 0.75  # Max 75% for medium fuzzy
                    if score > best_match_score:
                        best_match_score = score
                        best_match_resume_skill = resume_skill
                        match_type = "FUZZY_MED"
            
            # Add to total score
            total_match_score += best_match_score
            
            # Log the match
            if best_match_resume_skill:
                match_info = {
                    'required': jd_skill,
                    'found': best_match_resume_skill,
                    'score': best_match_score,
                    'type': match_type
                }
                matches.append(match_info)
                match_logger.info(f"✅ [{match_type}] '{jd_skill}' = '{best_match_resume_skill}' (Score: {best_match_score:.0%})")
            else:
                missing.append(jd_skill)
                match_logger.info(f"❌ [MISSING] '{jd_skill}' - NOT FOUND IN RESUME")
        
        # Calculate final score as average match quality
        final_score = total_match_score / len(jd_skills)
        
        # Calculate match statistics
        exact_matches = len([m for m in matches if m['score'] >= 0.95])
        good_matches = len([m for m in matches if 0.80 <= m['score'] < 0.95])
        partial_matches = len([m for m in matches if 0.50 <= m['score'] < 0.80])
        weak_matches = len([m for m in matches if 0 < m['score'] < 0.50])
        
        match_logger.info("\n" + "─"*80)
        match_logger.info(f"📊 SKILL SUMMARY:")
        match_logger.info(f"   Total Required: {len(jd_skills)}")
        match_logger.info(f"   Exact/Strong Matches: {exact_matches} ({exact_matches/len(jd_skills):.0%})")
        match_logger.info(f"   Good Matches: {good_matches} ({good_matches/len(jd_skills):.0%})")
        match_logger.info(f"   Partial Matches: {partial_matches} ({partial_matches/len(jd_skills):.0%})")
        match_logger.info(f"   Weak Matches: {weak_matches} ({weak_matches/len(jd_skills):.0%})")
        match_logger.info(f"   Missing: {len(missing)} ({len(missing)/len(jd_skills):.0%})")
        match_logger.info(f"   Average Match Quality: {final_score:.2%}")
        match_logger.info("─"*80)
        
        self.matching_log['skills'] = {
            'matches': matches,
            'missing': missing,
            'score': final_score,
            'statistics': {
                'exact': exact_matches,
                'good': good_matches,
                'partial': partial_matches,
                'weak': weak_matches,
                'missing': len(missing)
            }
        }
        
        return min(1.0, final_score)


    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill for comparison"""
        if not skill:
            return ""
        
        skill = skill.lower().strip()
        skill = re.sub(r'[^\w\s\.\+\#]', '', skill)
        
        normalizations = {
            'javascript': 'js',
            'typescript': 'ts',
            'node.js': 'nodejs',
            'react.js': 'react',
            'vue.js': 'vue',
            'power bi': 'powerbi',
            'ms excel': 'excel',
            'microsoft excel': 'excel',
        }
        
        for old, new in normalizations.items():
            if old in skill:
                skill = skill.replace(old, new)
        
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        return skill
    
    def _calculate_experience_score(self, jd_experience: float, resume_experience: float) -> float:
        """
        IMPROVED: Experience scoring that respects weight preferences
        
        Logic:
        - If JD requires little/no experience (0-1 years) → Fresh grads get high score
        - If candidate exceeds requirement → Perfect score
        - Scoring is more lenient to reflect HR's priorities via weights
        """
        match_logger.info("\n" + "─"*80)
        match_logger.info("💼 EXPERIENCE MATCHING ANALYSIS")
        match_logger.info("─"*80)
        
        match_logger.info(f"Required: {jd_experience} years")
        match_logger.info(f"Candidate: {resume_experience} years")
        
        # Case 1: No experience required → Everyone gets perfect score
        if jd_experience == 0:
            match_logger.info("✅ No experience required - Perfect match!")
            score = 1.0
        
        # Case 2: Candidate meets or exceeds requirement
        elif resume_experience >= jd_experience:
            match_logger.info("✅ Meets or exceeds requirement!")
            score = 1.0
        
        # Case 3: Entry-level position (0-2 years) → Very lenient scoring
        elif jd_experience <= 2.0:
            if resume_experience >= jd_experience * 0.5:  # Has at least 50% of required
                score = 0.85
                match_logger.info(f"✅ Entry-level position: Close enough ({resume_experience}/{jd_experience} years) - Score: 85%")
            elif resume_experience > 0:
                score = 0.70
                match_logger.info(f"✅ Entry-level position: Some experience ({resume_experience} years) - Score: 70%")
            else:
                # Fresh grad for entry-level → Still good score!
                score = 0.60
                match_logger.info(f"✅ Entry-level position: Fresh graduate accepted - Score: 60%")
        
        # Case 4: Mid-level position (2-5 years) → Moderate scoring
        elif jd_experience <= 5.0:
            ratio = resume_experience / jd_experience
            
            if ratio >= 0.8:
                score = 0.90
                match_logger.info(f"✅ Close to requirement: {ratio:.0%} of required experience - Score: 90%")
            elif ratio >= 0.6:
                score = 0.75
                match_logger.info(f"⚠️  Acceptable gap: {ratio:.0%} of required experience - Score: 75%")
            elif ratio >= 0.4:
                score = 0.55
                match_logger.info(f"⚠️  Below requirement: {ratio:.0%} of required experience - Score: 55%")
            elif ratio >= 0.2:
                score = 0.35
                match_logger.info(f"❌ Significant gap: {ratio:.0%} of required experience - Score: 35%")
            else:
                score = 0.20
                match_logger.info(f"❌ Major gap: {ratio:.0%} of required experience - Score: 20%")
        
        # Case 5: Senior position (5+ years) → Strict scoring
        else:
            ratio = resume_experience / jd_experience
            
            if ratio >= 0.8:
                score = 0.85
                match_logger.info(f"✅ Senior role - Close: {ratio:.0%} of required experience - Score: 85%")
            elif ratio >= 0.6:
                score = 0.65
                match_logger.info(f"⚠️  Senior role - Gap: {ratio:.0%} of required experience - Score: 65%")
            elif ratio >= 0.4:
                score = 0.45
                match_logger.info(f"❌ Senior role - Significant gap: {ratio:.0%} of required experience - Score: 45%")
            else:
                score = 0.25
                match_logger.info(f"❌ Senior role - Not qualified: {ratio:.0%} of required experience - Score: 25%")
        
        match_logger.info(f"Final Experience Score: {score:.0%}")
        match_logger.info("─"*80)
        
        self.matching_log['experience'] = {
            'required': jd_experience,
            'found': resume_experience,
            'score': score,
            'level': 'entry' if jd_experience <= 2 else ('mid' if jd_experience <= 5 else 'senior')
        }
        
        return score
    
    def _calculate_education_score(self, jd_education: List[str], resume_education: List[str]) -> float:
        """
        FIXED: Candidate needs to match ANY ONE requirement well = 100%
        JD lists alternatives (IT OR CS), not requirements to have both
        """
        match_logger.info("\n" + "─"*80)
        match_logger.info("🎓 EDUCATION MATCHING ANALYSIS")
        match_logger.info("─"*80)
        
        if not jd_education:
            match_logger.info("ℹ️  No education required in JD - Perfect match!")
            return 1.0
        
        if not resume_education:
            match_logger.info("⚠️  No education found in resume - No match!")
            return 0.0
        
        best_overall_score = 0.0
        best_jd_match = None
        best_resume_match = None
        best_match_details = ""
        
        match_logger.info(f"\nJD accepts ANY of these degrees: {jd_education}")
        match_logger.info(f"Candidate has: {resume_education}")
        match_logger.info("\nFinding best match...\n")
        
        try:
            for resume_edu in resume_education:
                resume_level = self._map_education_level(resume_edu)
                resume_field = resume_edu.lower()
                
                for jd_edu in jd_education:
                    jd_level = self._map_education_level(jd_edu)
                    jd_field = jd_edu.lower()
                    
                    # Check if the field/major matches well
                    field_in_resume = jd_field in resume_field
                    resume_in_field = resume_field in jd_field
                    
                    # Calculate field similarity
                    text_similarity = SequenceMatcher(None, jd_field, resume_field).ratio()
                    
                    # Strong field match: substring or high similarity
                    is_strong_field_match = field_in_resume or resume_in_field or text_similarity >= 0.6
                    
                    match_logger.info(f"  Comparing: '{resume_edu}' vs '{jd_edu}'")
                    match_logger.info(f"    Degree Level: {resume_level} vs {jd_level}")
                    match_logger.info(f"    Field Match: {text_similarity:.0%} (Strong: {is_strong_field_match})")
                    
                    # SCORING LOGIC:
                    if resume_level >= jd_level and is_strong_field_match:
                        combined_score = 1.0
                        level_msg = "PERFECT MATCH"
                    elif resume_level >= jd_level:
                        combined_score = 0.7 + (text_similarity * 0.3)
                        level_msg = "DEGREE LEVEL OK, FIELD PARTIAL"
                    elif is_strong_field_match:
                        level_ratio = resume_level / max(jd_level, 1)
                        combined_score = 0.6 + (level_ratio * 0.2)
                        level_msg = "FIELD MATCH, DEGREE BELOW"
                    else:
                        combined_score = text_similarity * 0.6
                        level_msg = "WEAK MATCH"
                    
                    match_logger.info(f"    Assessment: {level_msg}")
                    match_logger.info(f"    Score: {combined_score:.0%}")
                    
                    if combined_score > best_overall_score:
                        best_overall_score = combined_score
                        best_jd_match = jd_edu
                        best_resume_match = resume_edu
                        best_match_details = level_msg
            
            if best_resume_match:
                match_logger.info("\n✅ BEST MATCH FOUND:")
                match_logger.info(f"   Candidate: '{best_resume_match}'")
                match_logger.info(f"   Satisfies JD requirement: '{best_jd_match}'")
                match_logger.info(f"   Assessment: {best_match_details}")
                match_logger.info(f"   Score: {best_overall_score:.0%}")
            else:
                match_logger.info("\n❌ NO SUITABLE MATCH FOUND")
        
        except Exception as e:
            logger.error(f"Error in education score logic: {e}")
            return 0.0
        
        match_logger.info("\n" + "─"*80)
        match_logger.info(f"📊 EDUCATION SUMMARY:")
        match_logger.info(f"   Final Score: {best_overall_score:.2%}")
        match_logger.info("─"*80)
        
        self.matching_log['education'] = {
            'matches': [{
                'required': best_jd_match,
                'found': best_resume_match,
                'score': best_overall_score
            }] if best_resume_match else [],
            'score': best_overall_score
        }
        
        return best_overall_score
        
        match_logger.info("\n" + "─"*80)
        match_logger.info(f"📊 EDUCATION SUMMARY:")
        match_logger.info(f"   Final Score: {best_overall_score:.2%}")
        match_logger.info("─"*80)
        
        self.matching_log['education'] = {
            'matches': [{
                'required': best_jd_match,
                'found': best_resume_match,
                'score': best_overall_score
            }] if best_resume_match else [],
            'score': best_overall_score
        }
        
        return best_overall_score

    def _map_education_level(self, edu_text: str) -> int:
        """Map education text to numeric level"""
        edu_text = edu_text.lower()
        
        if 'phd' in edu_text or 'doctorate' in edu_text:
            return 4
        if 'master' in edu_text or 'mba' in edu_text or "master's" in edu_text:
            return 3
        if 'bachelor' in edu_text or 'degree' in edu_text or "bachelor's" in edu_text:
            return 2
        if 'associate' in edu_text or 'diploma' in edu_text:
            return 1
        
        return 0
    
    def _calculate_certification_score(self, jd_certs: List[str], resume_certs: List[str]) -> Optional[float]:
        """
        IMPROVED: Certification scoring with bonus system
        
        Logic:
        1. JD requires certs + candidate has them → Score based on match
        2. JD requires certs + candidate doesn't → Score = 0%
        3. JD doesn't require + candidate has them → BONUS score (e.g., 80%)
        4. JD doesn't require + candidate doesn't → Neutral (return None)
        
        Returns:
            float: Score 0.0-1.0 if certifications are relevant
            None: If certifications should not affect score (neutral)
        """
        match_logger.info("\n" + "─"*80)
        match_logger.info("📜 CERTIFICATION MATCHING ANALYSIS")
        match_logger.info("─"*80)
        
        # Case 1: No certs required, no certs provided → NEUTRAL
        if not jd_certs and not resume_certs:
            match_logger.info("ℹ️  No certifications required or provided - NEUTRAL")
            match_logger.info("   This component will not affect the overall score.")
            return None
        
        # Case 2: No certs required, BUT candidate has them → BONUS!
        if not jd_certs and resume_certs:
            bonus_score = 0.80  # 80% bonus for having certs when not required
            match_logger.info(f"✅ BONUS: Certifications not required but candidate has {len(resume_certs)}")
            match_logger.info(f"   Candidate shows initiative with: {', '.join(resume_certs[:3])}")
            match_logger.info(f"   Bonus Score: {bonus_score:.0%}")
            match_logger.info("   This adds value to the candidate's profile!")
            
            self.matching_log['certifications'] = {
                'type': 'bonus',
                'provided': resume_certs,
                'score': bonus_score
            }
            return bonus_score
        
        # Case 3: Certs required, none provided → FAIL
        if jd_certs and not resume_certs:
            match_logger.info(f"❌ Certifications required but none found in resume")
            match_logger.info(f"   Required: {', '.join(jd_certs)}")
            match_logger.info(f"   Score: 0%")
            
            self.matching_log['certifications'] = {
                'type': 'required',
                'required': jd_certs,
                'provided': [],
                'score': 0.0
            }
            return 0.0
        
        # Case 4: Certs required AND provided → MATCH THEM
        match_logger.info(f"🔍 Matching required certifications...")
        match_logger.info(f"   Required: {len(jd_certs)} certification(s)")
        match_logger.info(f"   Provided: {len(resume_certs)} certification(s)")
        
        total_match_score = 0.0
        matches = []
        missing = []
        
        for jd_cert in jd_certs:
            best_match_score = 0.0
            best_match_cert = None
            match_type = "NO MATCH"
            jd_cert_norm = self._normalize_skill(jd_cert)
            
            for resume_cert in resume_certs:
                resume_cert_norm = self._normalize_skill(resume_cert)
                
                # Exact match
                if jd_cert_norm == resume_cert_norm:
                    best_match_score = 1.0
                    best_match_cert = resume_cert
                    match_type = "EXACT"
                    break
                
                # Substring match (e.g., "PMP" in "PMP Certified")
                if jd_cert_norm in resume_cert_norm or resume_cert_norm in jd_cert_norm:
                    if 0.90 > best_match_score:
                        best_match_score = 0.90
                        best_match_cert = resume_cert
                        match_type = "SUBSTRING"
                    continue
                
                # Fuzzy match (typos, variations)
                similarity = SequenceMatcher(None, jd_cert_norm, resume_cert_norm).ratio()
                if similarity >= 0.80:  # High threshold for certifications
                    if similarity > best_match_score:
                        best_match_score = similarity * 0.95  # Cap fuzzy at 95%
                        best_match_cert = resume_cert
                        match_type = f"FUZZY"
            
            total_match_score += best_match_score
            
            if best_match_cert:
                matches.append({
                    'required': jd_cert,
                    'found': best_match_cert,
                    'score': best_match_score,
                    'type': match_type
                })
                match_logger.info(f"   ✅ [{match_type}] '{jd_cert}' = '{best_match_cert}' ({best_match_score:.0%})")
            else:
                missing.append(jd_cert)
                match_logger.info(f"   ❌ [MISSING] '{jd_cert}'")
        
        final_score = total_match_score / len(jd_certs)
        
        match_logger.info("\n" + "─"*80)
        match_logger.info(f"📊 CERTIFICATION SUMMARY:")
        match_logger.info(f"   Required: {len(jd_certs)}")
        match_logger.info(f"   Matched: {len(matches)} ({len(matches)/len(jd_certs):.0%})")
        match_logger.info(f"   Missing: {len(missing)} ({len(missing)/len(jd_certs):.0%})")
        match_logger.info(f"   Final Score: {final_score:.0%}")
        match_logger.info("─"*80)
        
        self.matching_log['certifications'] = {
            'type': 'required',
            'matches': matches,
            'missing': missing,
            'score': final_score
        }
        
        return final_score



    def _is_jd_noise(self, text: str) -> bool:
        """Filter out noise words from JD"""
        if not text or len(text) < 2 or len(text) > 60:
            return True
        
        noise_patterns = [
            r'^\d+$',  # Pure numbers - FIXED: added closing quote and $
            r'^[a-z]$',  # Single letters - FIXED: added closing quote and $
            r'www\.|@|http',  # URLs/emails
            r'^\s*$',  # Empty/whitespace - FIXED: added closing quote and $
            r'^(?:and|or|the|of|in|at|for|with|by|from|to|a|an|is|are|will|be)\b',  # Common words
        ]

        
        return any(re.search(pattern, text.lower()) for pattern in noise_patterns)
    
    def _generate_detailed_analysis(self, scores: Dict[str, float], 
                                    jd_requirements: Dict, resume_parsed: Dict) -> List[ScoreAnalysis]:
        """
        Generate detailed analysis for each component
        FIXED: Properly handles None certification score when certs not required
        """
        analysis = []
        
        for component in ['skills', 'experience', 'education', 'certifications']:
            # Get score - might be None for certifications
            score = scores.get(component)
            
            # CRITICAL FIX: Check if score is None BEFORE any comparisons
            if score is None:
                # Certifications not required - create neutral analysis entry
                match_logger.info(f"   {component}: Not required (neutral score)")
                analysis.append(ScoreAnalysis(
                    component=component,
                    score=0.0,
                    max_possible=0.0,
                    strengths=["Not required for this position"],
                    weaknesses=[],
                    suggestions=[],
                    matched_items=[],
                    missing_items=[]
                ))
                continue  # Move to next component
            
            # Now score is guaranteed to be a float, safe to do comparisons
            strengths = []
            weaknesses = []
            suggestions = []
            matched_items = []
            missing_items = []
            
            # Get matching log data if available
            if component in self.matching_log:
                log_data = self.matching_log[component]
                matched_items = log_data.get('matches', [])
                missing_items = log_data.get('missing', [])
            
            # Generate feedback based on score thresholds
            if score >= 0.8:
                strengths.append(f"Strong {component} alignment")
                if score == 1.0:
                    strengths.append(f"Perfect {component} match")
            elif score >= 0.6:
                strengths.append(f"Good {component} match")
                suggestions.append(f"Consider strengthening {component} to reach excellent level")
            elif score >= 0.4:
                weaknesses.append(f"Moderate {component} gap")
                suggestions.append(f"Improve {component} section to better match requirements")
            elif score >= 0.2:
                weaknesses.append(f"Significant {component} gap")
                suggestions.append(f"Major improvement needed in {component}")
            else:
                weaknesses.append(f"Low {component} compatibility")
                suggestions.append(f"Critical: {component} does not meet requirements")
            
            # Create analysis entry
            analysis.append(ScoreAnalysis(
                component=component,
                score=score,
                max_possible=1.0,
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions,
                matched_items=matched_items,
                missing_items=missing_items
            ))
        
        return analysis
    
    def _generate_hr_feedback(self, analysis: List[ScoreAnalysis], scores: Dict[str, float], 
                            weights: MatchingWeights) -> Dict[str, Any]:
        """
        FIXED: Filter out None scores before calculating average
        Generate feedback for HR
        """
        # Filter out None values (certifications not required)
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        
        if not valid_scores:
            total_score = 0.0
        else:
            total_score = sum(valid_scores.values()) / len(valid_scores)
        
        # Generate assessment based on total score
        if total_score >= 0.8:
            assessment = "Excellent candidate with strong alignment"
            recommendation = "Highly recommended for interview"
            priority = "High Priority"
        elif total_score >= 0.65:
            assessment = "Good candidate with solid qualifications"
            recommendation = "Recommended for interview"
            priority = "Medium-High Priority"
        elif total_score >= 0.5:
            assessment = "Moderate fit with some gaps"
            recommendation = "Consider for interview if candidate pool is limited"
            priority = "Medium Priority"
        elif total_score >= 0.35:
            assessment = "Below average fit with significant gaps"
            recommendation = "Not recommended unless specific skills are critical"
            priority = "Low Priority"
        else:
            assessment = "Limited alignment with requirements"
            recommendation = "Not recommended"
            priority = "Reject"
        
        # Extract strengths and concerns from analysis (skip None scores)
        key_strengths = []
        main_concerns = []
        
        for a in analysis:
            if a.score is None or a.score == 0.0:
                continue  # Skip not required components
            
            if a.strengths:
                key_strengths.extend(a.strengths)
            if a.weaknesses:
                main_concerns.extend(a.weaknesses)
        
        return {
            'overall_assessment': assessment,
            'hiring_recommendation': recommendation,
            'priority_level': priority,
            'key_strengths': key_strengths[:5],  # Top 5
            'main_concerns': main_concerns[:5],  # Top 5
            'average_score': round(total_score * 100, 2)
        }
    
    def _generate_candidate_feedback(self, analysis: List[ScoreAnalysis], 
                                    scores: Dict[str, float]) -> Dict[str, Any]:
        """
        FIXED: Filter out None scores before calculating average
        Generate feedback for candidate
        """
        # Filter out None values (certifications not required)
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        
        if not valid_scores:
            total_score = 0.0
        else:
            total_score = sum(valid_scores.values()) / len(valid_scores)
        
        # Generate message based on score
        if total_score >= 0.8:
            message = "You're a strong match for this position!"
            encouragement = "Your qualifications align excellently with the role requirements."
        elif total_score >= 0.65:
            message = "You have solid qualifications for this role with some areas for enhancement."
            encouragement = "Focus on the suggested improvements to become an even stronger candidate."
        elif total_score >= 0.5:
            message = "You have relevant experience but there are some gaps to address."
            encouragement = "Work on developing the missing skills to improve your match."
        elif total_score >= 0.35:
            message = "This role may require additional qualifications or experience."
            encouragement = "Consider gaining more experience in the key areas listed below."
        else:
            message = "This role requires significantly more qualifications or different experience."
            encouragement = "Focus on building foundational skills before applying to similar positions."
        
        # Collect all suggestions (skip None scores)
        all_suggestions = []
        for a in analysis:
            if a.score is not None and a.suggestions:
                all_suggestions.extend(a.suggestions)
        
        return {
            'overall_message': message,
            'encouragement': encouragement,
            'suggestions': all_suggestions[:5],  # Top 5 suggestions
            'match_percentage': round(total_score * 100, 2),
            'next_steps': [
                "Focus on developing key skills identified in suggestions",
                "Highlight relevant experience in your resume",
                "Consider gaining certifications in required areas"
            ]
        }
    
    def _create_error_result(self, error_message: str) -> MatchingResult:
        """Create error result"""
        return MatchingResult(
            overall_score=0.0,
            component_scores={'skills': 0.0, 'experience': 0.0, 'education': 0.0, 'certifications': 0.0},
            detailed_analysis=[],
            hr_feedback={'error': error_message},
            candidate_feedback={'error': error_message},
            weights_used=MatchingWeights(0.5, 0.3, 0.15, 0.05),
            extraction_log={},
            matching_log={}
        )



@simple_performance_monitor(matching_metrics)
def ProductionMatching(job_id: str, resume_id: str, skill_weight: int = 50, 
                       education_weight: int = 15, experience_weight: int = 30, 
                       certification_weight: int = 5) -> Dict:
    """
    Production matching function with detailed logging
    FIXED: Properly processes custom weights
    """
    if resumeFetchedData is None or JOBS is None:
        return {
            "error": "Database connection not available",
            "overall_score": 0.0,
            "component_scores": {"skills": 0, "experience": 0, "education": 0, "certifications": 0},
            "timestamp": datetime.now().isoformat()
        }
    
    start_time = time.time()
    sanitizer = InputSanitizer()
    
    try:
        if not job_id or not resume_id:
            return {
                "error": "job_id and resume_id are required",
                "overall_score": 0.0,
                "component_scores": {"skills": 0, "experience": 0, "education": 0, "certifications": 0},
                "timestamp": datetime.now().isoformat()
            }
        
        if not sanitizer.validate_object_id(job_id) or not sanitizer.validate_object_id(resume_id):
            return {
                "error": "Invalid ObjectId format",
                "overall_score": 0.0,
                "component_scores": {"skills": 0, "experience": 0, "education": 0, "certifications": 0},
                "timestamp": datetime.now().isoformat()
            }
        
        # ===== CRITICAL FIX: Proper weight processing =====
        logger.info("\n" + "="*80)
        logger.info("⚖️  PROCESSING CUSTOM WEIGHTS")
        logger.info("="*80)
        
        # Log received weights
        logger.info(f"Weights received from application:")
        logger.info(f"  Skills: {skill_weight}%")
        logger.info(f"  Experience: {experience_weight}%")
        logger.info(f"  Education: {education_weight}%")
        logger.info(f"  Certifications: {certification_weight}%")
        logger.info(f"  Total: {skill_weight + experience_weight + education_weight + certification_weight}%")
        
        # Prepare raw weights
        raw_weights = {
            'skills': skill_weight,
            'experience': experience_weight,
            'education': education_weight,
            'certifications': certification_weight
        }
        
        # Sanitize weights (removes invalid values)
        sanitized_weights = sanitizer.sanitize_weights(raw_weights)
        
        if not sanitized_weights:
            logger.warning("All weights were invalid, using defaults")
            sanitized_weights = {'skills': 50, 'experience': 30, 'education': 15, 'certifications': 5}
        
        # Calculate total
        total_weight = sum(sanitized_weights.values())
        logger.info(f"\nAfter sanitization, total weight: {total_weight}%")
        
        # CRITICAL: Check if weights need normalization
        if total_weight == 0:
            logger.warning("Total weight is 0, using defaults")
            sanitized_weights = {'skills': 50, 'experience': 30, 'education': 15, 'certifications': 5}
            total_weight = 100
        
        # If weights don't add up to 100, normalize them proportionally
        if abs(total_weight - 100) > 0.1:
            logger.info(f"Weights don't sum to 100%, normalizing proportionally...")
            
            # Store original for comparison
            original_weights = sanitized_weights.copy()
            
            # Normalize to 100
            sanitized_weights = {
                k: (v / total_weight) * 100 
                for k, v in sanitized_weights.items()
            }
            
            logger.info(f"\nNormalization applied:")
            for component in ['skills', 'experience', 'education', 'certifications']:
                orig = original_weights.get(component, 0)
                norm = sanitized_weights.get(component, 0)
                logger.info(f"  {component.title()}: {orig:.1f}% → {norm:.1f}%")
        else:
            logger.info(f"Weights already sum to 100%, no normalization needed")
        
        # Final weights that will be used
        logger.info(f"\n✅ FINAL WEIGHTS TO BE USED:")
        logger.info(f"  Skills: {sanitized_weights.get('skills', 50):.1f}%")
        logger.info(f"  Experience: {sanitized_weights.get('experience', 30):.1f}%")
        logger.info(f"  Education: {sanitized_weights.get('education', 15):.1f}%")
        logger.info(f"  Certifications: {sanitized_weights.get('certifications', 5):.1f}%")
        logger.info("="*80 + "\n")
        
        # ===== Pass weights to matcher =====
        matcher = IntelligentMatcher()
        result = matcher.match_resume_to_job(job_id, resume_id, sanitized_weights)
        
        # ===== Verify weights were applied =====
        # The result should contain the weights that were actually used
        weights_used = result.weights_used
        logger.info("\n" + "="*80)
        logger.info("✅ WEIGHTS VERIFICATION")
        logger.info("="*80)
        logger.info(f"Weights actually applied in calculation:")
        logger.info(f"  Skills: {weights_used.skills * 100:.1f}%")
        logger.info(f"  Experience: {weights_used.experience * 100:.1f}%")
        logger.info(f"  Education: {weights_used.education * 100:.1f}%")
        logger.info(f"  Certifications: {weights_used.certifications * 100:.1f}%")
        
        # Check if they match what we sent
        expected_skills = sanitized_weights.get('skills', 50) / 100
        actual_skills = weights_used.skills
        
        if abs(expected_skills - actual_skills) > 0.01:
            logger.error(f"⚠️  WEIGHT MISMATCH DETECTED!")
            logger.error(f"   Expected skills weight: {expected_skills:.3f}")
            logger.error(f"   Actually used: {actual_skills:.3f}")
        else:
            logger.info(f"✅ Weights match! Custom weights were correctly applied.")
        logger.info("="*80 + "\n")
        
        # Build response
        response = {
            "overall_score": result.overall_score,
            "component_scores": result.component_scores,
            "detailed_analysis": [
                {
                    "component": analysis.component,
                    "score": round(analysis.score * 100, 2) if analysis.score is not None else 0.0,
                    "strengths": analysis.strengths,
                    "weaknesses": analysis.weaknesses,
                    "suggestions": analysis.suggestions,
                    "matched_items": analysis.matched_items,
                    "missing_items": analysis.missing_items
                }
                for analysis in result.detailed_analysis
            ],
            "hr_feedback": result.hr_feedback,
            "candidate_feedback": result.candidate_feedback,
            "weights_used": {
                "skills": round(result.weights_used.skills * 100, 1),
                "experience": round(result.weights_used.experience * 100, 1),
                "education": round(result.weights_used.education * 100, 1),
                "certifications": round(result.weights_used.certifications * 100, 1)
            },
            "extraction_log": result.extraction_log,
            "matching_log": result.matching_log,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "system_version": "3.2_detailed_logging_fixed_weights"
        }
        
        return response
        
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Critical error in ProductionMatching: {e}", exc_info=True)
        return {
            "error": f"Matching failed: {str(e)}",
            "error_type": error_type,
            "overall_score": 0.0,
            "component_scores": {"skills": 0, "experience": 0, "education": 0, "certifications": 0},
            "detailed_analysis": [],
            "hr_feedback": {"error": str(e)},
            "candidate_feedback": {"error": str(e)},
            "weights_used": {"skills": 50.0, "experience": 30.0, "education": 15.0, "certifications": 5.0},
            "timestamp": datetime.now().isoformat()
        }



def Matching(job_id: str, resume_id: str, skill_weight: int = 50, education_weight: int = 15, 
             experience_weight: int = 30, certification_weight: int = 5) -> Dict:
    """Backward compatibility wrapper"""
    return ProductionMatching(job_id, resume_id, skill_weight, education_weight, 
                              experience_weight, certification_weight)