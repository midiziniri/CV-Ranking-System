# sentence_transformer_ranker.py - ENHANCED VERSION
"""
Sentence Transformer Integration for Resume Ranking System
Uses ONLY pre-trained Sentence Transformer models for semantic matching
NO custom NER model is used for semantic similarity calculation
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re
from bson.objectid import ObjectId
from functools import lru_cache

# Core sentence transformers import
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers loaded successfully")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ Sentence Transformers not available - install with: pip install sentence-traynsformers")

logger = logging.getLogger(__name__)

class SemanticResumeRanker:

    
    def __init__(self, model_name="TechWolf/JobBERT-v3"):
      
        self.model = None
        self.model_name = model_name
        self.cache = {}  # Simple caching for embeddings
        self.max_cache_size = 1000  # Prevent memory issues
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading PRE-TRAINED sentence transformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("✅ PRE-TRAINED Sentence Transformer loaded (no training required)")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("Sentence transformers not available - falling back to basic matching")
    
    def is_available(self) -> bool:
        """Check if semantic ranking is available"""
        return self.model is not None
    
    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate semantic embedding using PRE-TRAINED Sentence Transformer.
        NO custom model training involved - this uses the pre-trained model directly.
        
        Args:
            text: Input text to embed
            
        Returns:
            Dense vector embedding capturing semantic meaning
        """
        if not self.model:
            return None
        
        # Simple cache key (first 500 chars)
        cache_key = hash(text.strip()[:500])
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Clean and prepare text
            cleaned_text = self.clean_text_for_embedding(text)
            if not cleaned_text:
                return None
            
            # Get embedding using PRE-TRAINED model
            embedding = self.model.encode([cleaned_text], show_progress_bar=False)[0]
            
            # Cache management - prevent unlimited growth
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest 20% of cache
                remove_count = self.max_cache_size // 5
                keys_to_remove = list(self.cache.keys())[:remove_count]
                for key in keys_to_remove:
                    del self.cache[key]
            
            # Cache result
            self.cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def clean_text_for_embedding(self, text: str) -> str:
        """
        Clean text for better embedding quality.
        This is just text preprocessing - NO model involved here.
        """
        if not text:
            return ""
        
        # Standard cleaning
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text if item)
        elif not isinstance(text, str):
            text = str(text)
        
        # Remove excessive general words (optional noise filtering)
        filler_words = r'\b(responsible for|created a|implemented functionality|supervise|developed and updated|designed a|which accounted for|which enabled|saving institutions|to gain hands-on experience|to enhance technical skills|in a professional|to work offline|without losing data|of the philippines|date of birth|citizenship|civil status|name of father|name of mother|number|time-management|trustworthy|ability to manage assigned task)\b'
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters (keep meaningful punctuation)
        text = re.sub(r'[^\w\s\-\+\.\,\(\)]', ' ', text)
        
        # Limit length
        if len(text) > 2000:
            text = text[:2000]
            
        return text.strip()
    
    def extract_job_requirements(self, job_doc: Dict[str, Any]) -> str:
        """
        Extract job requirements NATURALLY - no artificial emphasis needed!
        """
        components = []
        
        def safe_text(value) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return ', '.join(str(v) for v in value if v)
            return str(value)
        
        # Just describe what the job needs - model understands importance
        
        if job_doc.get('Job_Profile'):
            components.append(f"Position: {safe_text(job_doc['Job_Profile'])}")
        
        if job_doc.get('Job_Description'):
            components.append(f"Description: {safe_text(job_doc['Job_Description'])}")
        
        # Inside extract_job_requirements
        if job_doc.get('Required_Skills'):
            # Repeat the required skills twice to increase their semantic weight
            skills = safe_text(job_doc['Required_Skills'])
            components.append(f"Required skills: {skills}. Required skills: {skills}")
        
        if job_doc.get('Preferred_Skills'):
            components.append(f"Preferred skills: {safe_text(job_doc['Preferred_Skills'])}")
        
        if job_doc.get('Experience_Required'):
            exp = safe_text(job_doc['Experience_Required'])
            if exp and str(exp) != '0':
                components.append(f"Experience needed: {exp}")
        
        if job_doc.get('Education_Required'):
            components.append(f"Education: {safe_text(job_doc['Education_Required'])}")
        
        return '. '.join(components)


    def _extract_requirement_sentences(self, text: str, max_sentences: int = 3) -> str:
        """
        Dynamically extract sentences that describe actual requirements.
        Uses linguistic patterns, NOT job-specific hardcoding.
        """
        if not text or len(text) < 50:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Requirement patterns (general, not domain-specific)
        requirement_indicators = [
            r'\b(require|required|must have|must be|should have|need|necessary|essential)\b',
            r'\b(proficien|experience with|knowledge of|familiar with|understanding of)\b',
            r'\b(bachelor|master|degree|certification|years? of experience)\b',
            r'\b(ability to|capable of|skilled in|expertise in)\b',
            r'\b(responsible for|will be expected to|duties include)\b'
        ]
        
        requirement_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:  # Skip very short fragments
                continue
            
            # Check if sentence contains requirement indicators
            has_requirement = False
            for pattern in requirement_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    has_requirement = True
                    break
            
            if has_requirement:
                # Clean the sentence
                sentence_clean = re.sub(r'\s+', ' ', sentence)
                requirement_sentences.append(sentence_clean)
            
            # Stop if we have enough
            if len(requirement_sentences) >= max_sentences:
                break
        
        return ' '.join(requirement_sentences)
    
    def extract_resume_content(self, resume_doc: Dict[str, Any]) -> str:
        """
        HYBRID APPROACH: Use structured NER entities but preserve context.
        """
        sections = []
        
        def safe_text(value) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return ', '.join(str(v) for v in value if v)
            return str(value)
        
        
        # WORK EXPERIENCE - Use FULL job descriptions
        if resume_doc.get('WORKED AS'):
            work_text = safe_text(resume_doc['WORKED AS'])
            if work_text and len(work_text) > 10:
                sections.append(f"Professional Experience: {work_text}")
        
        # SKILLS - Just list them (already extracted cleanly)
        if resume_doc.get('SKILLS'):
            skills_text = safe_text(resume_doc['SKILLS'])
            if skills_text:
                sections.append(f"Technical Skills: {skills_text}")
        
        # EDUCATION - Include major/specialization
        if resume_doc.get('EDUCATION'):
            edu_text = safe_text(resume_doc['EDUCATION'])
            if edu_text:
                sections.append(f"Education: {edu_text}")
        
        # CERTIFICATIONS
        if resume_doc.get('CERTIFICATION'):
            cert_text = safe_text(resume_doc['CERTIFICATION'])
            if cert_text:
                sections.append(f"Certifications: {cert_text}")
        
        # YEARS OF EXPERIENCE
        if resume_doc.get('YEARS OF EXPERIENCE'):
            years = safe_text(resume_doc['YEARS OF EXPERIENCE'])
            if years and str(years) not in ['0', '0.0', '']:
                sections.append(f"{years} years of experience")
        
        combined = '. '.join(sections)
        
        # If NER extraction gave us very little, fallback to raw resume
        if len(combined) < 200 and resume_doc.get('ResumeData'):
            raw_text = self.clean_text_for_embedding(resume_doc['ResumeData'])
            return raw_text
        
        return combined


    def _filter_technical_skills(self, skills_text: str) -> str:
        """
        Dynamically filter technical skills from soft skills.
        NO HARDCODED SKILL LISTS - uses patterns.
        """
        if not skills_text:
            return ""
        
        # Split into individual skills
        if isinstance(skills_text, str):
            skill_items = re.split(r'[,;/\n]+', skills_text)
        else:
            return str(skills_text)
        
        # Soft skill patterns to EXCLUDE (general patterns, not exhaustive lists)
        soft_skill_patterns = [
            r'^(time[\s\-]?management|communication|teamwork|problem[\s\-]?solving)$',
            r'^(adaptability|flexibility|leadership|creativity|critical[\s\-]?thinking)$',
            r'^(positive\s+attitude|trustworthy|hardworking|detail[\s\-]?oriented)$',
            r'^(organizational\s+skills|interpersonal|self[\s\-]?motivated)$',
            r'^(customer\s+service|presentation\s+skills)$'
        ]
        
        technical_skills = []
        for skill in skill_items:
            skill = skill.strip()
            if not skill or len(skill) < 2:
                continue
            
            # Check if it matches soft skill patterns
            is_soft_skill = False
            for pattern in soft_skill_patterns:
                if re.match(pattern, skill, re.IGNORECASE):
                    is_soft_skill = True
                    break
            
            # Keep if it's NOT a soft skill
            if not is_soft_skill:
                technical_skills.append(skill)
        
        return ', '.join(technical_skills)


    def _extract_technical_context(self, text: str, max_length: int = 400) -> str:
        """
        Extract technical context from longer text (e.g., project descriptions).
        Looks for sentences with technical keywords.
        NO HARDCODED TECH STACKS - uses general patterns.
        """
        if not text or len(text) < 50:
            return text[:max_length]
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Technical indicators (general patterns, not tech-specific)
        technical_indicators = [
            r'\b(app|application|system|software|platform|tool|framework|library)\b',
            r'\b(database|server|api|backend|frontend|mobile|web)\b',
            r'\b(data|analytics|visualization|report|dashboard)\b',
            r'\b(design|develop|implement|integrate|deploy|maintain)\b',
            r'\b(code|programming|algorithm|optimization)\b',
            r'\b(user|interface|experience|UI|UX)\b',
            # Pattern for technology names (CamelCase, acronyms, etc.)
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase like "Firebase", "PowerBI"
            r'\b[A-Z]{2,}\b',  # Acronyms like "SQL", "API", "PDF"
        ]
        
        technical_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check if sentence contains technical indicators
            has_technical = False
            for pattern in technical_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    has_technical = True
                    break
            
            if has_technical:
                technical_sentences.append(sentence)
            
            # Stop if we've collected enough
            if len(' '.join(technical_sentences)) > max_length:
                break
        
        # Join and truncate
        result = ' '.join(technical_sentences)
        if len(result) > max_length:
            result = result[:max_length]
        
        return result.strip()
        


    def clean_text_for_embedding(self, text: str) -> str:
        """
        MINIMAL cleaning to preserve context.
        Only remove truly meaningless metadata.
        """
        if not text:
            return ""
        
        # Handle list inputs
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text if item)
        elif not isinstance(text, str):
            text = str(text)
        
        # Remove ONLY metadata noise (not action verbs!)
        metadata_noise = r'\b(date of birth|citizenship|sex|civil status|gender|name of (father|mother)|phone|email|address|hereby certify|references available)\b'
        text = re.sub(metadata_noise, '', text, flags=re.IGNORECASE)
        
        # Remove excessive contact info patterns
        text = re.sub(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', '', text)  # Dates
        text = re.sub(r'\b\d{10,}\b', '', text)  # Phone numbers
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # Emails
        
        # Standard whitespace cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit total length
        if len(text) > 2000:
            text = text[:2000]
        
        return text.strip()
    
    def calculate_semantic_similarity(self, job_text: str, resume_text: str) -> float:
        """
        Calculate semantic similarity with ADJUSTED SCALING for better differentiation.
        Prevents all scores clustering in 70-80% range.
        """
        if not job_text or not resume_text:
            return 0.0
        
        try:
            # Get embeddings using PRE-TRAINED Sentence Transformer
            job_embedding = self.get_text_embedding(job_text)
            resume_embedding = self.get_text_embedding(resume_text)
            
            if job_embedding is None or resume_embedding is None:
                return 0.0
            
            # Reshape for cosine similarity
            if len(job_embedding.shape) == 1:
                job_embedding = job_embedding.reshape(1, -1)
            if len(resume_embedding.shape) == 1:
                resume_embedding = resume_embedding.reshape(1, -1)
            
            # Calculate raw cosine similarity
            raw_similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
            
            # Normalize from [-1, 1] to [0, 1]
            normalized = float((raw_similarity + 1) / 2)
            
            # APPLY NON-LINEAR SCALING for better differentiation
            # This spreads out the scores instead of clustering them
            if normalized < 0.50:
                # Weak matches: Penalize more (0-50% -> 0-35%)
                adjusted = normalized * 0.70
            elif normalized < 0.65:
                # Moderate matches: Slight penalty (50-65% -> 35-55%)
                adjusted = 0.35 + (normalized - 0.50) * 1.33
            elif normalized < 0.80:
                # Good matches: Linear (65-80% -> 55-75%)
                adjusted = 0.55 + (normalized - 0.65) * 1.33
            else:
                # Excellent matches: Reward (80-100% -> 75-95%)
                adjusted = 0.75 + (normalized - 0.80) * 1.0
            
            return float(max(0.0, min(1.0, adjusted)))
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0
    
    def calculate_combined_score(self, traditional_score: float, semantic_score: float, 
                                traditional_weight: float = 0.40, semantic_weight: float = 0.60) -> float:
        """
        Combine traditional keyword matching with semantic similarity.
        
        Traditional matching uses YOUR NER model (done elsewhere in your system).
        Semantic matching uses PRE-TRAINED Sentence Transformer (done in this class).
        
        Args:
            traditional_score: Keyword-based score from YOUR NER model (0-100)
            semantic_score: Semantic score from Sentence Transformer (0-100)
            traditional_weight: Weight for traditional score (default 40%)
            semantic_weight: Weight for semantic score (default 60%)
            
        Returns:
            Combined weighted score (0-100)
        """
        # Ensure semantic score is in 0-100 range
        if semantic_score < 2:  # Likely 0-1 range
            semantic_score = semantic_score * 100
        
        combined = (traditional_score * traditional_weight) + (semantic_score * semantic_weight)
        return round(combined, 2)
    
    def rank_candidates_for_job(self, job_id: str, mongo_collections: Dict) -> List[Dict[str, Any]]:
        """
        Rank all candidates for a job using hybrid approach:
        1. Traditional matching: Uses YOUR NER model results (already stored)
        2. Semantic matching: Uses PRE-TRAINED Sentence Transformer (calculated here)
        3. Combines both scores with weighting
        
        Args:
            job_id: MongoDB ObjectId string for the job
            mongo_collections: Dict with 'JOBS', 'Applied_EMP', 'resumeFetchedData' collections
        
        Returns:
            List of ranked candidates with scores
        """
        try:
            # Get job details
            job = mongo_collections['JOBS'].find_one({"_id": ObjectId(job_id)})
            if not job:
                logger.error(f"Job {job_id} not found")
                return []
            
            # Extract job requirements text (NO model - just text extraction)
            job_text = self.extract_job_requirements(job)
            logger.info(f"Job requirements extracted: {len(job_text)} characters")
            
            # Get all applications for this job
            applications = list(mongo_collections['Applied_EMP'].find({"job_id": ObjectId(job_id)}))
            logger.info(f"Found {len(applications)} applications for job {job_id}")
            
            ranked_candidates = []
            
            for app in applications:
                try:
                    # Get resume data (entities extracted by YOUR NER model previously)
                    resume = mongo_collections['resumeFetchedData'].find_one({"_id": app.get('resume_id')})
                    if not resume:
                        logger.warning(f"Resume not found for application {app.get('_id')}")
                        continue
                    
                    # Extract resume content text (NO model - just text extraction)
                    resume_text = self.extract_resume_content(resume)
                    
                    # Calculate semantic similarity using PRE-TRAINED Sentence Transformer
                    semantic_score = self.calculate_semantic_similarity(job_text, resume_text)
                    
                    # Get traditional score (from YOUR NER model, calculated elsewhere)
                    traditional_score = app.get('Matching_percentage', 0)
                    if isinstance(traditional_score, dict):
                        traditional_score = traditional_score.get('overall_score', 0)
                    
                    try:
                        traditional_score = float(traditional_score)
                    except:
                        traditional_score = 0.0
                    
                    # Combine scores: Traditional (YOUR NER) + Semantic (Sentence Transformer)
                    combined = self.calculate_combined_score(traditional_score, semantic_score * 100)
                    
                    # Create enhanced candidate record
                    candidate_data = {
                        'application_id': str(app['_id']),
                        'user_id': str(app.get('user_id')),
                        'resume_id': str(app.get('resume_id')),
                        'candidate_name': resume.get('Name', 'Unknown'),
                        'traditional_score': round(traditional_score, 2),  # From YOUR NER model
                        'semantic_score': round(semantic_score * 100, 2),  # From Sentence Transformer
                        'combined_score': combined,
                        'applied_date': app.get('applied_at'),
                        'status': app.get('status', 'pending'),
                        'skills': resume.get('SKILLS', []),
                        'experience': resume.get('WORKED AS', []),
                        'years_experience': resume.get('YEARS OF EXPERIENCE', []),
                        'education': resume.get('EDUCATION', []),
                        'certifications': resume.get('CERTIFICATION', [])
                    }
                    
                    ranked_candidates.append(candidate_data)
                    
                except Exception as e:
                    logger.error(f"Error processing application {app.get('_id')}: {e}")
                    continue
            
            # Sort by combined score (descending)
            ranked_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Add ranking positions
            for i, candidate in enumerate(ranked_candidates):
                candidate['rank'] = i + 1
            
            logger.info(f"Successfully ranked {len(ranked_candidates)} candidates for job {job_id}")
            return ranked_candidates
            
        except Exception as e:
            logger.error(f"Error ranking candidates for job {job_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def batch_update_applications_with_semantic_scores(self, mongo_collections: Dict, 
                                                     job_id: Optional[str] = None,
                                                     force_refresh: bool = False) -> int:
        """
        Update existing applications with semantic similarity scores.
        Uses PRE-TRAINED Sentence Transformer only.
        
        Args:
            mongo_collections: Dict with MongoDB collections
            job_id: Optional job ID to limit updates to specific job
            force_refresh: If True, recalculate even if scores exist
        
        Returns:
            Number of applications updated
        """
        if not self.model:
            logger.warning("Semantic model not available for batch updates")
            return 0
        
        try:
            # Build query
            query = {}
            if job_id:
                query["job_id"] = ObjectId(job_id)
            
            # Skip applications that already have semantic scores (unless force_refresh)
            if not force_refresh:
                query["semantic_similarity"] = {"$exists": False}
            
            # Get applications that need semantic scoring
            applications = list(mongo_collections['Applied_EMP'].find(query))
            logger.info(f"Processing {len(applications)} applications for semantic scoring")
            
            updated_count = 0
            
            for app in applications:
                try:
                    # Get job and resume data
                    job = mongo_collections['JOBS'].find_one({"_id": app.get('job_id')})
                    resume = mongo_collections['resumeFetchedData'].find_one({"_id": app.get('resume_id')})
                    
                    if not job or not resume:
                        continue
                    
                    # Extract texts and calculate similarity using PRE-TRAINED model
                    job_text = self.extract_job_requirements(job)
                    resume_text = self.extract_resume_content(resume)
                    semantic_score = self.calculate_semantic_similarity(job_text, resume_text)
                    
                    # Update application with semantic score
                    update_data = {
                        'semantic_similarity': semantic_score,
                        'semantic_updated_at': datetime.now(),
                        'semantic_model': self.model_name,
                        'enhanced_version': 'sentence_transformer_pretrained'
                    }
                    
                    result = mongo_collections['Applied_EMP'].update_one(
                        {"_id": app['_id']},
                        {"$set": update_data}
                    )
                    
                    if result.modified_count > 0:
                        updated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error updating application {app.get('_id')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully updated {updated_count} applications with semantic scores")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error in batch semantic score update: {e}")
            return 0
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self.cache.clear()
        logger.info("Embedding cache cleared")


# Flask Integration Functions (Singleton pattern)
_semantic_ranker_instance = None

def get_semantic_ranker() -> SemanticResumeRanker:
    """Get singleton instance of semantic ranker (using PRE-TRAINED model)"""
    global _semantic_ranker_instance
    if _semantic_ranker_instance is None:
        _semantic_ranker_instance = SemanticResumeRanker()
    return _semantic_ranker_instance


def enhance_application_with_semantic_score(job_id: str, resume_id: str, mongo_collections: Dict) -> Dict[str, Any]:
    """
    Calculate semantic similarity for a single job application using PRE-TRAINED Sentence Transformer.
    Use this when someone applies to a job.
    
    This does NOT use your custom NER model for similarity - only for entity extraction (done elsewhere).
    """
    ranker = get_semantic_ranker()
    
    if not ranker.is_available():
        return {'semantic_score': 0.0, 'enhanced': False, 'reason': 'model_unavailable'}
    
    try:
        # Get job and resume
        job = mongo_collections['JOBS'].find_one({"_id": ObjectId(job_id)})
        resume = mongo_collections['resumeFetchedData'].find_one({"_id": ObjectId(resume_id)})
        
        if not job or not resume:
            return {'semantic_score': 0.0, 'enhanced': False, 'reason': 'missing_data'}
        
        # Extract content and calculate similarity using PRE-TRAINED Sentence Transformer
        job_text = ranker.extract_job_requirements(job)
        resume_text = ranker.extract_resume_content(resume)
        semantic_score = ranker.calculate_semantic_similarity(job_text, resume_text)
        
        return {
            'semantic_score': semantic_score,
            'enhanced': True,
            'job_text_length': len(job_text),
            'resume_text_length': len(resume_text),
            'model_used': ranker.model_name,
            'model_type': 'pretrained_sentence_transformer'
        }
        
    except Exception as e:
        logger.error(f"Error enhancing application with semantic score: {e}")
        return {'semantic_score': 0.0, 'enhanced': False, 'reason': str(e)}


def get_enhanced_candidate_ranking(job_id: str, mongo_collections: Dict) -> List[Dict[str, Any]]:
    """
    Get semantically enhanced candidate ranking for HR dashboard.
    Combines:
    - Traditional scores (from YOUR NER model, calculated elsewhere)
    - Semantic scores (from PRE-TRAINED Sentence Transformer, calculated here)
    """
    ranker = get_semantic_ranker()
    
    if not ranker.is_available():
        logger.warning("Semantic ranker not available, using traditional ranking")
        return get_traditional_ranking(job_id, mongo_collections)
    
    return ranker.rank_candidates_for_job(job_id, mongo_collections)


def get_traditional_ranking(job_id: str, mongo_collections: Dict) -> List[Dict[str, Any]]:
    """
    Fallback traditional ranking when semantic ranker not available.
    Uses only YOUR NER model scores (already calculated and stored).
    """
    try:
        applications = list(mongo_collections['Applied_EMP'].find({"job_id": ObjectId(job_id)}))
        
        candidates = []
        for app in applications:
            resume = mongo_collections['resumeFetchedData'].find_one({"_id": app.get('resume_id')})
            if not resume:
                continue
            
            # Get traditional score from YOUR NER model
            traditional_score = app.get('Matching_percentage', 0)
            if isinstance(traditional_score, dict):
                traditional_score = traditional_score.get('overall_score', 0)
            
            try:
                traditional_score = float(traditional_score)
            except:
                traditional_score = 0.0
            
            candidates.append({
                'application_id': str(app['_id']),
                'user_id': str(app.get('user_id')),
                'candidate_name': resume.get('Name', 'Unknown'),
                'traditional_score': round(traditional_score, 2),
                'semantic_score': 0.0,
                'combined_score': round(traditional_score, 2),
                'applied_date': app.get('applied_at'),
                'status': app.get('status', 'pending')
            })
        
        # Sort by traditional score
        candidates.sort(key=lambda x: x['traditional_score'], reverse=True)
        
        # Add rankings
        for i, candidate in enumerate(candidates):
            candidate['rank'] = i + 1
        
        return candidates
        
    except Exception as e:
        logger.error(f"Error in traditional ranking: {e}")
        return []