"""
resume_processor.py
Intelligent Resume Processing Module with Statistical Validation

This module provides adaptive, ML-inspired resume parsing that works across
diverse resume formats without hardcoded rules.
"""

import re
import math
import logging

logger = logging.getLogger(__name__)


class IntelligentResumeProcessor:
    """
    Advanced resume processor using statistical validation and confidence scoring
    """
    
    def __init__(self):
        self.confidence_threshold_name = 0.5
        self.confidence_threshold_skills = 0.4
        self.confidence_threshold_education = 0.45
        self.confidence_threshold_certification = 0.45
    
    # ========== TEXT PREPROCESSING ==========
    
    @staticmethod
    def clean_resume_text(text):
        """Universal text cleaning that adapts to content"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'[ \t]+', ' ', text)  # Horizontal whitespace
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        
        # Remove non-printable characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\@\-\(\)\+\n\:\;\&\/\\]', ' ', text)
        
        # Fix common extraction artifacts
        text = re.sub(r'\|+', ' ', text)  # Pipe symbols
        text = re.sub(r'_{2,}', ' ', text)  # Multiple underscores
        
        return text.strip()
    
    @staticmethod
    def preprocess_for_nlp(text):
        """Intelligent preprocessing that maintains semantic structure"""
        # Preserve important patterns before cleaning
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        url_pattern = r'https?://[^\s]+'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        urls = re.findall(url_pattern, text)
        
        # Clean text
        text = IntelligentResumeProcessor.clean_resume_text(text)
        
        # Re-insert preserved patterns with clear spacing
        for email in emails:
            text = text.replace(email, f' {email} ')
        for phone in phones:
            text = text.replace(phone, f' {phone} ')
        for url in urls:
            text = text.replace(url, f' {url} ')
        
        # Normalize spacing
        text = ' '.join(text.split())
        
        return text
    
    # ========== STATISTICAL ANALYSIS ==========
    
    @staticmethod
    def calculate_text_entropy(text):
        """Calculate entropy to measure text randomness (lower = more structured)"""
        if not text:
            return 0
        text = text.lower()
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        entropy = 0
        text_len = len(text)
        for freq in char_freq.values():
            probability = freq / text_len
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def extract_contextual_features(text, entity_text):
        """Extract features from the context around an entity"""
        try:
            pos = text.lower().find(entity_text.lower())
            if pos == -1:
                return {}
            
            # Get surrounding context (50 chars before and after)
            context_before = text[max(0, pos-50):pos].lower()
            context_after = text[pos+len(entity_text):pos+len(entity_text)+50].lower()
            
            # Calculate features
            features = {
                'position_ratio': pos / len(text),  # Relative position in document
                'has_colon_before': ':' in context_before[-5:],  # Label pattern
                'has_bullet_before': '•' in context_before or '-' in context_before[-3:],
                'has_newline_before': '\n' in context_before[-10:],
                'capitalized_words': sum(1 for word in entity_text.split() if word and word[0].isupper()),
                'word_count': len(entity_text.split()),
                'char_count': len(entity_text),
                'has_numbers': bool(re.search(r'\d', entity_text)),
                'has_special_chars': bool(re.search(r'[^\w\s]', entity_text)),
                'entropy': IntelligentResumeProcessor.calculate_text_entropy(entity_text),
            }
            return features
        except:
            return {}
    
    @staticmethod
    def calculate_confidence_score(entity_text, entity_type, text_context, all_entities):
        """
        Calculate confidence score using multiple statistical signals
        Returns score between 0 and 1
        """
        score = 0.5  # Start neutral
        features = IntelligentResumeProcessor.extract_contextual_features(text_context, entity_text)
        
        if not features:
            return 0.5
        
        # Universal quality checks
        char_count = features['char_count']
        word_count = features['word_count']
        
        # 1. Length-based scoring
        if char_count < 2:
            score -= 0.3
        elif 3 <= char_count <= 100:
            score += 0.1
        
        # 2. Entropy-based scoring (structured text has lower entropy)
        entropy = features['entropy']
        if entity_type in ['NAME', 'SKILLS', 'CERTIFICATION', 'EDUCATION']:
            if entropy < 3.5:  # Low entropy = structured
                score += 0.1
            elif entropy > 4.5:  # High entropy = random
                score -= 0.1
        
        # 3. Context-based scoring
        if features['has_colon_before']:
            score += 0.15  # Likely a labeled field
        if features['has_bullet_before']:
            score += 0.1  # Likely a list item
        
        # 4. Position-based scoring
        pos_ratio = features['position_ratio']
        if entity_type == 'NAME' and pos_ratio < 0.15:
            score += 0.15
        elif entity_type == 'NAME' and pos_ratio > 0.5:
            score -= 0.1
        
        # 5. Format validation by entity type
        if entity_type == 'NAME':
            if 2 <= word_count <= 4 and features['capitalized_words'] >= 2:
                score += 0.2
            if features['has_numbers']:
                score -= 0.2
        
        elif entity_type in ['SKILLS', 'TECHNICAL_SKILL', 'SOFT_SKILL']:
            if 1 <= word_count <= 5 and char_count >= 3:
                score += 0.1
            skill_entities = [e for e in all_entities if e.get('type') in ['SKILLS', 'TECHNICAL_SKILL', 'SOFT_SKILL']]
            if len(skill_entities) > 3:
                score += 0.1
        
        elif entity_type == 'EDUCATION':
            edu_keywords = ['degree', 'bachelor', 'master', 'university', 'college', 'diploma', 'phd']
            text_lower = entity_text.lower()
            if any(keyword in text_lower for keyword in edu_keywords):
                score += 0.2
            if word_count >= 3:
                score += 0.1
        
        elif entity_type == 'CERTIFICATION':
            cert_keywords = ['certified', 'certificate', 'certification', 'license', 'course']
            text_lower = entity_text.lower()
            if any(keyword in text_lower for keyword in cert_keywords):
                score += 0.2
        
        # 6. Duplication penalty
        entity_count = sum(1 for e in all_entities if e.get('text', '').lower() == entity_text.lower())
        if entity_count > 3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    # ========== ENTITY EXTRACTION METHODS ==========
    
    def extract_name_intelligent(self, dic, text, fallback_processor, all_entities):
        """Extract name using statistical validation"""
        candidates = []
        
        # Get NAME entities from NLP
        resume_name = dic.get('NAME', [])
        
        # Score each candidate
        for name_candidate in resume_name:
            features = self.extract_contextual_features(text, name_candidate)
            confidence = self.calculate_confidence_score(name_candidate, 'NAME', text, all_entities)
            
            candidates.append({
                'text': name_candidate,
                'confidence': confidence,
                'position': features.get('position_ratio', 1)
            })
        
        # Sort by confidence, prefer names near top of document
        candidates.sort(key=lambda x: (x['confidence'], -x['position']), reverse=True)
        
        if candidates and candidates[0]['confidence'] >= self.confidence_threshold_name:
            logger.info(f"Name extracted with confidence {candidates[0]['confidence']:.2f}")
            return candidates[0]['text']
        
        # Fallback to processor
        processor_name = fallback_processor.extract_name_universal(text)
        if processor_name:
            logger.info(f"Used GeneralResumeProcessor for name extraction")
            return processor_name
        
        return None
    
    def extract_skills_intelligent(self, dic, text, fallback_processor, all_entities):
        """Extract skills using statistical deduplication"""
        all_skill_candidates = []
        
        # Gather from all skill-related entities
        skill_types = ['SKILLS', 'TECHNICAL_SKILL', 'SOFT_SKILL']
        for skill_type in skill_types:
            if skill_type in dic:
                all_skill_candidates.extend(dic[skill_type])
        
        # Score and filter
        scored_skills = []
        for skill in all_skill_candidates:
            confidence = self.calculate_confidence_score(skill, 'SKILLS', text, all_entities)
            if confidence >= self.confidence_threshold_skills:
                scored_skills.append({
                    'text': skill.strip().title(),
                    'confidence': confidence
                })
        
        # Deduplicate (case-insensitive)
        seen = set()
        unique_skills = []
        for skill in scored_skills:
            skill_lower = skill['text'].lower()
            if skill_lower not in seen and len(skill_lower) >= 2:
                seen.add(skill_lower)
                unique_skills.append(skill['text'])
        
        # Always augment with processor results
        processor_skills = fallback_processor.extract_skills_universal(text)
        if processor_skills:
            for skill in processor_skills:
                skill_lower = skill.lower()
                if skill_lower not in seen and len(skill_lower) >= 2:
                    seen.add(skill_lower)
                    unique_skills.append(skill.strip().title())
        
        result = unique_skills if unique_skills else None
        
        if result:
            logger.info(f"Intelligent skills extraction: {len(result)} unique skills")
        
        return result
    
    def extract_certifications_intelligent(self, dic, text, fallback_processor, all_entities):
        """Extract certifications using context-aware filtering"""
        cert_candidates = dic.get('CERTIFICATION', [])
        
        scored_certs = []
        for cert in cert_candidates:
            confidence = self.calculate_confidence_score(cert, 'CERTIFICATION', text, all_entities)
            if confidence >= self.confidence_threshold_certification:
                scored_certs.append({
                    'text': cert.strip(),
                    'confidence': confidence
                })
        
        # Deduplicate
        seen = set()
        unique_certs = []
        for cert in scored_certs:
            cert_lower = cert['text'].lower()
            if cert_lower not in seen and len(cert_lower) >= 5:
                seen.add(cert_lower)
                unique_certs.append(cert['text'])
        
        # Augment with processor
        processor_certs = fallback_processor.extract_certifications_universal(text)
        if processor_certs:
            for cert in processor_certs:
                cert_lower = cert.lower()
                if cert_lower not in seen and len(cert_lower) >= 5:
                    seen.add(cert_lower)
                    unique_certs.append(cert.strip())
        
        result = unique_certs if unique_certs else None
        
        if result:
            logger.info(f"Intelligent certification extraction: {len(result)} unique certifications")
        
        return result
    
    def extract_education_intelligent(self, dic, text, fallback_processor, all_entities):
        """Extract education using semantic validation"""
        edu_candidates = []
        
        # Get from multiple sources
        if 'EDUCATION' in dic:
            edu_candidates.extend(dic['EDUCATION'])
        if 'DEGREE' in dic:
            edu_candidates.extend(dic['DEGREE'])
        if 'UNIVERSITY' in dic:
            edu_candidates.extend(dic['UNIVERSITY'])
        
        scored_education = []
        for edu in edu_candidates:
            confidence = self.calculate_confidence_score(edu, 'EDUCATION', text, all_entities)
            if confidence >= self.confidence_threshold_education:
                scored_education.append({
                    'text': edu.strip(),
                    'confidence': confidence
                })
        
        # Deduplicate
        seen = set()
        unique_education = []
        for edu in scored_education:
            edu_lower = edu['text'].lower()
            if edu_lower not in seen and len(edu_lower) >= 5:
                seen.add(edu_lower)
                unique_education.append(edu['text'])
        
        # Augment with processor
        processor_education = fallback_processor.extract_education_universal(text)
        if processor_education:
            for edu in processor_education:
                edu_lower = edu.lower()
                if edu_lower not in seen and len(edu_lower) >= 5:
                    seen.add(edu_lower)
                    unique_education.append(edu.strip())
        
        result = unique_education if unique_education else None
        
        if result:
            logger.info(f"Intelligent education extraction: {len(result)} unique entries")
        
        return result
    
    # ========== MAIN PROCESSING METHOD ==========
    
    def process_nlp_entities(self, nlp_doc, text, fallback_processor):
        """
        Main processing method that takes NLP output and returns cleaned entities
        
        Args:
            nlp_doc: spaCy Doc object from NLP model
            text: Original resume text
            fallback_processor: GeneralResumeProcessor instance for fallback
        
        Returns:
            dict: Cleaned and validated entities
        """
        dic = {}
        all_entities_with_metadata = []
        
        # Extract entities from NLP model
        for ent in nlp_doc.ents:
            all_entities_with_metadata.append({
                'type': ent.label_,
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Build dictionary with confidence filtering
        for entity in all_entities_with_metadata:
            entity_text = entity['text'].strip()
            entity_type = entity['type']
            
            if len(entity_text) > 0:
                # Calculate confidence
                confidence = self.calculate_confidence_score(
                    entity_text, 
                    entity_type, 
                    text, 
                    all_entities_with_metadata
                )
                
                # Only add if confidence is above threshold
                if confidence >= 0.4:  # Base threshold
                    if entity_type in dic:
                        dic[entity_type].append(entity_text)
                    else:
                        dic[entity_type] = [entity_text]
        
        logger.info(f"NLP entities after confidence filtering: {dic.keys()}")
        
        # Extract specific fields using intelligent methods
        result = {
            'NAME': self.extract_name_intelligent(dic, text, fallback_processor, all_entities_with_metadata),
            'SKILLS': self.extract_skills_intelligent(dic, text, fallback_processor, all_entities_with_metadata),
            'CERTIFICATION': self.extract_certifications_intelligent(dic, text, fallback_processor, all_entities_with_metadata),
            'EDUCATION': self.extract_education_intelligent(dic, text, fallback_processor, all_entities_with_metadata),
            'LINKEDIN LINK': dic.get('LINKEDIN LINK', [None])[0] if dic.get('LINKEDIN LINK') else None,
            'WORKED AS': dic.get('WORKED AS') or dic.get('JOB_TITLE'),
            'YEARS OF EXPERIENCE': dic.get('YEARS OF EXPERIENCE'),
            'raw_entities': dic,  # Keep raw entities for reference
            'all_entities_metadata': all_entities_with_metadata
        }
        
        # Clean LinkedIn link
        if result['LINKEDIN LINK']:
            result['LINKEDIN LINK'] = re.sub('\n', '', result['LINKEDIN LINK'])
        
        return result


class ResumeTextExtractor:
    """
    Utility class for extracting text from various resume file formats
    """
    
    @staticmethod
    def extract_from_pdf(filepath):
        """Extract text from PDF file"""
        import fitz  # PyMuPDF
        with fitz.open(filepath) as doc:
            text = "".join([page.get_text() for page in doc])
        return text
    
    @staticmethod
    def extract_from_docx(filepath):
        """Extract text from DOCX file"""
        import docx
        doc = docx.Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    @staticmethod
    def extract_text(filepath):
        """
        Universal text extractor that auto-detects file type
        
        Args:
            filepath: Path to resume file
            
        Returns:
            str: Extracted text
            
        Raises:
            ValueError: If file format is unsupported or extraction fails
        """
        if filepath.lower().endswith('.pdf'):
            return ResumeTextExtractor.extract_from_pdf(filepath)
        elif filepath.lower().endswith('.docx'):
            return ResumeTextExtractor.extract_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")


# ========== CONVENIENCE FUNCTION ==========

def process_resume_with_nlp(nlp_model, resume_text, fallback_processor):
    """
    Convenience function to process resume in one call
    
    Args:
        nlp_model: Trained spaCy NLP model
        resume_text: Raw resume text
        fallback_processor: GeneralResumeProcessor instance
        
    Returns:
        dict: Extracted and validated resume entities
    """
    # Initialize processor
    processor = IntelligentResumeProcessor()
    
    # Preprocess text
    cleaned_text = processor.preprocess_for_nlp(resume_text)
    
    # Run NLP model
    nlp_doc = nlp_model(cleaned_text)
    
    # Process and validate entities
    result = processor.process_nlp_entities(nlp_doc, cleaned_text, fallback_processor)
    
    return result, cleaned_text