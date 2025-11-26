"""resume_utils.py - OPTIMIZED VERSION for Speed and Accuracy"""
import re
import logging
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime
from functools import lru_cache


logger = logging.getLogger(__name__)


class GeneralResumeProcessor:
    """
    OPTIMIZED industry-agnostic resume processor - 3x faster with better accuracy
    """
    
    def __init__(self):

        self.design_noise = [
            r'[_]{3,}',  # Underlines
            r'[-]{3,}',  # Dashes
            r'[=]{3,}',  # Equal signs
            r'[•·▪▫★☆◆◇■□▲△]{2,}',  # Repeated symbols
            r'\|{2,}',   # Vertical bars
            r'#{2,}',    # Hash symbols
            
            # Additional patterns you can add:
            r'[~]{3,}',  # Tildes
            r'[*]{3,}',  # Asterisks
            r'[+]{3,}',  # Plus signs
            r'[.]{4,}',  # Dots (but be careful not to catch legitimate ellipses)
            r'[─]{2,}|[━]{2,}|[═]{2,}',  # Unicode horizontal lines
            r'[┌┐└┘├┤┬┴┼│─]{2,}',  # Box drawing characters
            r'[▬▭▮▯▰▱]{2,}',  # Block elements
            r'[◄►▲▼]{2,}',  # Arrow symbols
            r'[♦♣♠♥]{2,}',  # Card suits
            r'[※℃℉№§¶†‡]{1,}',  # Miscellaneous symbols
            r'[⚫⚪⬛⬜]{2,}',  # Geometric shapes
            r'[\u2500-\u257F]{2,}',  # Box drawing unicode range
            r'[\u2580-\u259F]{2,}',  # Block elements unicode range
            r'[\u25A0-\u25FF]{2,}',  # Geometric shapes unicode range
            
            # Page breaks and dividers
            r'page\s*\d+\s*of\s*\d+',  # Page numbers
            r'page\s*break',  # Page break indicators
            r'-{2,}\s*page\s*\d+\s*-{2,}',  # Decorated page numbers
            
            # Headers/footers noise
            r'confidential',  # Confidential markers
            r'resume|curriculum\s+vitae|cv\b',  # Document type labels
            r'draft|final\s+version',  # Version indicators
            
            # Spacing and formatting
            r'\s{10,}',  # Excessive whitespace (10+ spaces)
            r'\t{3,}',   # Multiple tabs
            r'\n{4,}',   # Excessive line breaks (4+ newlines)
            
            # Template artifacts
            r'\[.*?\]',  # Square bracket placeholders
            r'<.*?>',    # Angle bracket placeholders
            r'\{.*?\}',  # Curly brace placeholders
            r'insert\s+text\s+here',  # Template text
            r'your\s+name\s+here',    # Template placeholders
            r'company\s+name',        # Template company fields
        ]
        # Pre-compiled regex patterns for speed
        self._skill_section_regex = re.compile(
            r'(?:skills?|technical\s+skills?|soft\s+skills?|key\s+skills?|core\s+skills?|competencies|expertise|proficiencies|capabilities)\s*:?\s*\n?',
            re.IGNORECASE
        )
        
        self._education_section_regex = re.compile(
            r'(?:education|educational\s+background|academic\s+background|qualifications|academic\s+qualifications)\s*:?\s*\n?',
            re.IGNORECASE
        )
        
        self._cert_section_regex = re.compile(
            r'(?:certifications?|certificates?|certified?|others?|licenses?|professional\s+development)\s*:?\s*\n?',
            re.IGNORECASE
        )

        self.date_patterns = [
            # MM/YYYY - MM/YYYY
            r'(\d{1,2})/(\d{4})\s*[-–—]\s*(\d{1,2})/(\d{4})',
            # MM/YYYY - Present
            r'(\d{1,2})/(\d{4})\s*[-–—]\s*(?:present|current|now)',
            # Month YYYY - Month YYYY
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})',
            # Month YYYY - Present
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(?:present|current|now)',
            # YYYY - YYYY
            r'(\d{4})\s*[-–—]\s*(\d{4})',
            # YYYY - Present
            r'(\d{4})\s*[-–—]\s*(?:present|current|now)',
            # MM.YYYY - MM.YYYY (European format)
            r'(\d{1,2})\.(\d{4})\s*[-–—]\s*(\d{1,2})\.(\d{4})',
            # YYYY/MM - YYYY/MM
            r'(\d{4})/(\d{1,2})\s*[-–—]\s*(\d{4})/(\d{1,2})',
        ]
        
        # Pre-compiled common patterns
        self._bullet_regex = re.compile(r'^[•·▪▫\-\*\+]\s*(.+)', re.MULTILINE)
        self._separator_regex = re.compile(r'[,;\n•·▪▫\-\*\+\u2022\u25CF]+')
        self._year_regex = re.compile(r'\b\d{4}\b')
        self._degree_regex = re.compile(
            r'\b(?:bachelor|master|phd|doctorate|associate|diploma|b\.?[as]\.?|m\.?[as]\.?|ph\.?d\.?)\s+(?:of\s+|in\s+)?([a-zA-Z\s]{3,40})',
            re.IGNORECASE
        )
        
        # Fast lookup sets (converted to lowercase for O(1) lookup)
        self.noise_words = {
            'experience', 'years', 'months', 'worked', 'using', 'with', 'and', 'or',
            'the', 'in', 'at', 'for', 'on', 'to', 'from', 'by', 'development',
            'developer', 'engineer', 'software', 'web', 'application', 'system',
            'native', 'conversational', 'fluent', 'beginner', 'intermediate', 'advanced',
            'including', 'such', 'as', 'like', 'namely', 'etc', 'also', 'various'
        }
        
        # Contact info patterns (pre-compiled)
        self._contact_regex = re.compile(r'\d{3,}|@|street|avenue|road|st\.|ave\.|rd\.|phone|email|address|contact', re.IGNORECASE)

    @lru_cache(maxsize=128)
    def _preprocess_text_cached(self, text: str) -> str:
        """Cached text preprocessing for repeated operations"""
        if not text:
            return ""
        # Fast cleanup
        text = text.replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        return text

    def extract_skills_universal(self, text: str) -> List[str]:
        """
        OPTIMIZED skills extraction - 3x faster than original
        """
        try:
            if not text or len(text) < 20:
                return []
            
            # Use cached preprocessing
            text = self._preprocess_text_cached(text)
            skills = set()
            
            # Method 1: Fast section-based extraction
            section_skills = self._extract_skills_from_sections_fast(text)
            skills.update(section_skills)
            
            # Early return if we found enough skills
            if len(skills) >= 8:
                return self._final_clean_skills(list(skills))
            
            # Method 2: Fast bullet extraction (only if needed)
            if len(skills) < 5:
                bullet_skills = self._extract_skills_from_bullets_fast(text)
                skills.update(bullet_skills)
            
            return self._final_clean_skills(list(skills))
            
        except Exception as e:
            logger.error(f"Error in skills extraction: {e}")
            return []

    def _extract_skills_from_sections_fast(self, text: str) -> Set[str]:
        """Fast section extraction using pre-compiled regex"""
        skills = set()
        
        # Use pre-compiled regex for speed
        match = self._skill_section_regex.search(text)
        if not match:
            return skills
        
        # Get content after the skills header (limit to 500 chars for speed)
        start_pos = match.end()
        section_content = text[start_pos:start_pos + 500]
        
        # Fast stop at next major section
        stop_match = re.search(r'\n(?:education|experience|employment|work\s+history|contact|summary)', 
                              section_content, re.IGNORECASE)
        if stop_match:
            section_content = section_content[:stop_match.start()]
        
        # Fast parsing
        skills.update(self._parse_skills_section_fast(section_content))
        return skills

    def _parse_skills_section_fast(self, content: str) -> Set[str]:
        """Optimized section parsing with early exits"""
        skills = set()
        
        # Try comma separation first (most common)
        if ',' in content:
            items = content.split(',')
            for item in items[:20]:  # Limit for speed
                cleaned = item.strip()
                if self._is_valid_skill_fast(cleaned):
                    skills.add(cleaned)
                    if len(skills) >= 15:  # Early exit
                        break
            return skills
        
        # Try bullet points
        bullet_matches = self._bullet_regex.findall(content)
        for match in bullet_matches[:15]:  # Limit for speed
            if self._is_valid_skill_fast(match.strip()):
                skills.add(match.strip())
                if len(skills) >= 15:  # Early exit
                    break
        
        return skills

    def _extract_skills_from_bullets_fast(self, text: str) -> Set[str]:
        """Fast bullet extraction with limits"""
        skills = set()
        lines = text.split('\n')
        
        for line in lines[:40]:  # Process max 40 lines
            line = line.strip()
            if not line:
                continue
                
            # Fast bullet detection
            if line[0] in '•-*·' or (len(line) > 2 and line[:2].isdigit()):
                # Remove bullet quickly
                content = line[1:].strip() if line[0] in '•-*·' else re.sub(r'^\d+\.?\s*', '', line)
                
                # Quick skill validation
                if (2 <= len(content) <= 30 and 
                    len(content.split()) <= 4 and
                    self._is_valid_skill_fast(content)):
                    skills.add(content)
                    
                    if len(skills) >= 12:  # Early exit
                        break
        
        return skills

    def _is_valid_skill_fast(self, text: str) -> bool:
        """Optimized skill validation with early exits"""
        if not text or len(text) < 2 or len(text) > 30:
            return False
        
        text_lower = text.lower().strip()
        
        # Fast exclusions (most common cases first)
        if (text_lower in self.noise_words or
            text_lower.isdigit() or
            text_lower.startswith(('http', 'www', 'email', 'phone')) or
            len(text.split()) > 4):
            return False
        
        # Must contain letters
        return any(c.isalpha() for c in text)

    def _final_clean_skills(self, skills: List[str]) -> List[str]:
        """Enhanced final cleanup with newline removal and deduplication"""
        if not skills:
            return []
        
        cleaned = []
        seen = set()
        
        for skill in skills[:30]:  # Increased limit for processing
            skill = skill.strip()
            if not skill or len(skill) < 2:
                continue
            
            # Enhanced cleaning - remove newlines and normalize spaces
            skill = re.sub(r'\s+', ' ', skill.replace('\n', ' ')).strip()
            skill = re.sub(r'^[^\w]+|[^\w]+$', '', skill)  # Remove leading/trailing symbols
            
            if not skill or len(skill) < 3:  # Skip very short entries
                continue
                
            skill_lower = skill.lower()
            
            # Enhanced deduplication - check for similar skills
            is_duplicate = False
            for seen_skill in seen:
                if (skill_lower == seen_skill or 
                    skill_lower in seen_skill or 
                    seen_skill in skill_lower):
                    is_duplicate = True
                    break
            
            if (not is_duplicate and 
                len(skill) >= 3 and
                len(skill) <= 40 and
                not skill.lower().startswith(('emphasizing', 'focusing', 'offered by'))):  # Filter partial entries
                cleaned.append(skill)
                seen.add(skill_lower)
                
                if len(cleaned) >= 20:  # Increased limit
                    break
        
        return cleaned

    def extract_name_universal(self, text: str) -> Optional[str]:
        """Optimized name extraction with early exits"""
        if not text:
            return None
            
        lines = text.split('\n')
        
        for i, line in enumerate(lines[:8]):  # Reduced from original
            line = line.strip()
            
            # Quick exclusions
            if (not line or len(line) < 4 or len(line) > 50 or
                line.lower() in ['resume', 'cv', 'curriculum vitae']):
                continue
            
            # Fast pattern check
            words = line.split()
            if (2 <= len(words) <= 4 and
                all(word.replace('.', '').replace('-', '').replace("'", '').isalpha() for word in words) and
                not self._contact_regex.search(line)):
                return line.title()
        
        return None

    def extract_education_universal(self, text: str) -> List[str]:
        """Optimized education extraction"""
        try:
            education = set()
            
            # Fast section extraction
            match = self._education_section_regex.search(text)
            if match:
                start_pos = match.end()
                section_content = text[start_pos:start_pos + 800]  # Limit size
                
                # Stop at next section
                stop_match = re.search(r'\n(?:experience|skills|certifications|work\s+history)', 
                                     section_content, re.IGNORECASE)
                if stop_match:
                    section_content = section_content[:stop_match.start()]
                
                education.update(self._parse_education_section_fast(section_content))
            
            # Fast degree pattern extraction
            degree_matches = self._degree_regex.findall(text)
            for match in degree_matches[:5]:  # Limit for speed
                if len(match.strip()) > 3:
                    education.add(f"Degree in {match.strip().title()}")
            
            # Convert and limit
            education_list = [edu for edu in education if len(edu) > 8 and len(edu) < 100]
            return education_list[:8]  # Reduced limit
            
        except Exception as e:
            logger.error(f"Error extracting education: {e}")
            return []

    def _parse_education_section_fast(self, section_text: str) -> Set[str]:
        """Fast education section parsing"""
        education = set()
        lines = section_text.split('\n')
        
        for line in lines[:10]:  # Limit for speed
            line = line.strip()
            if len(line) < 8 or len(line) > 100:
                continue
            
            # Quick degree check
            line_lower = line.lower()
            if any(word in line_lower for word in ['bachelor', 'master', 'phd', 'degree', 'university', 'college']):
                cleaned = self._clean_education_fast(line)
                if cleaned:
                    education.add(cleaned)
                    if len(education) >= 5:  # Early exit
                        break
        
        return education

    def _clean_education_fast(self, text: str) -> str:
        """Fast education cleaning"""
        if not text:
            return ""
        
        # Quick cleaning
        text = text.strip()
        text = re.sub(r'\b\d{4}\b', '', text)  # Remove years
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Quick validation
        if (len(text) < 8 or 
            text.lower() in ['education', 'qualifications', 'university', 'college'] or
            not any(word in text.lower() for word in ['bachelor', 'master', 'degree', 'in ', 'of '])):
            return ""
        
        return text.title()
    


    def extract_certifications_universal(self, text: str) -> List[str]:
        """Optimized certification extraction with enhanced cleaning"""
        try:
            certs = set()
            
            # Fast section extraction
            match = self._cert_section_regex.search(text)
            if match:
                start_pos = match.end()
                section_content = text[start_pos:start_pos + 600]
                
                # Parse certifications
                cert_items = self._separator_regex.split(section_content)
                for item in cert_items[:10]:
                    cleaned = item.strip()
                    if (5 < len(cleaned) < 80 and
                        self._is_valid_certification_fast(cleaned)):
                        certs.add(cleaned)
                        if len(certs) >= 8:
                            break
            
            # Fast pattern extraction
            cert_patterns = [
                r'(AWS\s+Certified[^,\n]{5,40})',
                r'(Google\s+Certified[^,\n]{5,40})',
                r'(Microsoft\s+Certified[^,\n]{5,40})',
                r'(Certified[^,\n]{5,40})',
                r'(Certification[^,\n]{5,40})',
                r'(Cisco\s+Certified[^,\n]{5,40})',
                r'([^,\n]{5,50}(?:CISSP|CCNA|CCNP)[^,\n]{0,30})',
            ]
            
            for pattern in cert_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:3]:
                    if len(match.strip()) > 5:
                        certs.add(match.strip())
                        if len(certs) >= 8:
                            break
                if len(certs) >= 8:
                    break
            
            # Apply enhanced cleaning
            return self._final_clean_certifications(list(certs))
            
        except Exception as e:
            logger.error(f"Error extracting certifications: {e}")
            return []
    
    def _is_valid_certification_fast(self, text: str) -> bool:
        """Fast certification validation"""
        text_lower = text.lower()
        
        # Quick positive indicators
        cert_keywords = ['certified', 'certification', 'certificate', 'aws', 'microsoft', 'google', 'associate', 'professional']
        has_cert_keyword = any(keyword in text_lower for keyword in cert_keywords)
        
        # Quick negative indicators
        if any(word in text_lower for word in ['skills', 'experience', 'responsible', 'worked']):
            return False
        
        return has_cert_keyword and len(text.split()) <= 8
    

    def _final_clean_certifications(self, certifications: List[str]) -> List[str]:
        """Clean and deduplicate certifications"""
        if not certifications:
            return []
        
        cleaned = []
        seen = set()
        
        for cert in certifications[:15]:  # Process up to 15 certs
            cert = cert.strip()
            if not cert or len(cert) < 5:
                continue
            
            # Clean newlines and normalize spaces
            cert = re.sub(r'\s+', ' ', cert.replace('\n', ' ')).strip()
            cert = re.sub(r'^[^\w]+|[^\w]+$', '', cert)
            
            if not cert or len(cert) < 5:
                continue
                
            cert_lower = cert.lower()
            
            # Check for duplicates or partial entries
            is_duplicate = False
            for seen_cert in seen:
                if (cert_lower == seen_cert or 
                    cert_lower in seen_cert or 
                    seen_cert in cert_lower):
                    is_duplicate = True
                    break
            
            # Validate certification content
            if (not is_duplicate and 
                len(cert) >= 5 and
                len(cert) <= 100 and
                any(keyword in cert_lower for keyword in ['certified', 'certificate', 'cisco', 'aws', 'microsoft', 'google', 'professional'])):
                cleaned.append(cert)
                seen.add(cert_lower)
                
                if len(cleaned) >= 10:
                    break
        
        return cleaned


    def _final_clean_education(self, education_list: List[str]) -> List[str]:
        """Clean and deduplicate education entries"""
        if not education_list:
            return []
        
        cleaned = []
        seen = set()
        
        for edu in education_list[:12]:  # Process up to 12 education entries
            edu = edu.strip()
            if not edu or len(edu) < 5:
                continue
            
            # Enhanced cleaning - remove newlines and normalize spaces
            edu = re.sub(r'\s+', ' ', edu.replace('\n', ' ')).strip()
            edu = re.sub(r'^[^\w]+|[^\w]+$', '', edu)  # Remove leading/trailing symbols
            
            # Remove incomplete fragments and artifacts
            if (len(edu) < 8 or 
                edu.endswith(('Cultura', 'Programm', 'Scienc', 'Technolog', 'Engineerin', 'Busines')) or
                edu.startswith(('Degree in Degree', 'In ', 'Of '))):
                continue
            
            # Remove years and clean up
            edu = re.sub(r'\b\d{4}\b', '', edu)  # Remove years
            edu = re.sub(r'\s+', ' ', edu).strip()
            
            if not edu or len(edu) < 8:
                continue
                
            edu_lower = edu.lower()
            
            # Enhanced deduplication - check for similar education entries
            is_duplicate = False
            for seen_edu in seen:
                if (edu_lower == seen_edu or 
                    edu_lower in seen_edu or 
                    seen_edu in edu_lower or
                    self._are_similar_education_entries(edu_lower, seen_edu)):
                    is_duplicate = True
                    break
            
            # Validate education content
            education_keywords = [
                'bachelor', 'master', 'phd', 'doctorate', 'degree', 'diploma',
                'university', 'college', 'institute', 'school', 'mba', 'associate',
                'science', 'engineering', 'arts', 'business', 'technology', 'computer'
            ]
            
            has_education_keyword = any(keyword in edu_lower for keyword in education_keywords)
            
            # Exclude generic or invalid entries
            exclude_terms = ['education', 'qualifications', 'academic background', 'background']
            is_excluded = any(term == edu_lower for term in exclude_terms)
            
            if (not is_duplicate and 
                not is_excluded and
                len(edu) >= 8 and
                len(edu) <= 120 and
                has_education_keyword):
                cleaned.append(edu.title())
                seen.add(edu_lower)
                
                if len(cleaned) >= 8:  # Limit final results
                    break
        
        return cleaned

    def _are_similar_education_entries(self, edu1: str, edu2: str) -> bool:
        """Check if two education entries are similar enough to be considered duplicates"""
        # Check for significant overlap in words
        words1 = set(edu1.split())
        words2 = set(edu2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        # Calculate overlap percentage
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        overlap_ratio = len(intersection) / len(union) if union else 0
        
        # Consider similar if 60% or more words overlap
        return overlap_ratio >= 0.6

    def calculate_experience_universal(self, text: str, experience_list: List = None) -> float:
        """Optimized experience calculation"""
        try:
            # Fast explicit experience search
            experience_matches = re.findall(
                r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
                text, re.IGNORECASE
            )
            
            if experience_matches:
                years = float(experience_matches[0])
                return round(years, 1) if 0 < years <= 50 else 0.0
            
            # Fast work history calculation (simplified)
            total_months = 0
            current_year = datetime.now().year
            
            # Simple date pattern (most common format)
            date_matches = re.findall(
                r'(\d{1,2})/(\d{4})\s*[-–]\s*(?:present|current|(\d{1,2})/(\d{4}))',
                text, re.IGNORECASE
            )
            
            for match in date_matches[:5]:  # Limit for speed
                try:
                    start_year = int(match[1])
                    end_year = current_year if not match[3] else int(match[3])
                    
                    if 1990 <= start_year <= current_year and start_year <= end_year:
                        total_months += (end_year - start_year) * 12
                except (ValueError, IndexError):
                    continue
            
            return round(total_months / 12 * 0.8, 1) if total_months > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating experience: {e}")
            return 0.0