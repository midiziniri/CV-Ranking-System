"""resume_data_cleaner.py"""
import re
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ResumeDataCleaner:
    """Cleans and validates extracted resume data"""

    # --- ENHANCED NOISE/GENERAL PATTERNS ---
    NOISE_PATTERNS = [
        # Improved Date/Year Patterns
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',    # Dates like 01/2022, 1-1-2023
        r'^\d{4}\s*[-–]\s*(\d{4}|Present|Current)$', # Year ranges like 2020-2023 or 2020-Present
        r'^\d{2}/\d{4}\s*[-–]\s*\d{2}/\d{4}$', # Month/Year ranges
        # Numbers & Short Acronyms
        r'^\d+$',                             # Pure numbers (e.g., 2019, 203)
        r'^[A-Z]{2,4}$',                      # 2-4 letter acronyms alone (e.g., BA, IT, DV, GAS) - Increased to 4
        # Contact Info/IDs - More robust detection
        r'^\s*(\+?\d{1,3}[-.\s]?)?(\(\d{3}\)[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}(?:x\d{1,5})?$', # General phone format
        r'\d{10,}',                           # Phone numbers (10+ digits without formatting)
        r'^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$', # Basic email format
        r'(?:Page\s+\d+|[Pp]age\s*\d+\s+of\s+\d+)', # Page numbers/indicators
    ]

    # Personal info/Punctuation words (Often wrongly extracted)
    PERSONAL_INFO_PATTERNS = [
        r'\b(male|female|m|f|single|married|divorced|widowed|filipino|american|citizenship|civil\s+status|sex|birth|date\s+of\s+birth)\b',
        r'\b(applicant|name\s+of\s+father|name\s+of\s+mother|mother|father)\b', # Common non-skill/job words
        r'\b(\d{1,2} (?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b', # Birthdates with month name
        r'^\s*([.]{2,}|[-]{2,}|[*]{2,}|[:]{2,})\s*$' # Rows of punctuation
    ]

    # Address patterns (Not a skill or cert)
    ADDRESS_PATTERNS = [
        r'\b(street|st\.|avenue|ave\.|road|rd\.|blvd|boulevard|purok|barangay|brgy|village|city|municipality)\b',
        r'\b\d+\s+[A-Za-z]+\s+(street|st\.|avenue|road)\b',
        r'teachers (village|vilage)', # Specific common misspellings/phrases
        r'bonga (menor|mayor)',
    ]

    # Common non-skill words/phrases to filter out lists
    # Common non-skill words/phrases to filter out lists
    COMMON_WORDS = {
        'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'a', 'an', 'as', 'by', 'from', 'is', 'was', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'shall', 'etc', 'i', 'my',
        'real', 'world', 'difficulties', 'career', 'data', 'files', 'reports',
        'information', 'system', 'management', 'basic', 'entry', 'issue', 'issues',
        # Removed 'engineering', 'technology', 'science', 'bachelor' to preserve degree names.
        'experience', 'year', 'yr', 'month', 'mon', 'worked' # Added general context words
    }

    # Education level keywords (only for filtering out from skills, not education itself)
    EDUCATION_LEVELS_FOR_SKILLS = {
        'elementary', 'primary', 'secondary', 'tertiary'
    }

    # Certification quality patterns - must have at least one indicator
    CERTIFICATION_INDICATORS = [
        r'\b(certificate|certification|certified|credential|license|accreditation)\b',
        r'\b(training|course|program|seminar|workshop)\b',
        r'\b(issued by|provided by|from|authority)\b',
        r'\b(professional|technical|specialist|expert|associate)\b',
    ]

    # Added job title keywords for filtering lists that are likely NOT job titles
    NON_JOB_KEYWORDS = {
        'phone', 'email', 'contact', 'address', 'skill', 'certificate', 'education'
    }

    # Dynamic cleaning patterns - to avoid hardcoding resume-specific cleanup
    CLEANUP_PATTERNS = [
        r'\s{2,}',          # Replace multiple spaces with one
        r'^\s*[-•–—]\s*',   # Remove leading bullet points/dashes
        r'[\r\n]+',         # Replace newlines/carriage returns
        r'[;,]$',           # Remove trailing commas or semicolons
    ]

    def __init__(self):
        self.stats = {
            'skills_removed': 0,
            'certs_removed': 0,
            'education_removed': 0,
            'jobs_removed': 0
        }

    # ... (clean_all method remains the same) ...

    def clean_all(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all extracted resume data
        """
        cleaned = {}

        # Clean each field
        cleaned['Name'] = self._clean_name(extracted_data.get('Name'))
        cleaned['LINKEDIN LINK'] = self._clean_linkedin(extracted_data.get('LINKEDIN LINK'))
        cleaned['SKILLS'] = self._clean_skills(extracted_data.get('SKILLS', []))
        cleaned['CERTIFICATION'] = self._clean_certifications(extracted_data.get('CERTIFICATION', []))
        cleaned['EDUCATION'] = self._clean_education(extracted_data.get('EDUCATION', []))
        cleaned['WORKED AS'] = self._clean_job_titles(extracted_data.get('WORKED AS', []))
        cleaned['YEARS OF EXPERIENCE'] = self._clean_experience(extracted_data.get('YEARS OF EXPERIENCE'))
        cleaned['CALCULATED_EXPERIENCE_YEARS'] = extracted_data.get('CALCULATED_EXPERIENCE_YEARS', 0.0)

        # Log cleaning stats
        logger.info(f"Cleaning stats: {self.stats}")

        return cleaned

    def _is_noise(self, text: str) -> bool:
        """Check if text matches noise patterns (dates, numbers, short words)"""
        text_clean = self._normalize_text(text)

        if not text_clean:
            return True

        # *** FIX 2: EXPLICITLY ALLOW COMMON 2-LETTER EDUCATION/TECH ACRONYMS ***
        # If it's a 2-letter word, and it's a known degree/field, it's NOT noise.
        if len(text_clean) == 2 and text_clean.upper() in ['IT', 'BA', 'BS', 'MS', 'MA']:
            return False

        # Check noise patterns
        for pattern in self.NOISE_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True

        # Check if too short after cleanup (e.g. single letters/short words)
        # Only check for length < 3 now, since we handled length=2 above
        if len(text_clean) < 3:
              return True
        
        # Check if only common words
        words = text_clean.lower().split()
        if all(word in self.COMMON_WORDS for word in words) and len(words) < 4:
            # Short phrases consisting only of common words are likely noise
            return True
        
        # Check if primarily non-alpha characters
        if sum(c.isalpha() for c in text_clean) / max(1, len(text_clean)) < 0.3:
            return True

        return False

    def _is_personal_info(self, text: str) -> bool:
        """Check if text is personal information or a name (not a skill/cert)"""
        text_lower = text.lower()

        for pattern in self.PERSONAL_INFO_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        # Check if it's a full name (Title Case with multiple words)
        words = text.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w) and len(text) < 50:
             # Check if it contains job/skill keywords, if not, likely a name
             if not any(kw in text_lower for kw in ['skill', 'cert', 'analysis', 'developer', 'engineer', 'manager']):
                 return True

        return False

    def _is_address(self, text: str) -> bool:
        """Check if text is an address"""
        text_lower = text.lower()
        if '@' in text: # Catches emails which are often tagged with addresses
            return True
        for pattern in self.ADDRESS_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def _normalize_text(self, text: str) -> str:
        """Applies dynamic cleanup patterns and final strip/lower/title case"""
        if not text:
            return ""

        for pattern in self.CLEANUP_PATTERNS:
            text = re.sub(pattern, ' ', text).strip()

        return text

    # --- ENHANCED FIELD CLEANERS ---

    def _clean_name(self, name: Optional[str]) -> Optional[str]:
        """Clean and validate name"""
        if not name:
            return None

        # Apply normalization/cleanup
        name = self._normalize_text(name)
        
        # Re-check noise/personal info after cleanup, as extraction might be messy
        if self._is_noise(name) or self._is_personal_info(name):
             return None

        # Existing checks
        if len(name) < 3 or len(name) > 100:
            return None
        if sum(c.isdigit() for c in name) > 3:
            return None

        return name

    def _clean_linkedin(self, linkedin: Optional[str]) -> Optional[str]:
        """Clean and validate LinkedIn URL"""
        if not linkedin:
            return None

        linkedin = self._normalize_text(linkedin)

        if 'linkedin.com' in linkedin.lower():
            # Basic URL sanitation (e.g., ensure it starts with http/https if possible)
            if not re.match(r'https?://', linkedin, re.IGNORECASE):
                linkedin = f"http://{linkedin}"
            return linkedin

        return None

    def _clean_list(self, items: Optional[List[str]], category: str, additional_filter: Optional[callable] = None) -> Optional[List[str]]:
        """Generic list cleaner to avoid repeated code."""
        if not items:
            return None

        cleaned = []
        seen_lower = set() 
        stat_key = f'{category}_removed'

        for item in items:
            original_item = item.strip()
            item = self._normalize_text(original_item)

            if not item:
                self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                continue

            item_lower = item.lower()
            if item_lower in seen_lower:
                self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                logger.debug(f"Duplicate removed ({category}): {item}")
                continue
            
            seen_lower.add(item_lower)


            # General Filters
            if self._is_noise(item) or self._is_personal_info(item) or self._is_address(item):
                self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                continue
            
            # Category-Specific Filters (Length, format, etc.)
            if category == 'skills':
                # Be more lenient with skills - only remove obvious non-skills
                if len(item) > 100:  # Increased from 80
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
                
                # Only filter out pure education-level tags (elementary, primary, etc.)
                if item.lower() in self.EDUCATION_LEVELS_FOR_SKILLS:
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue

                if len(item) == 2 and item.upper() in ['IT', 'AI', 'ML', 'UI', 'UX', 'DB', 'OS', 'QA', 'CI', 'CD']:
                    pass  # Don't reject these
                elif len(item) < 2:
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue

            elif category == 'certs':
                # BE STRICT: Certifications should be substantial and legitimate
                if len(item) < 10:  # Minimum 10 characters for valid certification
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
                
                # Must contain at least one certification indicator
                has_cert_indicator = any(
                    re.search(pattern, item, re.IGNORECASE) 
                    for pattern in self.CERTIFICATION_INDICATORS
                )
                
                # If no indicators, check if it looks like a proper certification name
                # (contains org name, year, or specific cert format)
                looks_like_cert = (
                    re.search(r'\b(CompTIA|AWS|Microsoft|Cisco|Google|Oracle|PMP|ITIL)\b', item, re.IGNORECASE) or
                    re.search(r'\d{4}', item) or  # Contains a year
                    re.search(r'\b[A-Z]{2,}[\s-]+\d+\b', item)  # Format like "AWS-123" or "IT 101"
                )
                
                if not has_cert_indicator and not looks_like_cert:
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
            
            elif category == 'education':
                # Be LENIENT with education - accept school names, degrees, etc.
                if len(item) < 3:  # Only reject very short items
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue

                if re.match(r'^\d{2,4}$', item):
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
                
                # Don't reject education entries - they can be varied
                # Just ensure they're not personal info (already checked above)
            
            elif category == 'jobs':
                # Job titles are usually 3-70 characters (increased upper limit)
                if len(item) < 3 or len(item) > 70:
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
                
                # Skip only obvious non-job indicators
                if any(kw in item.lower() for kw in self.NON_JOB_KEYWORDS):
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue
                
                # Check for "Name of Father/Mother" type generic personal info missed by main filter
                if re.search(r'\b(Name\s+of\s+(Father|Mother)|Applicant)\b', item, re.IGNORECASE):
                    self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                    continue


            # Apply optional custom filter if provided
            if additional_filter and not additional_filter(item):
                self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
                continue

            cleaned.append(item)

        # Remove duplicates while preserving order
        cleaned = list(dict.fromkeys(cleaned))

        return cleaned if cleaned else None

    # Redefine public cleaning methods to use the generic cleaner

    def _clean_skills(self, skills: Optional[List[str]]) -> Optional[List[str]]:
        """Clean skills list using the generic list cleaner."""
        return self._clean_list(skills, 'skills')

    def _clean_certifications(self, certs: Optional[List[str]]) -> Optional[List[str]]:
        """Clean certifications list using the generic list cleaner."""
        return self._clean_list(certs, 'certs')

    def _clean_education(self, education: Optional[List[str]]) -> Optional[List[str]]:
        """Clean education list using the generic list cleaner."""
        return self._clean_list(education, 'education')

    def _clean_job_titles(self, jobs: Optional[List[str]]) -> Optional[List[str]]:
        """Clean job titles list using the generic list cleaner."""
        return self._clean_list(jobs, 'jobs')

    def _clean_experience(self, experience: Optional[List[str]]) -> Optional[List[str]]:
        """Clean years of experience - (Remains mostly as you had it, focused on numbers/keywords)"""
        if not experience:
            return None

        cleaned = []

        for exp in experience:
            exp = self._normalize_text(exp)
            exp_lower = exp.lower()

            if not exp:
                continue

            # Must contain a number
            if not re.search(r'\d+', exp):
                continue

            # Skip if it's clearly not experience (e.g., "HIGH SCHOOL")
            if any(level in exp_lower for level in self.EDUCATION_LEVELS_FOR_SKILLS):
                continue

            # Must contain year/month indicator or just be a number (already covered by initial checks if it's not noise)
            if re.search(r'\b(year|yr|month|mon|experience)\b', exp_lower) or re.match(r'^\d{1,3}$', exp):
                 cleaned.append(exp)

        # Remove duplicates while preserving order
        cleaned = list(dict.fromkeys(cleaned))

        return cleaned if cleaned else None


# Convenience function for easy import
def clean_resume_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean extracted resume data
    """
    cleaner = ResumeDataCleaner()
    return cleaner.clean_all(extracted_data)