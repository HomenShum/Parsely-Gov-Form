# parsely_tool/data_gatherer.py
"""Data gathering module for research assistant."""

import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import json
import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)

class DataGatherer:
    """Class for gathering data from various sources."""
    
    COMPANY_RESEARCH_CATEGORIES = {
        "basic_info": "Basic Company Information",
        "recent_news": "Recent News",
        "funding": "Funding History",
        "linkedin": "LinkedIn Profile",
        "overview": "Company Overview",
        "press": "Press Releases",
        "investors": "Investors",
        "products": "Products and Services"
    }
    
    PROSPECT_RESEARCH_CATEGORIES = {
        "background": "Professional Background",
        "education": "Education History",
        "current_role": "Current Role",
        "linkedin": "LinkedIn Profile",
        "contact": "Contact Information",
        "achievements": "Achievements",
        "publications": "Publications"
    }
    
    BANKING_RESEARCH_CATEGORIES = {
        "seed_rounds": "New Seed Round Companies",
        "research_reports": "Latest Research Reports",
        "fda_updates": "FDA Regulation Updates",
        "biotech_news": "Fierce Biotech Updates",
        "founder_contacts": "Founder Contact Information"
    }
    
    RESEARCH_CATEGORIES = {
        "general_info": "General Information",
        "form_fields": "Form Fields",
        "requirements": "Requirements",
        "documents": "Required Documents",
        "instructions": "Instructions",
        "contact": "Contact Information",
        "deadlines": "Important Deadlines",
        "fees": "Associated Fees"
    }
    
    def __init__(self, config):
        """Initialize DataGatherer with configuration."""
        self.config = config
        self.setup_chrome_driver()
        self.session_data = {
            "screenshots": [],
            "form_fields": {},
            "visited_urls": set(),
            "extracted_info": {},
            "document_content": {}
        }
    
    def setup_chrome_driver(self):
        """Setup Chrome WebDriver with optimal configuration."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def save_screenshot(self, name: str, category: str) -> str:
        """Save screenshot of current page."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("assets", f"{name}_{timestamp}", "screenshots")
        os.makedirs(base_path, exist_ok=True)
        
        filename = os.path.join(base_path, f"{category}.png")
        self.driver.save_screenshot(filename)
        self.session_data["screenshots"].append(filename)
        return filename
    
    def save_metadata(self, name: str, category: str, metadata: Dict) -> None:
        """Save metadata for research."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("assets", f"{name}_{timestamp}")
        os.makedirs(base_path, exist_ok=True)
        
        filename = os.path.join(base_path, "metadata.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {}
        
        existing_metadata[category] = metadata
        with open(filename, "w") as f:
            json.dump(existing_metadata, f, indent=4)
    
    def search_google(self, query: str, site: Optional[str] = None) -> List[str]:
        """Perform Google search and return results."""
        if site:
            search_url = f"https://www.google.com/search?q=site:{site}+{query}"
        else:
            search_url = f"https://www.google.com/search?q={query}"
        
        self.driver.get(search_url)
        
        try:
            results = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g"))
            )
            return [result.text for result in results[:5]]
        except Exception as e:
            print(f"Error in Google search: {str(e)}")
            return []
    
    def extract_text_content(self, url: str) -> str:
        """Extract text content from URL."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text()
        except Exception as e:
            print(f"Error extracting text content: {str(e)}")
            return ""
    
    def gather_research_data(self, name: str, categories: List[str], category_map: Dict[str, str], 
                             default_query_prefix: str = "", 
                             specialized_config: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        A unified method to gather research data for various entity types
        (companies, prospects, banking research, etc.)
        
        Args:
            name (str): Name/identifier for the entity or category group.
            categories (iterable): Categories to research.
            category_map (dict or similar): Map of category keys to 
                                            category-specific descriptions/queries.
            default_query_prefix (str): A prefix to add to each query (e.g. the entity name).
            specialized_config (dict): Additional query configuration for special cases.
                                       For example, for banking research categories might have 
                                       their own default queries in a dict:
                                       {
                                           "seed_rounds": {
                                               "default_query": "new seed round ...",
                                               "description": "New Seed Round Companies"
                                           }
                                       }
        
        Returns:
            dict: Structured results including screenshots, metadata, text_content.
        """
        results = {}
        
        # If no specialized config, assume category_map is a simple {key: description} dict
        # If specialized config, assume category_map keys align with specialized_config entries
        for category in categories:
            results[category] = {
                'screenshots': [],
                'metadata': {},
                'text_content': ""
            }
            
            # Determine search query
            if specialized_config and category in specialized_config:
                # Use specialized query if available
                base_query = specialized_config[category].get("default_query", "")
                category_desc = specialized_config[category].get("description", category)
            else:
                # Assume category_map is {key: description} 
                category_desc = category_map[category]
                base_query = f"{default_query_prefix} {category_desc}".strip()
            
            search_results = self.search_google(base_query)
            
            # Save screenshot
            screenshot = self.save_screenshot(name, category)
            results[category]['screenshots'].append(screenshot)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'search_query': base_query,
                'num_results': len(search_results)
            }
            results[category]['metadata'] = metadata
            self.save_metadata(name, category, metadata)
            
            # Extract text content (here it's just the search results, as before)
            results[category]['text_content'] = "\n".join(search_results)
        
        return results
    
    def get_company_info(self, company_name: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get company information based on selected categories using the unified method."""
        if categories is None:
            categories = self.COMPANY_RESEARCH_CATEGORIES.keys()
        return self.gather_research_data(company_name, categories, self.COMPANY_RESEARCH_CATEGORIES, 
                                         default_query_prefix=company_name)
    
    def research_prospect(self, prospect_name: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Research prospect based on selected categories using the unified method."""
        if categories is None:
            categories = self.PROSPECT_RESEARCH_CATEGORIES.keys()
        return self.gather_research_data(prospect_name, categories, self.PROSPECT_RESEARCH_CATEGORIES, 
                                         default_query_prefix=prospect_name)
    
    def banking_research(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform specialized banking research using the unified method."""
        if categories is None:
            categories = self.BANKING_RESEARCH_CATEGORIES.keys()
        return self.gather_research_data("banking", categories, {},
                                         specialized_config=self.BANKING_RESEARCH_CATEGORIES)
    
    def search_prospect(self, prospect_name: str) -> Dict[str, Any]:
        """
        Search for information about a prospect. This method remains a specialized, 
        structured data return as before.
        """
        try:
            prospect_info = {
                "basic_info": {
                    "name": prospect_name,
                    "title": "Professional Title",
                    "location": "City, Country",
                    "industry": "Industry"
                },
                "work_history": [
                    {
                        "company": "Current Company",
                        "title": "Current Title",
                        "duration": "2020-Present"
                    },
                    {
                        "company": "Previous Company",
                        "title": "Previous Title",
                        "duration": "2018-2020"
                    }
                ],
                "education": [
                    {
                        "institution": "University Name",
                        "degree": "Degree Name",
                        "year": "2018"
                    }
                ],
                "skills": [
                    "Skill 1",
                    "Skill 2",
                    "Skill 3"
                ]
            }
            return prospect_info
        except Exception as e:
            logger.error(f"Error searching for prospect {prospect_name}: {str(e)}")
            return {
                "error": f"Failed to retrieve prospect information: {str(e)}",
                "prospect_name": prospect_name
            }

    def __del__(self):
        """Clean up Chrome driver."""
        if hasattr(self, 'driver'):
            self.driver.quit()

    def search_general(self, query: str) -> Dict[str, Any]:
        """Perform general web search and return results."""
        search_url = f"https://www.google.com/search?q={query}"
        self.driver.get(search_url)
        
        results = []
        try:
            search_results = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g"))
            )
            
            for result in search_results[:5]:  # Get top 5 results
                try:
                    title = result.find_element(By.CSS_SELECTOR, "h3").text
                    link = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    snippet = result.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet
                    })
                except Exception as e:
                    logger.warning(f"Error extracting search result: {str(e)}")
            
            screenshot = self.save_screenshot("search", "search_results")
            return {
                "results": results,
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return {"error": str(e)}
    
    def browse_website(self, url: str) -> Dict[str, Any]:
        """Browse website and extract content."""
        if url in self.session_data["visited_urls"]:
            return {"message": "URL already visited", "url": url}
        
        try:
            self.driver.get(url)
            self.session_data["visited_urls"].add(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract text content
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Take screenshot
            screenshot = self.save_screenshot("browse", url.split('/')[-1])
            
            # Look for forms
            forms = soup.find_all('form')
            form_info = []
            for form in forms:
                form_data = self._analyze_form(form)
                if form_data:
                    form_info.append(form_data)
            
            # Extract requirements and deadlines
            requirements = self._extract_requirements(text_content)
            deadlines = self._extract_deadlines(text_content)
            
            return {
                "url": url,
                "text_content": text_content[:1000],  # First 1000 chars
                "forms_found": len(form_info),
                "forms": form_info,
                "requirements": requirements,
                "deadlines": deadlines,
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error browsing website: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_form(self, form_soup) -> Dict[str, Any]:
        """Analyze a form and its fields in detail."""
        try:
            form_data = {
                "action": form_soup.get('action', ''),
                "method": form_soup.get('method', ''),
                "fields": []
            }
            
            # Find all input fields
            for input_tag in form_soup.find_all(['input', 'select', 'textarea']):
                field_info = self._analyze_form_field(input_tag)
                if field_info:
                    form_data["fields"].append(field_info)
            
            return form_data
        except Exception as e:
            logger.error(f"Error analyzing form: {str(e)}")
            return None
    
    def _analyze_form_field(self, field_tag) -> Optional[Dict[str, Any]]:
        """Analyze a single form field in detail."""
        try:
            field_type = field_tag.get('type', field_tag.name)
            if field_type in ['hidden', 'submit', 'button']:
                return None
            
            field_info = {
                "type": field_type,
                "name": field_tag.get('name', ''),
                "id": field_tag.get('id', ''),
                "placeholder": field_tag.get('placeholder', ''),
                "required": field_tag.get('required') is not None,
                "label": self._find_field_label(field_tag),
                "validation": {
                    "pattern": field_tag.get('pattern', ''),
                    "min": field_tag.get('min', ''),
                    "max": field_tag.get('max', ''),
                    "minlength": field_tag.get('minlength', ''),
                    "maxlength": field_tag.get('maxlength', '')
                }
            }
            
            # Add options for select fields
            if field_tag.name == 'select':
                field_info["options"] = [
                    {"value": option.get('value', ''), "text": option.text.strip()}
                    for option in field_tag.find_all('option')
                ]
            
            return field_info
        except Exception as e:
            logger.error(f"Error analyzing form field: {str(e)}")
            return None
    
    def _find_field_label(self, field_tag) -> str:
        """Find the label for a form field."""
        try:
            # Check for associated label
            field_id = field_tag.get('id')
            if field_id:
                label_tag = field_tag.find_parent().find('label', attrs={'for': field_id})
                if label_tag:
                    return label_tag.text.strip()
            
            # Check for placeholder text
            placeholder = field_tag.get('placeholder')
            if placeholder:
                return placeholder
            
            # Check for aria-label
            aria_label = field_tag.get('aria-label')
            if aria_label:
                return aria_label
            
            return ""
        except Exception:
            return ""
    
    def _extract_requirements(self, text_content: str) -> List[str]:
        """Extract requirements from text content."""
        requirements = []
        
        # Common requirement patterns
        requirement_patterns = [
            r"(?i)required?:?\s*(.*?)(?:\.|$)",
            r"(?i)you (?:must|need to|should):?\s*(.*?)(?:\.|$)",
            r"(?i)requirements?:?\s*(.*?)(?:\.|$)",
            r"(?i)eligibility:?\s*(.*?)(?:\.|$)"
        ]
        
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text_content)
            for match in matches:
                requirement = match.group(1).strip()
                if requirement and len(requirement) > 10:  # Avoid very short matches
                    requirements.append(requirement)
        
        return list(set(requirements))  # Remove duplicates
    
    def _extract_deadlines(self, text_content: str) -> List[str]:
        """Extract deadlines from text content."""
        deadlines = []
        
        # Common deadline patterns
        deadline_patterns = [
            r"(?i)deadline:?\s*(.*?)(?:\.|$)",
            r"(?i)due (?:date|by):?\s*(.*?)(?:\.|$)",
            r"(?i)closes?:?\s*(?:on|by)?\s*(.*?)(?:\.|$)",
            r"(?i)submit (?:by|before):?\s*(.*?)(?:\.|$)"
        ]
        
        for pattern in deadline_patterns:
            matches = re.finditer(pattern, text_content)
            for match in matches:
                deadline = match.group(1).strip()
                if deadline and len(deadline) > 5:  # Avoid very short matches
                    deadlines.append(deadline)
        
        return list(set(deadlines))  # Remove duplicates
    
    def handle_form(self, url: str, fields: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Handle form fields on a webpage."""
        try:
            self.driver.get(url)
            
            # Find all form fields
            form_elements = self.driver.find_elements(By.TAG_NAME, "form")
            forms_data = []
            
            for form in form_elements:
                form_data = self._analyze_form(BeautifulSoup(form.get_attribute('outerHTML'), 'html.parser'))
                if form_data:
                    forms_data.append(form_data)
            
            # Store form fields in session data
            self.session_data["form_fields"][url] = forms_data
            
            # Take screenshot of the form
            screenshot = self.save_screenshot("form", "form_fields")
            
            return {
                "url": url,
                "forms": forms_data,
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error handling form: {str(e)}")
            return {"error": str(e)}
    
    def recommend_fill(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for form fields."""
        try:
            recommendations = {}
            form_fields = context.get("form_fields", [])
            user_data = context.get("user_data", {})
            document_content = self.session_data.get("document_content", {})
            
            for field in form_fields:
                field_type = field.get("type", "")
                field_name = field.get("name", "").lower()
                field_label = field.get("label", "").lower()
                
                # Try to find relevant information in documents
                doc_recommendation = self._find_in_documents(
                    field_name, 
                    field_label, 
                    document_content
                )
                
                if doc_recommendation:
                    recommendations[field_name] = doc_recommendation
                    continue
                
                # Generate recommendations based on field type and name
                if field_type == "email":
                    recommendations[field_name] = user_data.get("email", "")
                elif "name" in field_name:
                    if "first" in field_name:
                        recommendations[field_name] = user_data.get("first_name", "")
                    elif "last" in field_name:
                        recommendations[field_name] = user_data.get("last_name", "")
                elif "phone" in field_name:
                    recommendations[field_name] = user_data.get("phone", "")
                elif "address" in field_name:
                    recommendations[field_name] = user_data.get("address", "")
            
            return {
                "recommendations": recommendations,
                "missing_info": [field for field in form_fields 
                               if field.get("name", "").lower() not in recommendations]
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    def _find_in_documents(self, field_name: str, field_label: str, document_content: Dict[str, str]) -> Optional[str]:
        """Find relevant information in uploaded documents."""
        try:
            # Create search patterns based on field name and label
            patterns = [
                re.compile(rf"(?i){re.escape(field_name)}:?\s*(.*?)(?:\.|$)"),
                re.compile(rf"(?i){re.escape(field_label)}:?\s*(.*?)(?:\.|$)")
            ]
            
            # Search through all documents
            for doc_content in document_content.values():
                for pattern in patterns:
                    match = pattern.search(doc_content)
                    if match:
                        return match.group(1).strip()
            
            return None
        except Exception:
            return None
    
    def add_document_content(self, filename: str, content: str):
        """Add document content to the session data."""
        self.session_data["document_content"][filename] = content
    
    def search_general_research(self, query: str) -> Dict[str, Any]:
        """Perform general web search and return results."""
        search_url = f"https://www.google.com/search?q={query}"
        self.driver.get(search_url)
        
        results = []
        try:
            search_results = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g"))
            )
            
            for result in search_results[:5]:  # Get top 5 results
                try:
                    title = result.find_element(By.CSS_SELECTOR, "h3").text
                    link = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    snippet = result.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet
                    })
                except Exception as e:
                    logger.warning(f"Error extracting search result: {str(e)}")
            
            self.save_screenshot("search", "search_results")
            return {
                "results": results,
                "screenshot": self.session_data["screenshots"][-1]
            }
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return {"error": str(e)}
    
    def browse_website_research(self, url: str) -> Dict[str, Any]:
        """Browse website and extract content."""
        if url in self.session_data["visited_urls"]:
            return {"message": "URL already visited", "url": url}
        
        try:
            self.driver.get(url)
            self.session_data["visited_urls"].add(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract text content
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Take screenshot
            screenshot = self.save_screenshot("browse", url.split('/')[-1])
            
            # Look for forms
            forms = soup.find_all('form')
            form_info = []
            for form in forms:
                form_info.append({
                    "action": form.get('action', ''),
                    "method": form.get('method', ''),
                    "fields": [{"name": input.get('name', ''), "type": input.get('type', '')} 
                             for input in form.find_all('input')]
                })
            
            return {
                "url": url,
                "text_content": text_content[:1000],  # First 1000 chars
                "forms_found": len(form_info),
                "forms": form_info,
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error browsing website: {str(e)}")
            return {"error": str(e)}
    
    def handle_form_research(self, url: str, fields: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Handle form fields on a webpage."""
        try:
            self.driver.get(url)
            
            # Find all form fields
            form_elements = self.driver.find_elements(By.TAG_NAME, "input")
            form_fields = []
            
            for element in form_elements:
                field_type = element.get_attribute("type")
                field_name = element.get_attribute("name")
                field_id = element.get_attribute("id")
                placeholder = element.get_attribute("placeholder")
                
                if field_type not in ["hidden", "submit"]:
                    field_info = {
                        "type": field_type,
                        "name": field_name,
                        "id": field_id,
                        "placeholder": placeholder,
                        "required": element.get_attribute("required") == "true"
                    }
                    form_fields.append(field_info)
            
            # Store form fields in session data
            self.session_data["form_fields"][url] = form_fields
            
            # Take screenshot of the form
            screenshot = self.save_screenshot("form", "form_fields")
            
            return {
                "url": url,
                "fields": form_fields,
                "screenshot": screenshot
            }
        except Exception as e:
            logger.error(f"Error handling form: {str(e)}")
            return {"error": str(e)}
    
    def recommend_fill_research(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for form fields."""
        try:
            recommendations = {}
            form_fields = context.get("form_fields", [])
            user_data = context.get("user_data", {})
            
            for field in form_fields:
                field_type = field.get("type", "")
                field_name = field.get("name", "").lower()
                
                # Generate recommendations based on field type and name
                if field_type == "email":
                    recommendations[field_name] = user_data.get("email", "")
                elif "name" in field_name:
                    if "first" in field_name:
                        recommendations[field_name] = user_data.get("first_name", "")
                    elif "last" in field_name:
                        recommendations[field_name] = user_data.get("last_name", "")
                elif "phone" in field_name:
                    recommendations[field_name] = user_data.get("phone", "")
                elif "address" in field_name:
                    recommendations[field_name] = user_data.get("address", "")
                
            return {
                "recommendations": recommendations,
                "missing_info": [field for field in form_fields 
                               if field.get("name", "").lower() not in recommendations]
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
