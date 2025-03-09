import json
import os
import re
import time
import urllib
from urllib.error import HTTPError, URLError
import wikipedia
import arxiv
import PyPDF2
import requests
import config
import logging
from datetime import datetime

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_PATH,
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger('retrieval')

def load_cache(cache_path):
    """Load cached data from file if it exists"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cache from {cache_path}: {str(e)}")
    return {}

def save_cache(cache_path, data):
    """Save data to cache file"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving cache to {cache_path}: {str(e)}")

def fetch_arxiv(query, max_results=10, max_retries=3):
    """Fetch papers from ArXiv with improved search and retry logic"""
    papers = []
    
    # Try to load from cache first
    cache = load_cache(config.ARXIV_CACHE_PATH)
    cache_key = f"{query}_{max_results}"
    
    if cache_key in cache and time.time() - cache[cache_key].get('timestamp', 0) < 86400:  # 24 hour cache
        logger.info(f"Using cached ArXiv results for '{query}'")
        return cache[cache_key]['data']
    
    logger.info(f"Fetching ArXiv papers for query: '{query}'")
    
    for attempt in range(max_retries):
        try:
            # Clean query for ArXiv search
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            
            # Format query for better results
            if ' ' in clean_query and len(clean_query.split()) > 1:
                search_query = f'"{clean_query}"'
            else:
                search_query = clean_query
                
            # Add filters to improve result quality
            search_query = f"ti:{search_query} OR abs:{search_query}"
            
            # Get client with proper identification
            client = arxiv.Client(
                page_size = max_results,
                delay_seconds = 3.0,
                num_retries = 3,
                http_headers = {"User-Agent": f"ScholarSage/1.0 (mailto:{config.ARXIV_EMAIL or 'example@example.com'})"}
            )
            
            # Run search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Get results
            results = list(client.results(search))
            
            for paper in results:
                papers.append({
                    'title': paper.title,
                    'summary': paper.summary,
                    'url': paper.pdf_url,
                    'published': paper.published.strftime('%Y-%m-%d') if hasattr(paper, 'published') else 'Unknown',
                    'authors': ', '.join([author.name for author in paper.authors]) if hasattr(paper, 'authors') else 'Unknown'
                })
            
            # Cache the results
            cache[cache_key] = {
                'timestamp': time.time(),
                'data': papers
            }
            save_cache(config.ARXIV_CACHE_PATH, cache)
            
            logger.info(f"Found {len(papers)} papers from ArXiv for '{query}'")
            break  # Success, exit the retry loop
            
        except Exception as e:
            logger.error(f"ArXiv fetch error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch ArXiv data for '{query}' after {max_retries} attempts.")
            time.sleep(2)
    
    return papers

def fetch_wikipedia(query, max_retries=3):
    """Fetch data from Wikipedia with multiple search attempts"""
    # Try to load from cache first
    cache = load_cache(config.WIKI_CACHE_PATH)
    if query in cache and time.time() - cache[query].get('timestamp', 0) < 604800:  # 7 day cache
        logger.info(f"Using cached Wikipedia results for '{query}'")
        return cache[query]['content']
    
    logger.info(f"Fetching Wikipedia content for: '{query}'")
    wikipedia.set_lang("en")
    
    for attempt in range(max_retries):
        try:
            # Try direct page access first
            try:
                page = wikipedia.page(query, auto_suggest=True)
                content = page.content
                
                # Cache the result
                cache[query] = {
                    'timestamp': time.time(),
                    'content': content
                }
                save_cache(config.WIKI_CACHE_PATH, cache)
                
                logger.info(f"Successfully retrieved Wikipedia page for '{query}'")
                return content
                
            except wikipedia.DisambiguationError as e:
                # If disambiguation, try the first option
                try:
                    logger.info(f"Disambiguation page for '{query}'. Trying first option: {e.options[0]}")
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    content = page.content
                    
                    # Cache the result
                    cache[query] = {
                        'timestamp': time.time(),
                        'content': content
                    }
                    save_cache(config.WIKI_CACHE_PATH, cache)
                    
                    return content
                except Exception as inner_e:
                    logger.warning(f"Failed to get first disambiguation option: {str(inner_e)}")
                    pass
                    
            except wikipedia.PageError:
                # If page not found, try search
                search_results = wikipedia.search(query)
                if search_results:
                    try:
                        logger.info(f"No exact page for '{query}'. Trying search result: {search_results[0]}")
                        page = wikipedia.page(search_results[0], auto_suggest=False)
                        content = page.content
                        
                        # Cache the result
                        cache[query] = {
                            'timestamp': time.time(),
                            'content': content
                        }
                        save_cache(config.WIKI_CACHE_PATH, cache)
                        
                        return content
                    except Exception as inner_e:
                        logger.warning(f"Failed to get search result: {str(inner_e)}")
                        pass
            
            # If direct approaches fail, try more aggressive search techniques
            if attempt > 0:
                # Try extracting key terms and search for those
                terms = query.split()
                if len(terms) > 2:
                    key_terms = [term for term in terms if len(term) > 3 and term.lower() not in ['what', 'when', 'where', 'why', 'who', 'how', 'the', 'and', 'that']]
                    if key_terms:
                        search_query = " ".join(key_terms)
                        logger.info(f"Trying key terms search for '{query}': '{search_query}'")
                        search_results = wikipedia.search(search_query)
                        if search_results:
                            try:
                                page = wikipedia.page(search_results[0], auto_suggest=False)
                                content = page.content
                                
                                # Cache the result
                                cache[query] = {
                                    'timestamp': time.time(),
                                    'content': content
                                }
                                save_cache(config.WIKI_CACHE_PATH, cache)
                                
                                return content
                            except Exception as inner_e:
                                logger.warning(f"Failed to get key terms search result: {str(inner_e)}")
                                pass
            
        except Exception as e:
            logger.error(f"Wikipedia fetch error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch Wikipedia data for '{query}' after {max_retries} attempts.")
            time.sleep(1)
    
    logger.error(f"Could not find any Wikipedia information for '{query}'")
    return ""

def read_pdf(file_path):
    """Extract text from a PDF file"""
    text = ""
    logger.info(f"Reading PDF from: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            try:
                # Try with PyPDF2 first
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {str(e)}")
            except Exception as e:
                logger.error(f"PyPDF2 failed to read the PDF: {str(e)}")
                # If PyPDF2 fails, try with PyMuPDF (fitz) if available
                try:
                    import fitz
                    file.seek(0)  # Reset file pointer
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    for page_num in range(len(doc)):
                        try:
                            page = doc[page_num]
                            text += page.get_text() + "\n"
                        except Exception as e:
                            logger.warning(f"PyMuPDF: Error extracting page {page_num}: {str(e)}")
                except ImportError:
                    logger.error("PyMuPDF not available as fallback")
                
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return ""
