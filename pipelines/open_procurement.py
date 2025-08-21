"""
Open Procurement Albania Web Scraping Pipeline using ZenML
"""

import logging
import time
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin
import json

import requests
from bs4 import BeautifulSoup

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from mongo import get_mongodb_connection, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://openprocurement.al"
TENDER_LIST_URL = f"{BASE_URL}/sq/tender/list/faqe"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def check_next_page_link(soup) -> bool:
    """Check for next page links in the soup object."""
    # Look for the "me pas >" text and check if it has a link
    next_links = soup.find_all('a', string=re.compile(r'me pas', re.IGNORECASE))
    for link in next_links:
        if 'me pas' in link.get_text().lower():
            href = link.get('href', '')
            if href and '/faqe/' in href:
                logger.debug(f"Found next page link: {href}")
                return True
    
    # Also check for pagination with ">"
    next_symbols = soup.find_all('a', string=re.compile(r'>', re.IGNORECASE))
    for link in next_symbols:
        if '>' in link.get_text():
            href = link.get('href', '')
            if href and '/faqe/' in href:
                logger.debug(f"Found next page symbol link: {href}")
                return True
    
    return False


def extract_links(soup) -> List[str]:
    """Extract tender links from the soup object."""
    links = []
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link.get('href', '')
        if '/sq/tender/view/id/' in href:
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(links))


@step
def get_page_data(page_number: int = 1) -> tuple[List[str], bool]:
    """Get tender links and check for next page in a single request."""
    try:
        url = f"{TENDER_LIST_URL}/{page_number}"
        logger.info(f"Scraping page {page_number}: {url}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = extract_links(soup)
        has_next = check_next_page_link(soup)
        
        logger.info(f"Successfully scraped {len(links)} unique tender links from page {page_number}")
        logger.debug(f"Next page available: {has_next}")
        
        return links, has_next
    except Exception as e:
        logger.error(f"Error scraping page {page_number}: {e}")
        return [], False


@step
def has_next_page(page_number: int) -> bool:
    """Check if there's a next page by looking for the 'me pas >' button with a link."""
    # This function is now deprecated but kept for backward compatibility
    # The next page check is now done in get_page_data()
    logger.warning("has_next_page() is deprecated. Use get_page_data() instead.")
    return False


def process_table_data(results_table) -> Dict:
    """Process the results table and extract tender data."""
    data = {}
    rows = results_table.find_all('tr')
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            key, value = cells[0].get_text(), cells[1].get_text()
            # Clean up the data - preserve newlines but normalize whitespace
            key = re.sub(r'[ \t]+', ' ', key).strip()
            value = re.sub(r'[ \t]+', ' ', value).strip()
            
            if key and value:
                data[key] = value
    
    return data


def log_data(data: Dict) -> None:
    """Log tender data in a formatted way."""
    tender_id = data.get('tender_id', 'Unknown')
    logger.debug("="*60)
    logger.debug(f"TENDER ID: {tender_id}")
    logger.debug("="*60)
    
    for key, value in data.items():
        if key not in ['tender_url', 'scraped_at', 'tender_id']:
            logger.debug(f"{key}: {value}")
    
    logger.debug("-"*60)
    logger.debug(f"URL: {data.get('tender_url', '')}")
    logger.debug(f"Scraped at: {data.get('scraped_at', '')}")
    logger.debug("="*60)


@step
def extract_tender_data(tender_url: str) -> Optional[Dict]:
    """Extract tender data from HTML table on tender page."""
    try:
        logger.info(f"Scraping details for tender: {tender_url}")
        response = requests.get(tender_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the results_table which contains the tender data
        results_table = soup.find('table', id='results_table')
        if not results_table:
            logger.warning(f"No results_table found in {tender_url}")
            return None
        
        data = process_table_data(results_table)
        
        # Add metadata
        data['tender_url'] = tender_url
        data['scraped_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract tender ID from URL
        tender_id_match = re.search(r'/id/(\d+)', tender_url)
        if tender_id_match:
            data['tender_id'] = tender_id_match.group(1)
        
        logger.info(f"Successfully scraped details for tender ID: {data.get('tender_id', 'Unknown')}")
        
        # Log tender data
        log_data(data)
        return data
    except Exception as e:
        logger.error(f"Error scraping details for tender {tender_url}: {e}")
        return {
            "tender_url": tender_url,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tender_id": "Unknown"
        }


@step
def store_tender_in_mongodb(tender_data: Dict) -> bool:
    """Store tender data in MongoDB using tender_id as the unique identifier."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, "openprocurement_albania")
        collection = db["tenders"]
        tender_id = tender_data.get('tender_id', 'Unknown')
        
        if tender_id == 'Unknown' or not tender_id:
            logger.warning(f"Tender with ID {tender_id} has no valid tender_id, skipping MongoDB storage")
            return False
        
        result = collection.update_one({"tender_id": tender_id}, {"$set": tender_data}, upsert=True)
        
        if result.upserted_id:
            logger.info(f"Inserted new tender with ID: {tender_id}")
        elif result.modified_count > 0:
            logger.info(f"Updated existing tender with ID: {tender_id}")
        else:
            logger.info(f"No changes needed for tender with ID: {tender_id}")
        
        client.close()
        return True
    except Exception as e:
        logger.error(f"Error storing tender with ID {tender_data.get('tender_id', 'Unknown')} in MongoDB: {e}")
        return False


def process_single_tender(tender_url: str) -> Optional[Dict]:
    """Process a single tender URL and return the data."""
    data = extract_tender_data(tender_url)
    if data:
        stored = store_tender_in_mongodb(data)
        storage_status = "✅ Stored" if stored else "❌ Failed"
        logger.debug(f"MongoDB: {storage_status}")
        return data
    return None


@step
def process_page(page_number: int = 1) -> tuple[List[Dict], bool]:
    """Process tenders from a specific page and return data with next page status."""
    try:
        links, has_next = get_page_data(page_number)
        processed = []
        
        for url in links:
            data = process_single_tender(url)
            if data:
                processed.append(data)
        
        logger.info(f"Successfully processed {len(processed)} tenders from page {page_number}")
        return processed, has_next
    except Exception as e:
        logger.error(f"Error processing page {page_number}: {e}")
        return [], False


@step
def process_all_pages() -> List[Dict]:
    """Process tenders from all available pages until no next page is found."""
    all_data = []
    current_page = 1
    
    while True:
        logger.info(f"Processing page {current_page}")
        try:
            # Process current page and get next page status
            page_data, has_next = process_page(current_page)
            all_data.extend(page_data)
            
            # Check if there's a next page
            if not has_next:
                logger.info(f"No more pages found after page {current_page}")
                break
            
            # Move to next page
            current_page += 1
            # Add a delay between pages
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error processing page {current_page}: {e}")
            # Try next page even if current page failed
            current_page += 1
            # But break if we've tried too many consecutive failures
            if current_page > 34:  # Safety limit
                logger.error("Reached safety limit of 34 pages, stopping")
                break
    
    logger.info(f"Completed processing all pages. Total tenders processed: {len(all_data)}")
    return all_data


@pipeline
def open_procurement_scraping_pipeline():
    """Main pipeline for scraping Open Procurement Albania website."""
    # Process all pages until no next page is found
    process_all_pages()


if __name__ == "__main__":
    open_procurement_scraping_pipeline()
