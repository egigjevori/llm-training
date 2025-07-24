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


@step
def has_next_page(page_number: int) -> bool:
    """Check if there's a next page by looking for the 'me pas >' button with a link."""
    try:
        url = f"{TENDER_LIST_URL}/{page_number}"
        logger.info(f"Checking for next page from: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the "me pas >" text and check if it has a link
        next_links = soup.find_all('a', string=re.compile(r'me pas', re.IGNORECASE))
        
        for link in next_links:
            if 'me pas' in link.get_text().lower():
                href = link.get('href', '')
                if href and '/faqe/' in href:
                    logger.info(f"Found next page link: {href}")
                    return True
        
        # Also check for pagination with ">"
        next_symbols = soup.find_all('a', string=re.compile(r'>', re.IGNORECASE))
        for link in next_symbols:
            if '>' in link.get_text():
                href = link.get('href', '')
                if href and '/faqe/' in href:
                    logger.info(f"Found next page symbol link: {href}")
                    return True
        
        logger.info(f"No next page found for page {page_number}")
        return False
        
    except Exception as e:
        logger.error(f"Error checking for next page from page {page_number}: {e}")
        return False


@step
def get_tender_links_from_page(page_number: int = 1) -> List[str]:
    """Extract tender links from a specific page of the tender list."""
    try:
        url = f"{TENDER_LIST_URL}/{page_number}"
        logger.info(f"Scraping tender links from page {page_number}: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tender links that match the pattern /sq/tender/view/id/{id}
        tender_links = []
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            if '/sq/tender/view/id/' in href:
                full_url = urljoin(BASE_URL, href)
                tender_links.append(full_url)
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(tender_links))
        
        logger.info(f"Found {len(unique_links)} unique tender links on page {page_number}")
        return unique_links
        
    except Exception as e:
        logger.error(f"Error scraping tender links from page {page_number}: {e}")
        return []


@step
def extract_tender_data(tender_url: str) -> Optional[Dict]:
    """Extract tender data from HTML table on tender page."""
    try:
        logger.info(f"Extracting data from tender: {tender_url}")
        
        response = requests.get(tender_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the results_table which contains the tender data
        results_table = soup.find('table', id='results_table')
        
        if not results_table:
            logger.warning(f"No results_table found in {tender_url}")
            return None
        
        tender_data = {}
        
        # Extract data from the table rows
        rows = results_table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                
                # Clean up the data
                key = key.replace('\n', ' ').replace('\r', ' ').strip()
                value = value.replace('\n', ' ').replace('\r', ' ').strip()
                
                if key and value:
                    tender_data[key] = value
        
        # Add metadata
        tender_data['tender_url'] = tender_url
        tender_data['scraped_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract tender ID from URL
        tender_id_match = re.search(r'/id/(\d+)', tender_url)
        if tender_id_match:
            tender_data['tender_id'] = tender_id_match.group(1)
        
        logger.info(f"Successfully extracted data for tender ID: {tender_data.get('tender_id', 'Unknown')}")
        
        # Print tender data immediately
        print("\n" + "="*60)
        print(f"TENDER ID: {tender_data.get('tender_id', 'Unknown')}")
        print("="*60)
        for key, value in tender_data.items():
            if key not in ['tender_url', 'scraped_at', 'tender_id']:
                print(f"{key}: {value}")
        print("-"*60)
        print(f"URL: {tender_data.get('tender_url', '')}")
        print(f"Scraped at: {tender_data.get('scraped_at', '')}")
        print("="*60)
        
        return tender_data
        
    except Exception as e:
        logger.error(f"Error extracting tender data from {tender_url}: {e}")
        return None


@step
def store_tender_in_mongodb(tender_data: Dict) -> bool:
    """
    Store tender data in MongoDB using tender_id as the unique identifier.
    
    Args:
        tender_data: Dictionary containing tender information
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        # Get MongoDB connection
        client = get_mongodb_connection()
        db = get_database(client, "openprocurement_albania")
        collection = db["tenders"]
        
        # Use tender_id as the unique identifier
        tender_id = tender_data.get('tender_id', 'Unknown')
        
        if tender_id == 'Unknown' or not tender_id:
            logger.warning(f"Tender with ID {tender_id} has no valid tender_id, skipping MongoDB storage")
            return False
        
        # Use upsert to update existing records or insert new ones
        result = collection.update_one(
            {"tender_id": tender_id},  # Filter by tender_id
            {"$set": tender_data},  # Update with new data
            upsert=True  # Insert if not exists
        )
        
        if result.upserted_id:
            logger.info(f"Inserted new tender with ID: {tender_id}")
        elif result.modified_count > 0:
            logger.info(f"Updated existing tender with ID: {tender_id}")
        else:
            logger.info(f"No changes needed for tender with ID: {tender_id}")
        
        # Close connection
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing tender with ID {tender_data.get('tender_id', 'Unknown')} in MongoDB: {e}")
        return False





@step
def process_tenders_from_page(page_number: int = 1, max_tenders_per_page: int = 5) -> List[Dict]:
    """Process tenders from a specific page by extracting data from HTML tables."""
    tender_links = get_tender_links_from_page(page_number)
    
    processed_tenders = []
    processed_count = 0
    
    for tender_url in tender_links:
        if processed_count >= max_tenders_per_page:
            break
            
        tender_data = extract_tender_data(tender_url)
        if tender_data:
            # Store tender data in MongoDB
            stored = store_tender_in_mongodb(tender_data)
            
            processed_tenders.append(tender_data)
            processed_count += 1
            
            # Print storage status
            storage_status = "✅ Stored" if stored else "❌ Failed"
            print(f"MongoDB: {storage_status}")
                
            # Add a small delay to be respectful to the server
            time.sleep(1)
    
    logger.info(f"Processed {len(processed_tenders)} tenders from page {page_number}")
    return processed_tenders


@step
def process_all_pages(max_tenders_per_page: int = 3) -> List[Dict]:
    """Process tenders from all available pages until no next page is found."""
    all_tender_data = []
    current_page = 1
    
    while True:
        logger.info(f"Processing page {current_page}")
        try:
            # Process current page
            page_tenders = process_tenders_from_page(current_page, max_tenders_per_page)
            all_tender_data.extend(page_tenders)
            
            # Check if there's a next page
            if not has_next_page(current_page):
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
                logger.error("Reached safety limit of 100 pages, stopping")
                break
    
    logger.info(f"Completed processing all pages. Total tenders processed: {len(all_tender_data)}")
    return all_tender_data


@pipeline
def open_procurement_scraping_pipeline():
    """Main pipeline for scraping Open Procurement Albania website."""
    # Process all pages until no next page is found (limiting to 3 tenders per page for efficiency)
    all_tender_data = process_all_pages(max_tenders_per_page=3)


if __name__ == "__main__":
    open_procurement_scraping_pipeline()
