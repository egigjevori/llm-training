"""
OpenCorporates Albania Web Scraping Pipeline using ZenML
"""

import logging
import time
from typing import Dict, List
from urllib.parse import urljoin
import re

import requests
from bs4 import BeautifulSoup

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from mongo import get_mongodb_connection, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://opencorporates.al"
SEARCH_URL = f"{BASE_URL}/sq/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


@step(enable_cache=False)
def get_total_pages() -> int:
    """Get the total number of pages to scrape from the search results."""
    response = requests.get(f"{SEARCH_URL}?page=1", headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    pagination = soup.find('ul', class_='pagination')
    last_link = pagination.find('a', string='Last')
    href = last_link.get('href', '')
    page_match = re.search(r'page=(\d+)', href)
    last_page = int(page_match.group(1))
    logger.info(f"Found {last_page} pages from last button")
    return last_page


def process_company_card(card) -> Dict:
    """Process a single company card and extract basic information."""
    try:
        name_elem = card.find('h4', class_='mb-0')
        name = name_elem.get_text(strip=True) if name_elem else "Unknown"
        
        company_id_elem = card.find('a', class_='font-weight-bold text-muted')
        company_id = company_id_elem.get_text(strip=True) if company_id_elem else "Unknown"
        
        desc_elem = card.find('p', class_='text-muted mb-2 text-sm')
        description = ""
        if desc_elem:
            company_id_link = desc_elem.find('a')
            if company_id_link:
                company_id_link.extract()
            description = desc_elem.get_text(strip=True)
        
        location = "Unknown"
        currency = "Unknown"
        info_elem = card.find_all('p', class_='text-muted mb-2 text-sm')[1] if len(card.find_all('p', class_='text-muted mb-2 text-sm')) > 1 else None
        if info_elem:
            location_match = re.search(r'<i class="fa fa-map-marker"></i>\s*([^<]+)', str(info_elem))
            if location_match:
                location = location_match.group(1).strip()
            
            currency_match = re.search(r'<i class="fa fa-money[^"]*"></i>\s*([^<]+)', str(info_elem))
            if currency_match:
                currency = currency_match.group(1).strip()
        
        detail_url = ""
        detail_link = card.find('a', class_='btn btn-danger text-uppercase font-weight-bold d-lg-block')
        if detail_link:
            detail_url = urljoin(BASE_URL, detail_link.get('href', ''))
        
        company = {
            "name": name,
            "company_id": company_id,
            "description": description,
            "location": location,
            "currency": currency,
            "detail_url": detail_url,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.debug(f"Extracted company: {name} ({company_id})")
        return company
        
    except Exception as e:
        logger.error(f"Error extracting company from card: {e}")
        return None


def process_page_content(soup) -> List[Dict]:
    """Process the content of a page and extract all company listings."""
    companies = []
    company_cards = soup.find_all('div', class_='card px-3 py-4 mb-3 row-hover pos-relative')
    
    for card in company_cards:
        company = process_company_card(card)
        if company:
            companies.append(company)
    
    return companies


@step(enable_cache=False)
def scrape_company_listings(page: int) -> List[Dict]:
    """Scrape company listings from a specific page."""
    companies = []
    
    try:
        url = f"{SEARCH_URL}?page={page}"
        logger.info(f"Scraping page {page}: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        companies = process_page_content(soup)
        
        logger.info(f"Successfully scraped {len(companies)} companies from page {page}")
        
    except Exception as e:
        logger.error(f"Error scraping page {page}: {e}")
    
    return companies


def process_table_data(table) -> Dict:
    """Process table data and extract structured information."""
    table_data = {}
    rows = table.find_all('tr')
    
    for row in rows:
        th = row.find('th')
        td = row.find('td')
        if th and td:
            header = th.get_text(strip=True)
            value = td.get_text(strip=True)
            
            if 'Viti i Themelimit' in header or 'Data e regjistrimit' in header:
                table_data["registration_date"] = value
            elif 'Statusi' in header or 'Status' in header:
                table_data["status"] = value
            elif 'Forma ligjore' in header or 'Legal form' in header:
                table_data["legal_form"] = value
            elif 'Kapitali Themeltar' in header or 'Capital' in header:
                table_data["capital"] = value
            elif 'Adresa' in header or 'Address' in header:
                table_data["address"] = value
            elif 'Telefoni' in header or 'Phone' in header:
                table_data["phone"] = value
            elif 'Email' in header or 'E-mail' in header:
                table_data["email"] = value
            elif 'Website' in header or 'Web' in header:
                table_data["website"] = value
            elif 'Administrator' in header or 'Drejtuesit' in header:
                admin_links = td.find_all('a')
                if admin_links:
                    table_data["administrators"] = [link.get_text(strip=True) for link in admin_links]
                else:
                    table_data["administrators"] = [value] if value and value != '-' else []
            elif 'Objekti i Veprimtarisë' in header:
                table_data["business_object"] = value
                table_data["description"] = value
            elif 'Emërtime të tjera Tregtare' in header:
                table_data["other_trading_names"] = value
            elif 'Rrethi' in header:
                table_data["region"] = value
            elif 'Leje/Licensa' in header:
                table_data["license"] = value
            elif 'Ndryshime' in header:
                # Add appropriate spaces in ndryshime text
                cleaned_value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
                cleaned_value = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_value)
                cleaned_value = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned_value)
                table_data["changes"] = cleaned_value
            elif 'Akte. Tjetërsim Kapitali' in header:
                # Extract document links
                links = td.find_all('a')
                if links:
                    document_links = []
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        if href and text:
                            full_url = urljoin(BASE_URL, href)
                            document_links.append({
                                'text': text,
                                'url': full_url
                            })
                    table_data["capital_amendment_documents"] = document_links
                else:
                    table_data["capital_amendment_documents"] = value
            elif 'Dokumenta Financiare' in header:
                # Extract document links
                links = td.find_all('a')
                if links:
                    document_links = []
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        if href and text:
                            full_url = urljoin(BASE_URL, href)
                            document_links.append({
                                'text': text,
                                'url': full_url
                            })
                    table_data["financial_documents"] = document_links
                else:
                    table_data["financial_documents"] = value
    
    return table_data


def process_shareholders(soup) -> List[str]:
    """Process shareholder information from the page."""
    shareholders = []
    all_list_groups = soup.find_all('ul', class_='list-group-striped')
    
    for list_group in all_list_groups:
        shareholder_items = list_group.find_all('li', class_='list-group-item')
        if shareholder_items:
            for item in shareholder_items:
                link = item.find('a')
                if link:
                    shareholder_text = link.get_text(strip=True)
                    clean_text = ' '.join(shareholder_text.split())
                    shareholders.append(clean_text)
            break
    
    return shareholders


def process_financial_section(soup) -> Dict:
    """Process financial information section."""
    financial_data = {}
    financial_section = soup.find('h4', string=re.compile(r'Informacione Financiare', re.I))
    
    if financial_section:
        financial_text = financial_section.get_text(strip=True)
        if financial_text:
            financial_data["financial_information_text"] = financial_text
        
        profit_info = financial_section.find_next('table')
        if profit_info:
            profit_rows = profit_info.find_all('tr')
            financial_data["financial_information"] = {}
            for row in profit_rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    year = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if year and value:
                        # Add appropriate spaces in financial data
                        cleaned_year = re.sub(r'([a-z])([A-Z])', r'\1 \2', year)
                        cleaned_value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
                        financial_data["financial_information"][cleaned_year] = cleaned_value
        
        next_element = financial_section.find_next_sibling()
        if next_element:
            financial_detail_text = next_element.get_text(strip=True)
            if financial_detail_text and len(financial_detail_text) > 50:
                # Add appropriate spaces in financial detail text
                cleaned_financial_data = re.sub(r'([a-z])([A-Z])', r'\1 \2', financial_detail_text)
                cleaned_financial_data = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_financial_data)
                cleaned_financial_data = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned_financial_data)
                financial_data["financial_information_details"] = cleaned_financial_data[:1000]
    
    return financial_data


def process_additional_sections(soup) -> Dict:
    """Process additional sections from the page."""
    additional_sections_data = {}
    additional_sections = soup.find_all(['h3', 'h4', 'h5'])
    
    for section in additional_sections:
        section_title = section.get_text(strip=True)
        if section_title and len(section_title) > 3:
            next_element = section.find_next_sibling()
            if next_element:
                content = next_element.get_text(strip=True)
                if content and len(content) > 10:
                    # Add appropriate spaces in additional section content
                    cleaned_content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
                    cleaned_content = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_content)
                    cleaned_content = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned_content)
                    additional_sections_data[section_title] = cleaned_content[:500]
    
    return additional_sections_data


@step(enable_cache=False)
def process_company_link(company_summary: Dict) -> Dict:
    """Process a single company link and extract detailed information."""
    try:
        logger.info(f"Scraping details for: {company_summary['name']} ({company_summary['company_id']})")
        
        response = requests.get(company_summary['detail_url'], headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        detail = {
            "name": company_summary['name'],
            "company_id": company_summary['company_id'],
            "description": company_summary['description'],
            "location": company_summary['location'],
            "currency": company_summary['currency'],
            "detail_url": company_summary['detail_url'],
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Process table data
        table = soup.find('table', class_='rwd-table')
        if table:
            table_data = process_table_data(table)
            detail.update(table_data)
        
        # Process shareholders
        if "shareholders" not in detail:
            shareholders = process_shareholders(soup)
            if shareholders:
                detail["shareholders"] = shareholders
        
        # Process financial section
        financial_data = process_financial_section(soup)
        detail.update(financial_data)
        
        # Process additional sections
        additional_sections_data = process_additional_sections(soup)
        detail["additional_sections"] = additional_sections_data
        
        logger.info(f"Successfully scraped details for {detail['name']}")
        return detail
        
    except Exception as e:
        logger.error(f"Error scraping details for {company_summary['name']}: {e}")
        return {
            "name": company_summary['name'],
            "company_id": company_summary['company_id'],
            "description": company_summary['description'],
            "location": company_summary['location'],
            "currency": company_summary['currency'],
            "detail_url": company_summary['detail_url'],
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }


@step(enable_cache=False)
def store_company_in_mongodb(company_detail: Dict) -> bool:
    """Store company details in MongoDB using company_id as the unique identifier."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, "opencorporates_albania")
        collection = db["companies"]
        company_id = company_detail.get('company_id', 'Unknown')
        
        if company_id == 'Unknown' or not company_id:
            logger.warning(f"Company with company_id {company_id} has no valid company_id, skipping MongoDB storage")
            return False
        
        result = collection.update_one({"company_id": company_id}, {"$set": company_detail}, upsert=True)
        
        if result.upserted_id:
            logger.info(f"Inserted new company with company_id: {company_id}")
        elif result.modified_count > 0:
            logger.info(f"Updated existing company with company_id: {company_id}")
        else:
            logger.info(f"No changes needed for company with company_id: {company_id}")
        
        client.close()
        return True
    except Exception as e:
        logger.error(f"Error storing company with company_id {company_detail.get('company_id', 'Unknown')} in MongoDB: {e}")
        return False


def process_page(page: int) -> List[Dict]:
    """Process a single page and return company details."""
    companies = scrape_company_listings(page)
    company_details = []
    
    for company in companies:
        try:
            detail = process_company_link(company)
            company_details.append(detail)
        except Exception as e:
            logger.error(f"Error processing company {company.get('name', 'Unknown')}: {e}")
            continue
    
    return company_details


@step(enable_cache=False)
def process_multiple_pages(total_pages: int) -> None:
    """Process multiple pages of company listings and their details."""
    for page in range(1, total_pages):
        try:
            company_details = process_page(page)
            
            # Store all companies from this page
            for detail in company_details:
                stored = store_company_in_mongodb(detail)
                
                name = detail.get('name', 'Unknown')
                company_id = detail.get('company_id', 'Unknown')
                location = detail.get('location', 'Unknown')
                status = detail.get('status', 'Unknown')
                registration_date = detail.get('registration_date', 'Unknown')
                storage_status = "✅ Stored" if stored else "❌ Failed"
                
                logger.info(f"Company: {name} | Company ID: {company_id} | Location: {location} | Status: {status} | Registration: {registration_date} | MongoDB: {storage_status}")
                
        except Exception as e:
            logger.error(f"Error processing page {page}: {e}")
            continue


@pipeline
def opencorporates_scraping_pipeline():
    """Main pipeline for scraping OpenCorporates Albania website."""
    total_pages = get_total_pages()
    process_multiple_pages(total_pages)


if __name__ == "__main__":
    opencorporates_scraping_pipeline()
