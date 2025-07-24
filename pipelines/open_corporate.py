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


@step
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



@step
def scrape_company_listings(page: int) -> List[Dict]:
    """Scrape company listings from a specific page."""
    companies = []
    
    try:
        url = f"{SEARCH_URL}?page={page}"
        logger.info(f"Scraping page {page}: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        company_cards = soup.find_all('div', class_='card px-3 py-4 mb-3 row-hover pos-relative')
        
        for card in company_cards:
            try:
                name_elem = card.find('h4', class_='mb-0')
                name = name_elem.get_text(strip=True) if name_elem else "Unknown"
                
                nipt_elem = card.find('a', class_='font-weight-bold text-muted')
                nipt = nipt_elem.get_text(strip=True) if nipt_elem else "Unknown"
                
                desc_elem = card.find('p', class_='text-muted mb-2 text-sm')
                description = ""
                if desc_elem:
                    nipt_link = desc_elem.find('a')
                    if nipt_link:
                        nipt_link.extract()
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
                    "emri": name,
                    "nipt": nipt,
                    "përshkrimi": description,
                    "vendndodhja": location,
                    "monedha": currency,
                    "url_detaje": detail_url,
                    "data_mbledhjes": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                companies.append(company)
                logger.debug(f"Extracted company: {name} ({nipt})")
                
            except Exception as e:
                logger.error(f"Error extracting company from card: {e}")
                continue
        
        logger.info(f"Successfully scraped {len(companies)} companies from page {page}")
        
    except Exception as e:
        logger.error(f"Error scraping page {page}: {e}")
    
    return companies


@step
def scrape_company_details(company_summary: Dict) -> Dict:
    """Scrape detailed information for a specific company."""
    try:
        logger.info(f"Scraping details for: {company_summary['emri']} ({company_summary['nipt']})")
        
        response = requests.get(company_summary['url_detaje'], headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        detail = {
            "emri": company_summary['emri'],
            "nipt": company_summary['nipt'],
            "përshkrimi": company_summary['përshkrimi'],
            "vendndodhja": company_summary['vendndodhja'],
            "monedha": company_summary['monedha'],
            "url_detaje": company_summary['url_detaje'],
            "data_mbledhjes": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        main_content = soup.find('div', class_='main-content')
        if main_content:
            content_text = main_content.get_text(separator='\n', strip=True)
            if content_text:
                detail["përmbajtja_e_plotë"] = content_text[:2000]
        
        table = soup.find('table', class_='rwd-table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    header = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    
                    if 'Viti i Themelimit' in header or 'Data e regjistrimit' in header:
                        detail["data_regjistrimit"] = value
                    elif 'Statusi' in header or 'Status' in header:
                        detail["statusi"] = value
                    elif 'Forma ligjore' in header or 'Legal form' in header:
                        detail["forma_ligjore"] = value
                    elif 'Kapitali Themeltar' in header or 'Capital' in header:
                        detail["kapitali"] = value
                    elif 'Adresa' in header or 'Address' in header:
                        detail["adresa"] = value
                    elif 'Telefoni' in header or 'Phone' in header:
                        detail["telefoni"] = value
                    elif 'Email' in header or 'E-mail' in header:
                        detail["email"] = value
                    elif 'Website' in header or 'Web' in header:
                        detail["website"] = value
                    elif 'Administrator' in header or 'Drejtuesit' in header:
                        admin_links = td.find_all('a')
                        if admin_links:
                            detail["administratorët"] = [link.get_text(strip=True) for link in admin_links]
                        else:
                            detail["administratorët"] = [value] if value and value != '-' else []
                    elif 'Objekti i Veprimtarisë' in header:
                        detail["objekti_veprimtarisë"] = value
                        detail["përshkrimi"] = value
                    elif 'Emërtime të tjera Tregtare' in header:
                        detail["emërtime_tjera_tregtare"] = value
                    elif 'Rrethi' in header:
                        detail["rrethi"] = value
                    elif 'Leje/Licensa' in header:
                        detail["leje_licensa"] = value
                    elif 'Ndryshime' in header:
                        # Add appropriate spaces in ndryshime text
                        cleaned_value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
                        cleaned_value = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_value)
                        cleaned_value = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned_value)
                        detail["ndryshime"] = cleaned_value
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
                            detail["akte_tjetërsim_kapitali"] = document_links
                        else:
                            detail["akte_tjetërsim_kapitali"] = value
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
                            detail["dokumenta_financiare"] = document_links
                        else:
                            detail["dokumenta_financiare"] = value

        if "zotëruesit" not in detail:
            all_list_groups = soup.find_all('ul', class_='list-group-striped')
            for list_group in all_list_groups:
                shareholder_items = list_group.find_all('li', class_='list-group-item')
                if shareholder_items:
                    detail["zotëruesit"] = []
                    for item in shareholder_items:
                        link = item.find('a')
                        if link:
                            shareholder_text = link.get_text(strip=True)
                            clean_text = ' '.join(shareholder_text.split())
                            detail["zotëruesit"].append(clean_text)
                    break

        financial_section = soup.find('h4', string=re.compile(r'Informacione Financiare', re.I))
        if financial_section:
            financial_text = financial_section.get_text(strip=True)
            if financial_text:
                detail["informacione_financiare_text"] = financial_text
            
            profit_info = financial_section.find_next('table')
            if profit_info:
                profit_rows = profit_info.find_all('tr')
                detail["informacione_financiare"] = {}
                for row in profit_rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        year = cells[0].get_text(strip=True)
                        value = cells[1].get_text(strip=True)
                        if year and value:
                            # Add appropriate spaces in financial data
                            cleaned_year = re.sub(r'([a-z])([A-Z])', r'\1 \2', year)
                            cleaned_value = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
                            detail["informacione_financiare"][cleaned_year] = cleaned_value
            
            next_element = financial_section.find_next_sibling()
            if next_element:
                financial_data = next_element.get_text(strip=True)
                if financial_data and len(financial_data) > 50:
                    # Add appropriate spaces in financial detail text
                    cleaned_financial_data = re.sub(r'([a-z])([A-Z])', r'\1 \2', financial_data)
                    cleaned_financial_data = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_financial_data)
                    cleaned_financial_data = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned_financial_data)
                    detail["informacione_financiare_detaje"] = cleaned_financial_data[:1000]
        
        additional_sections = soup.find_all(['h3', 'h4', 'h5'])
        detail["seksione_shtesë"] = {}
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
                        detail["seksione_shtesë"][section_title] = cleaned_content[:500]
        
        logger.info(f"Successfully scraped details for {detail['emri']}")
        return detail
        
    except Exception as e:
        logger.error(f"Error scraping details for {company_summary['emri']}: {e}")
        return {
            "emri": company_summary['emri'],
            "nipt": company_summary['nipt'],
            "përshkrimi": company_summary['përshkrimi'],
            "vendndodhja": company_summary['vendndodhja'],
            "monedha": company_summary['monedha'],
            "url_detaje": company_summary['url_detaje'],
            "data_mbledhjes": time.strftime("%Y-%m-%d %H:%M:%S")
        }


@step
def store_company_in_mongodb(company_detail: Dict) -> bool:
    """
    Store company details in MongoDB using NIPT as the unique identifier.
    
    Args:
        company_detail: Dictionary containing company information
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        # Get MongoDB connection
        client = get_mongodb_connection()
        db = get_database(client, "opencorporates_albania")
        collection = db["companies"]
        
        # Use NIPT as the unique identifier
        nipt = company_detail.get('nipt', 'Unknown')
        
        if nipt == 'Unknown' or not nipt:
            logger.warning(f"Company with NIPT {nipt} has no valid NIPT, skipping MongoDB storage")
            return False
        
        # Use upsert to update existing records or insert new ones
        result = collection.update_one(
            {"nipt": nipt},  # Filter by NIPT
            {"$set": company_detail},  # Update with new data
            upsert=True  # Insert if not exists
        )
        
        if result.upserted_id:
            logger.info(f"Inserted new company with NIPT: {nipt}")
        elif result.modified_count > 0:
            logger.info(f"Updated existing company with NIPT: {nipt}")
        else:
            logger.info(f"No changes needed for company with NIPT: {nipt}")
        
        # Close connection
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error storing company with NIPT {company_detail.get('nipt', 'Unknown')} in MongoDB: {e}")
        return False


@step
def process_multiple_pages(total_pages: int) -> None:
    """Process multiple pages of company listings and their details."""
    for page in range(1, min(6, total_pages + 1)):
        companies = scrape_company_listings(page)
        
        # Process all companies from this page
        for company in companies:
            try:
                detail = scrape_company_details(company)
                
                # Store company details in MongoDB
                stored = store_company_in_mongodb(detail)
                
                name = detail.get('emri', 'Unknown')
                nipt = detail.get('nipt', 'Unknown')
                location = detail.get('vendndodhja', 'Unknown')
                status = detail.get('statusi', 'Unknown')
                registration_date = detail.get('data_regjistrimit', 'Unknown')
                storage_status = "✅ Stored" if stored else "❌ Failed"
                
                print(f"Company: {name} | NIPT: {nipt} | Location: {location} | Status: {status} | Registration: {registration_date} | MongoDB: {storage_status}")
                
            except Exception as e:
                logger.error(f"Error processing company {company.get('emri', 'Unknown')}: {e}")
                continue

@pipeline
def opencorporates_scraping_pipeline():
    """Main pipeline for scraping OpenCorporates Albania website."""
    total_pages = get_total_pages()
    process_multiple_pages(total_pages)


if __name__ == "__main__":
    opencorporates_scraping_pipeline()
