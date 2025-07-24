#!/usr/bin/env python3
"""
OpenCorporates Albania Web Scraping Runner

This script provides a command-line interface to run the scraping pipeline
with configurable parameters.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.open_corporate import (
    get_total_pages,
    scrape_company_listings,
    scrape_company_details,
    combine_all_companies,
    save_to_csv
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_scraping_pipeline(max_pages: int = 5, max_companies_per_page: int = None):
    """Run the complete scraping pipeline with the given parameters."""
    logger.info("Starting OpenCorporates Albania scraping pipeline")
    
    logger.info("Step 1: Getting total number of pages...")
    total_pages = get_total_pages()
    logger.info(f"Found {total_pages} total pages")
    
    pages_to_scrape = min(max_pages, total_pages)
    logger.info(f"Will scrape {pages_to_scrape} pages")
    
    logger.info("Step 2: Scraping company listings...")
    all_companies = []
    
    for page in range(1, pages_to_scrape + 1):
        logger.info(f"Scraping page {page}/{pages_to_scrape}")
        companies = scrape_company_listings(page)
        
        if max_companies_per_page:
            companies = companies[:max_companies_per_page]
        
        all_companies.extend(companies)
        logger.info(f"Found {len(companies)} companies on page {page}")
    
    logger.info(f"Total companies found: {len(all_companies)}")
    
    logger.info("Step 3: Scraping detailed company information...")
    company_details = []
    
    for i, company in enumerate(all_companies, 1):
        logger.info(f"Scraping details for company {i}/{len(all_companies)}: {company['emri']}")
        
        try:
            detail = scrape_company_details(company)
            company_details.append(detail)
            logger.info(f"Successfully scraped details for {company['emri']}")
        except Exception as e:
            logger.error(f"Error scraping details for {company['emri']}: {e}")
            company_details.append(detail)
    
    logger.info(f"Successfully scraped details for {len(company_details)} companies")
    
    logger.info("Step 4: Combining data and saving to CSV...")
    df = combine_all_companies(company_details)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"opencorporates_albania_{timestamp}.csv"
    
    save_to_csv(df, filename)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Data saved to: {filename}")
    logger.info(f"Total companies scraped: {len(company_details)}")
    
    print("\n" + "=" * 50)
    print("SCRAPING SUMMARY")
    print("=" * 50)
    print(f"Pages scraped: {pages_to_scrape}")
    print(f"Companies found: {len(all_companies)}")
    print(f"Companies with details: {len(company_details)}")
    print(f"Output file: {filename}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="OpenCorporates Albania Web Scraper")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to scrape (default: 5)"
    )
    parser.add_argument(
        "--max-companies-per-page",
        type=int,
        default=None,
        help="Maximum number of companies to scrape per page (default: all)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal data"
    )
    
    args = parser.parse_args()
    
    if args.test_mode:
        logger.info("Running in test mode")
        run_scraping_pipeline(max_pages=1, max_companies_per_page=2)
    else:
        run_scraping_pipeline(
            max_pages=args.max_pages,
            max_companies_per_page=args.max_companies_per_page
        )


if __name__ == "__main__":
    main() 