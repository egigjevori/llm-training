"""
UK Companies House CSV Parser Pipeline using ZenML
Parses the Companies House CSV data and stores it in MongoDB.
"""

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Generator, Optional, Any
import sys
import os
import itertools

from tqdm import tqdm
from pymongo import ReplaceOne

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from utils.mongo import get_mongodb_connection, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_field_value(value: str) -> str:
    """Clean and normalize field values."""
    if not value or value.strip() == '':
        return ""

    # Remove quotes and strip whitespace
    cleaned = value.strip().strip('"').strip("'").strip()
    return cleaned


def parse_date(date_str: str) -> str:
    """Parse date from DD/MM/YYYY format to ISO format."""
    if not date_str or date_str.strip() == "":
        return ""

    try:
        # Handle DD/MM/YYYY format
        if "/" in date_str:
            day, month, year = date_str.strip().split("/")
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return date_str.strip()
    except (ValueError, AttributeError):
        return date_str.strip() if date_str else ""


def build_full_address(address_dict: Dict[str, str]) -> str:
    """Build a full address string from address components."""
    parts = []
    for key in ['care_of', 'line1', 'line2', 'post_town', 'county', 'country', 'post_code']:
        value = address_dict.get(key, '').strip()
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def parse_company_row(row: Dict[str, str], row_index: int) -> Optional[Dict[str, Any]]:
    """Parse a single CSV row into a company document."""
    try:
        # Skip if no company number (primary key)
        company_number = row.get('CompanyNumber', '').strip()
        if not company_number:
            logger.warning(f"Row {row_index+1}: Missing company number, skipping")
            return None

        # Build address
        address = {
            'care_of': row.get('RegAddress.CareOf', ''),
            'po_box': row.get('RegAddress.POBox', ''),
            'line1': row.get('RegAddress.AddressLine1', ''),
            'line2': row.get('RegAddress.AddressLine2', ''),
            'post_town': row.get('RegAddress.PostTown', ''),
            'county': row.get('RegAddress.County', ''),
            'country': row.get('RegAddress.Country', ''),
            'post_code': row.get('RegAddress.PostCode', ''),
        }
        address['full_address'] = build_full_address(address)

        # Build accounts info
        accounts = {
            'ref_day': int(row.get('Accounts.AccountRefDay', '0') or '0'),
            'ref_month': int(row.get('Accounts.AccountRefMonth', '0') or '0'),
            'next_due': parse_date(row.get('Accounts.NextDueDate', '')),
            'last_made_up': parse_date(row.get('Accounts.LastMadeUpDate', '')),
            'category': row.get('Accounts.AccountCategory', '')
        }

        # Build returns info
        returns = {
            'next_due': parse_date(row.get('Returns.NextDueDate', '')),
            'last_made_up': parse_date(row.get('Returns.LastMadeUpDate', ''))
        }

        # Build mortgages info
        mortgages = {
            'charges': int(row.get('Mortgages.NumMortCharges', '0') or '0'),
            'outstanding': int(row.get('Mortgages.NumMortOutstanding', '0') or '0'),
            'part_satisfied': int(row.get('Mortgages.NumMortPartSatisfied', '0') or '0'),
            'satisfied': int(row.get('Mortgages.NumMortSatisfied', '0') or '0')
        }

        # Collect SIC codes
        sic_codes = []
        for i in range(1, 5):
            sic_text = row.get(f'SICCode.SicText_{i}', '').strip()
            if sic_text and sic_text != '':
                sic_codes.append(sic_text)

        # Collect previous names
        previous_names = []
        for i in range(1, 11):
            name = row.get(f'PreviousName_{i}.CompanyName', '').strip()
            date = row.get(f'PreviousName_{i}.CONDATE', '').strip()
            if name and name != '':
                previous_names.append({
                    'date': parse_date(date) if date else '',
                    'name': name
                })

        # Build partnerships info
        partnerships = {
            'general_partners': int(row.get('LimitedPartnerships.NumGenPartners', '0') or '0'),
            'limited_partners': int(row.get('LimitedPartnerships.NumLimPartners', '0') or '0')
        }

        # Build confirmation statement info
        confirmation_statement = {
            'next_due': parse_date(row.get('ConfStmtNextDueDate', '')),
            'last_made_up': parse_date(row.get('ConfStmtLastMadeUpDate', ''))
        }

        # Build final document
        company_doc = {
            '_id': company_number,  # Use company number as primary key
            'company_name': row.get('CompanyName', ''),
            'company_number': company_number,
            'status': row.get('CompanyStatus', ''),
            'category': row.get('CompanyCategory', ''),
            'country_of_origin': row.get('CountryOfOrigin', ''),
            'incorporation_date': parse_date(row.get('IncorporationDate', '')),
            'dissolution_date': parse_date(row.get('DissolutionDate', '')),
            'address': address,
            'accounts': accounts,
            'returns': returns,
            'mortgages': mortgages,
            'sic_codes': sic_codes,
            'previous_names': previous_names,
            'partnerships': partnerships,
            'confirmation_statement': confirmation_statement,
            'uri': row.get('URI', ''),
            'imported_at': datetime.now().isoformat()
        }

        return company_doc

    except Exception as e:
        logger.warning(f"Row {row_index+1}: Error parsing company {row.get('CompanyName', 'Unknown')}: {e}")
        return None


@step(enable_cache=False)
def process_companies_sequentially(
    csv_path: str = "data/uk_companies_house/BasicCompanyData-2025-09-01-part1_7.csv",
    chunk_size: int = 1000
) -> Dict[str, int]:
    """Process CSV companies in chunks to preserve memory."""
    logger.info(f"Processing companies from: {csv_path} in chunks of {chunk_size}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        # Initialize MongoDB connection
        client = get_mongodb_connection()
        db = get_database(client, "uk_companies_house")
        collection = db["companies"]

        # Create index on company_number for fast lookups
        collection.create_index("company_number", unique=True)
        logger.info("Created index on company_number")

        # Initialize stats
        overall_stats = {
            'total_processed': 0,
            'inserted': 0,
            'updated': 0,
            'errors': 0
        }

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Clean headers by removing leading/trailing spaces
            original_fieldnames = reader.fieldnames
            cleaned_fieldnames = [name.strip() for name in original_fieldnames]

            logger.info(f"Original headers: {len(original_fieldnames)}")
            logger.info(f"Headers with leading spaces found: {[h for h in original_fieldnames if h.startswith(' ')]}")

            chunk_num = 0
            row_index = 0

            while True:
                # Read a chunk of rows
                chunk_rows = list(itertools.islice(reader, chunk_size))
                if not chunk_rows:
                    break  # No more data

                chunk_num += 1
                logger.info(f"Processing chunk {chunk_num} with {len(chunk_rows)} rows")

                # Parse chunk
                parsed_companies = []
                for row_dict in chunk_rows:
                    # Create new dict with cleaned headers
                    cleaned_row = {}
                    for original_key, cleaned_key in zip(original_fieldnames, cleaned_fieldnames):
                        cleaned_row[cleaned_key] = clean_field_value(row_dict[original_key])

                    # Parse the row
                    company_doc = parse_company_row(cleaned_row, row_index)
                    if company_doc:
                        parsed_companies.append(company_doc)

                    row_index += 1

                # Store chunk to MongoDB if we have data
                if parsed_companies:
                    try:
                        # Prepare bulk operations
                        operations = []
                        for company in parsed_companies:
                            operation = ReplaceOne(
                                filter={'_id': company['_id']},
                                replacement=company,
                                upsert=True
                            )
                            operations.append(operation)

                        # Execute batch
                        result = collection.bulk_write(operations)
                        overall_stats['total_processed'] += len(parsed_companies)
                        overall_stats['inserted'] += result.upserted_count
                        overall_stats['updated'] += result.modified_count

                        logger.info(f"Chunk {chunk_num}: "
                                  f"Processed {len(parsed_companies)}, "
                                  f"Inserted {result.upserted_count}, "
                                  f"Updated {result.modified_count}")

                    except Exception as e:
                        logger.error(f"Error storing chunk {chunk_num}: {e}")
                        overall_stats['errors'] += len(parsed_companies)

                # Clear chunk from memory
                parsed_companies.clear()

        client.close()
        logger.info(f"Sequential processing completed. Stats: {overall_stats}")
        return overall_stats

    except Exception as e:
        logger.warning(f"MongoDB not available: {e}")

        stats = {
            'total_processed': 0,
            'inserted': 0,
            'updated': 0,
            'errors': 0
        }

        logger.info(f"MongoDB storage failed")
        return stats



@step(enable_cache=False)
def store_to_mongodb(companies: List[Dict[str, Any]], batch_size: int = 1000) -> Dict[str, int]:
    """Store companies in MongoDB with batch processing."""
    logger.info(f"Storing {len(companies)} companies to MongoDB...")

    try:
        client = get_mongodb_connection()
        db = get_database(client, "uk_companies_house")
        collection = db["companies"]

        # Create index on company_number for fast lookups
        collection.create_index("company_number", unique=True)
        logger.info("Created index on company_number")

        # Stats tracking
        stats = {
            'total_processed': 0,
            'inserted': 0,
            'updated': 0,
            'errors': 0
        }

        # Process in batches
        for i in tqdm(range(0, len(companies), batch_size), desc="Storing batches"):
            batch = companies[i:i + batch_size]

            try:
                # Prepare bulk operations
                operations = []
                for company in batch:
                    operation = ReplaceOne(
                        filter={'_id': company['_id']},
                        replacement=company,
                        upsert=True
                    )
                    operations.append(operation)

                # Execute batch
                if operations:
                    result = collection.bulk_write(operations)
                    stats['total_processed'] += len(batch)
                    stats['inserted'] += result.upserted_count
                    stats['updated'] += result.modified_count

                    logger.info(f"Batch {i//batch_size + 1}: "
                              f"Processed {len(batch)}, "
                              f"Inserted {result.upserted_count}, "
                              f"Updated {result.modified_count}")

            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                stats['errors'] += len(batch)
                continue

        client.close()

        logger.info(f"Storage completed. Stats: {stats}")
        return stats

    except Exception as e:
        logger.warning(f"MongoDB not available: {e}")

        stats = {
            'total_processed': len(companies),
            'inserted': 0,
            'updated': 0,
            'errors': len(companies)
        }

        logger.info(f"MongoDB storage failed for {len(companies)} companies")
        return stats


@pipeline
def uk_companies_parser_pipeline(
    csv_path: str = "../data/uk_companies_house/BasicCompanyData-2025-09-01-part1_7.csv",
    chunk_size: int = 1000
):
    """Main pipeline for parsing UK Companies House CSV and storing in MongoDB with memory optimization."""

    logger.info("Starting UK Companies House CSV parser pipeline with sequential processing")

    # Process companies sequentially in chunks
    stats = process_companies_sequentially(csv_path, chunk_size)

    logger.info("Pipeline completed successfully!")
    logger.info("Check MongoDB for stored companies data.")

    return {
        "csv_path": csv_path,
        "stats": stats,
        "pipeline_completed": True
    }


if __name__ == "__main__":
    uk_companies_parser_pipeline()