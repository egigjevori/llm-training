"""
UK Companies Instruction Dataset Generation Pipeline using ZenML
Generates diverse instruction-response pairs from UK Companies House data
"""

import logging
import json
import random
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from utils.mongo import get_mongodb_connection, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step(enable_cache=False)
def fetch_uk_companies(limit: int = 1000) -> List[Dict]:
    """Fetch UK Companies House data from MongoDB."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, "uk_companies_house")
        collection = db["companies"]

        # Fetch active companies first, then dissolved ones for diversity
        active_companies = list(collection.find({"status": "Active"}).limit(limit // 2))
        remaining_limit = limit - len(active_companies)

        if remaining_limit > 0:
            other_companies = list(collection.find({"status": {"$ne": "Active"}}).limit(remaining_limit))
            documents = active_companies + other_companies
        else:
            documents = active_companies

        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])

        client.close()
        logger.info(f"Fetched {len(documents)} UK companies (Active: {len(active_companies)}, Other: {len(documents) - len(active_companies)})")
        return documents

    except Exception as e:
        logger.error(f"Error fetching UK companies data: {e}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or text == "Unknown":
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-.,;:!?()%€$]', '', text)
    return text


def generate_uk_company_instructions(company: Dict) -> List[Dict]:
    """Generate instruction-response pairs from UK Companies House data."""
    instructions = []

    # Helper function to check if value is meaningful
    def is_meaningful(value):
        if not value or value == "Unknown" or value == "-" or value == "" or value == "N/A":
            return False
        return True

    # Get company basic info
    company_name = clean_text(company.get('company_name', ''))
    company_number = company.get('company_number', '')

    if not is_meaningful(company_name) or not is_meaningful(company_number):
        return instructions

    # Basic company information questions
    instructions.append({
        "instruction": f"What is the company number for {company_name}?",
        "response": f"The company number for {company_name} is {company_number}.",
        "category": "company_info",
        "source": "uk_companies"
    })

    instructions.append({
        "instruction": f"What is the name of the company with number {company_number}?",
        "response": f"The company with number {company_number} is {company_name}.",
        "category": "company_info",
        "source": "uk_companies"
    })

    # Company status
    status = company.get('status', '')
    if is_meaningful(status):
        instructions.append({
            "instruction": f"What is the current status of {company_name}?",
            "response": f"The current status of {company_name} is {status}.",
            "category": "status",
            "source": "uk_companies"
        })

    # Company category/type
    category = company.get('category', '')
    if is_meaningful(category):
        instructions.append({
            "instruction": f"What type of company is {company_name}?",
            "response": f"{company_name} is a {category}.",
            "category": "company_type",
            "source": "uk_companies"
        })

        instructions.append({
            "instruction": f"What is the legal structure of {company_name}?",
            "response": f"The legal structure of {company_name} is {category}.",
            "category": "company_type",
            "source": "uk_companies"
        })

    # Incorporation date
    incorporation_date = company.get('incorporation_date', '')
    if is_meaningful(incorporation_date):
        instructions.append({
            "instruction": f"When was {company_name} incorporated?",
            "response": f"{company_name} was incorporated on {incorporation_date}.",
            "category": "incorporation",
            "source": "uk_companies"
        })

        instructions.append({
            "instruction": f"What is the incorporation date of {company_name}?",
            "response": f"The incorporation date of {company_name} is {incorporation_date}.",
            "category": "incorporation",
            "source": "uk_companies"
        })

    # Address information
    address = company.get('address', {})
    if address and is_meaningful(address.get('full_address', '')):
        full_address = address.get('full_address', '')
        instructions.append({
            "instruction": f"What is the registered address of {company_name}?",
            "response": f"The registered address of {company_name} is {full_address}.",
            "category": "address",
            "source": "uk_companies"
        })

        post_town = address.get('post_town', '')
        if is_meaningful(post_town):
            instructions.append({
                "instruction": f"In which town is {company_name} registered?",
                "response": f"{company_name} is registered in {post_town}.",
                "category": "location",
                "source": "uk_companies"
            })

        county = address.get('county', '')
        if is_meaningful(county):
            instructions.append({
                "instruction": f"In which county is {company_name} located?",
                "response": f"{company_name} is located in {county}.",
                "category": "location",
                "source": "uk_companies"
            })

    # SIC codes (industry classification)
    sic_codes = company.get('sic_codes', [])
    if sic_codes and len(sic_codes) > 0:
        primary_sic = sic_codes[0]
        instructions.append({
            "instruction": f"What industry does {company_name} operate in?",
            "response": f"{company_name} operates in: {primary_sic}.",
            "category": "industry",
            "source": "uk_companies"
        })

        if len(sic_codes) > 1:
            all_sics = "; ".join(sic_codes)
            instructions.append({
                "instruction": f"What are all the business activities of {company_name}?",
                "response": f"The business activities of {company_name} include: {all_sics}.",
                "category": "industry",
                "source": "uk_companies"
            })

    # Accounts information
    accounts = company.get('accounts', {})
    if accounts:
        accounts_category = accounts.get('category', '')
        if is_meaningful(accounts_category):
            instructions.append({
                "instruction": f"What is the accounts category for {company_name}?",
                "response": f"The accounts category for {company_name} is {accounts_category}.",
                "category": "accounts",
                "source": "uk_companies"
            })

        next_due = accounts.get('next_due', '')
        if is_meaningful(next_due):
            instructions.append({
                "instruction": f"When are the next accounts due for {company_name}?",
                "response": f"The next accounts for {company_name} are due on {next_due}.",
                "category": "accounts",
                "source": "uk_companies"
            })

    # Mortgages/charges information
    mortgages = company.get('mortgages', {})
    if mortgages:
        total_charges = mortgages.get('charges', 0)
        outstanding = mortgages.get('outstanding', 0)

        if total_charges > 0:
            instructions.append({
                "instruction": f"Does {company_name} have any charges registered against it?",
                "response": f"Yes, {company_name} has {total_charges} charges registered, with {outstanding} outstanding.",
                "category": "charges",
                "source": "uk_companies"
            })
        else:
            instructions.append({
                "instruction": f"Does {company_name} have any charges registered against it?",
                "response": f"No, {company_name} has no charges registered against it.",
                "category": "charges",
                "source": "uk_companies"
            })

    # Previous names
    previous_names = company.get('previous_names', [])
    if previous_names and len(previous_names) > 0:
        latest_previous = previous_names[0]
        old_name = latest_previous.get('name', '')
        change_date = latest_previous.get('date', '')

        if is_meaningful(old_name):
            instructions.append({
                "instruction": f"Has {company_name} had any previous names?",
                "response": f"Yes, {company_name} was previously known as {old_name}" + (f" until {change_date}" if is_meaningful(change_date) else "") + ".",
                "category": "history",
                "source": "uk_companies"
            })

    # Confirmation statement
    confirmation_statement = company.get('confirmation_statement', {})
    if confirmation_statement:
        next_due = confirmation_statement.get('next_due', '')
        if is_meaningful(next_due):
            instructions.append({
                "instruction": f"When is the next confirmation statement due for {company_name}?",
                "response": f"The next confirmation statement for {company_name} is due on {next_due}.",
                "category": "confirmation_statement",
                "source": "uk_companies"
            })

    # Company overview/summary
    summary_parts = [f"{company_name} (company number: {company_number})"]

    if is_meaningful(category):
        summary_parts.append(f"is a {category}")
    if is_meaningful(status):
        summary_parts.append(f"with status {status}")
    if is_meaningful(incorporation_date):
        summary_parts.append(f"incorporated on {incorporation_date}")
    if address and is_meaningful(address.get('post_town', '')):
        summary_parts.append(f"based in {address.get('post_town')}")
    if sic_codes and len(sic_codes) > 0:
        summary_parts.append(f"operating in {sic_codes[0].split(' - ')[1] if ' - ' in sic_codes[0] else sic_codes[0]}")

    instructions.append({
        "instruction": f"Provide an overview of {company_name}.",
        "response": " ".join(summary_parts) + ".",
        "category": "overview",
        "source": "uk_companies"
    })

    return instructions

def generate_uk_comparison_instructions(uk_companies: List[Dict]) -> List[Dict]:
    """Generate comparison-based instruction-response pairs from UK companies data."""
    instructions = []

    def is_meaningful(value):
        return value and value != "Unknown" and value != "-" and value != ""

    # Group companies by location (post_town)
    location_companies = {}
    for company in uk_companies:
        address = company.get('address', {})
        post_town = address.get('post_town', '') if address else ''
        company_name = company.get('company_name', '')

        if is_meaningful(post_town) and is_meaningful(company_name):
            if post_town not in location_companies:
                location_companies[post_town] = []
            location_companies[post_town].append(company)

    # Generate location-based questions
    for post_town, companies in location_companies.items():
        if len(companies) >= 2:
            company_names = [c.get('company_name', '') for c in companies[:5]]  # Limit to 5 companies
            company_names = [name for name in company_names if is_meaningful(name)]

            if len(company_names) >= 2:
                instructions.append({
                    "instruction": f"Which companies are registered in {post_town}?",
                    "response": f"Companies registered in {post_town} include: {', '.join(company_names)}.",
                    "category": "location_comparison",
                    "source": "uk_companies"
                })

    # Group companies by industry (SIC codes)
    industry_companies = {}
    for company in uk_companies:
        sic_codes = company.get('sic_codes', [])
        company_name = company.get('company_name', '')

        if sic_codes and len(sic_codes) > 0 and is_meaningful(company_name):
            primary_industry = sic_codes[0].split(' - ')[1] if ' - ' in sic_codes[0] else sic_codes[0]

            if primary_industry not in industry_companies:
                industry_companies[primary_industry] = []
            industry_companies[primary_industry].append(company)

    # Generate industry-based questions
    for industry, companies in industry_companies.items():
        if len(companies) >= 2:
            company_names = [c.get('company_name', '') for c in companies[:3]]  # Limit to 3 companies
            company_names = [name for name in company_names if is_meaningful(name)]

            if len(company_names) >= 2:
                instructions.append({
                    "instruction": f"Which companies operate in {industry}?",
                    "response": f"Companies operating in {industry} include: {', '.join(company_names)}.",
                    "category": "industry_comparison",
                    "source": "uk_companies"
                })

    # Group companies by status
    status_companies = {}
    for company in uk_companies:
        status = company.get('status', '')
        company_name = company.get('company_name', '')

        if is_meaningful(status) and is_meaningful(company_name):
            if status not in status_companies:
                status_companies[status] = []
            status_companies[status].append(company)

    # Generate status comparison questions
    active_count = len(status_companies.get('Active', []))
    dissolved_count = len(status_companies.get('Dissolved', []))
    liquidation_count = len(status_companies.get('Liquidation', []))

    if active_count > 0 and dissolved_count > 0:
        instructions.append({
            "instruction": "How many active companies are there compared to dissolved companies?",
            "response": f"There are {active_count} active companies and {dissolved_count} dissolved companies in this dataset.",
            "category": "status_analysis",
            "source": "uk_companies"
        })

    # Company category comparison
    category_companies = {}
    for company in uk_companies:
        category = company.get('category', '')
        if is_meaningful(category):
            if category not in category_companies:
                category_companies[category] = 0
            category_companies[category] += 1

    if len(category_companies) >= 2:
        sorted_categories = sorted(category_companies.items(), key=lambda x: x[1], reverse=True)
        top_categories = sorted_categories[:3]

        category_text = ", ".join([f"{count} {category}s" for category, count in top_categories])

        instructions.append({
            "instruction": "What are the most common types of companies in this dataset?",
            "response": f"The most common company types are: {category_text}.",
            "category": "category_analysis",
            "source": "uk_companies"
        })

    return instructions


@step(enable_cache=False)
def generate_instruction_dataset(uk_companies: List[Dict]) -> List[Dict]:
    """Generate comprehensive instruction dataset from UK companies data."""
    all_instructions = []

    logger.info("Generating UK company instructions...")
    for company in uk_companies:
        company_instructions = generate_uk_company_instructions(company)
        all_instructions.extend(company_instructions)

    logger.info("Generating UK comparison instructions...")
    comparison_instructions = generate_uk_comparison_instructions(uk_companies)
    all_instructions.extend(comparison_instructions)

    # Add metadata to each instruction
    for instruction in all_instructions:
        instruction['generated_at'] = datetime.now().isoformat()
        instruction['dataset_version'] = '2.0'
        instruction['data_source'] = 'uk_companies_house'

    logger.info(f"Generated {len(all_instructions)} instruction-response pairs")
    return all_instructions


@step(enable_cache=False)
def save_instruction_dataset(instructions: List[Dict], output_path: str = "instruction_dataset.jsonl") -> str:
    """Save instruction dataset to JSONL file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for instruction in instructions:
                f.write(json.dumps(instruction, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(instructions)} instructions to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving instruction dataset: {e}")
        raise


@step(enable_cache=False)
def generate_dataset_statistics(instructions: List[Dict]) -> Dict:
    """Generate statistics about the instruction dataset."""
    stats = {
        'total_instructions': len(instructions),
        'categories': {},
        'sources': {},
        'avg_instruction_length': 0,
        'avg_response_length': 0
    }
    
    total_instruction_length = 0
    total_response_length = 0
    
    for instruction in instructions:
        # Count categories
        category = instruction.get('category', 'unknown')
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Count sources
        source = instruction.get('source', 'unknown')
        stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        # Calculate lengths
        instruction_text = instruction.get('instruction', '')
        response_text = instruction.get('response', '')
        
        total_instruction_length += len(instruction_text)
        total_response_length += len(response_text)
    
    if len(instructions) > 0:
        stats['avg_instruction_length'] = total_instruction_length / len(instructions)
        stats['avg_response_length'] = total_response_length / len(instructions)
    
    logger.info(f"Dataset statistics: {stats}")
    return stats


@pipeline
def instruction_dataset_pipeline(
    company_limit: int = 50,
    output_path: str = "datasets/uk_companies_instruction_dataset.jsonl"
):
    """Main pipeline for generating instruction dataset from UK Companies House data."""

    # Fetch UK companies from MongoDB
    uk_companies = fetch_uk_companies(company_limit)

    # Generate instruction dataset
    instructions = generate_instruction_dataset(uk_companies)

    # Save dataset
    saved_path = save_instruction_dataset(instructions, output_path)

    # Generate statistics
    stats = generate_dataset_statistics(instructions)

    # Log completion
    logger.info(f"UK Companies instruction dataset pipeline completed successfully.")
    logger.info(f"Dataset saved to: {saved_path}")
    logger.info(f"Statistics generated.")

    return {
        "instructions_count": len(instructions) if hasattr(instructions, '__len__') else "Unknown",
        "saved_path": saved_path,
        "statistics": stats,
        "data_source": "uk_companies_house",
        "pipeline_completed": True
    }


if __name__ == "__main__":
    instruction_dataset_pipeline()
