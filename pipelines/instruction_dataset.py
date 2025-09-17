"""
Custom Instruction Dataset Generation Pipeline using ZenML
Generates diverse instruction-response pairs from corporate data
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
def fetch_corporate_data(limit: int = 1000) -> List[Dict]:
    """Fetch corporate data from MongoDB."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, "opencorporates_albania")
        collection = db["companies"]
        
        # Fetch documents with limit
        documents = list(collection.find().limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        client.close()
        logger.info(f"Fetched {len(documents)} corporate documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching corporate data: {e}")
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


def generate_corporate_instructions(company: Dict) -> List[Dict]:
    """Generate instruction-response pairs from corporate data."""
    instructions = []
    
    # Helper function to check if value is meaningful
    def is_meaningful(value):
        if not value or value == "Unknown" or value == "-" or value == "":
            return False
        return True
    
    # Basic company information questions
    if is_meaningful(company.get('name')):
        company_name = company.get('name', '').strip('"')  # Remove quotes
        company_id = company.get('company_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the name of the company with company ID {company_id}?",
            "response": f"The company name is {company_name}.",
            "category": "company_info",
            "source": "corporate"
        })
        
        # Company description
        description = company.get('description', '')
        if is_meaningful(description):
            instructions.append({
                "instruction": f"Tell me about the company {company_name}.",
                "response": f"{company_name} is a company registered in Albania with company ID {company_id}. {description}",
                "category": "company_description",
                "source": "corporate"
            })
        
        # Company summary with available data
        location = company.get('location', 'Unknown location')
        reg_date = company.get('registration_date', 'Unknown date')
        status = company.get('status', 'Unknown')
        legal_form = company.get('legal_form', 'Unknown')
        
        summary_parts = [f"{company_name} is a company registered in Albania with company ID {company_id}"]
        if is_meaningful(location):
            summary_parts.append(f"located in {location}")
        if is_meaningful(reg_date):
            summary_parts.append(f"registered on {reg_date}")
        if is_meaningful(status):
            summary_parts.append(f"with status {status}")
        if is_meaningful(legal_form):
            summary_parts.append(f"and legal form {legal_form}")
        if is_meaningful(description):
            summary_parts.append(description)
        
        instructions.append({
            "instruction": f"Provide a summary of {company_name} including key details.",
            "response": ". ".join(summary_parts) + ".",
            "category": "company_summary",
            "source": "corporate"
        })
    
    # Location-based questions
    if is_meaningful(company.get('location')):
        location = company.get('location')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Where is {company_name} located?",
            "response": f"{company_name} is located in {location}.",
            "category": "location",
            "source": "corporate"
        })

    # Registration date questions
    if is_meaningful(company.get('registration_date')):
        reg_date = company.get('registration_date')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"When was {company_name} registered?",
            "response": f"{company_name} was registered on {reg_date}.",
            "category": "registration",
            "source": "corporate"
        })

    # Status questions
    if is_meaningful(company.get('status')):
        status = company.get('status')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the current status of {company_name}?",
            "response": f"The current status of {company_name} is {status}.",
            "category": "status",
            "source": "corporate"
        })

    # Legal form questions
    if is_meaningful(company.get('legal_form')):
        legal_form = company.get('legal_form')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the legal form of {company_name}?",
            "response": f"The legal form of {company_name} is {legal_form}.",
            "category": "legal_form",
            "source": "corporate"
        })

    # Capital questions
    if is_meaningful(company.get('capital')):
        capital = company.get('capital')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the capital of {company_name}?",
            "response": f"The capital of {company_name} is {capital}.",
            "category": "capital",
            "source": "corporate"
        })

    # Address questions
    if is_meaningful(company.get('address')):
        address = company.get('address')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the address of {company_name}?",
            "response": f"The address of {company_name} is {address}.",
            "category": "address",
            "source": "corporate"
        })
    
    # Business activity questions
    if is_meaningful(company.get('business_object')):
        business_activity = company.get('business_object')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the business activity of {company_name}?",
            "response": f"The business activity of {company_name} is: {business_activity}.",
            "category": "business_activity",
            "source": "corporate"
        })
    
    # Administrator questions
    if company.get('administrators') and len(company.get('administrators', [])) > 0:
        admins = company.get('administrators', [])
        admin_list = ", ".join(admins)
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Who are the administrators of {company_name}?",
            "response": f"The administrators of {company_name} are: {admin_list}.",
            "category": "management",
            "source": "corporate"
        })
    
    # Shareholder questions
    if company.get('shareholders') and len(company.get('shareholders', [])) > 0:
        shareholders = company.get('shareholders', [])
        shareholder_list = ", ".join(shareholders)
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Who are the shareholders of {company_name}?",
            "response": f"The shareholders of {company_name} are: {shareholder_list}.",
            "category": "ownership",
            "source": "corporate"
        })

    # Currency information
    if is_meaningful(company.get('currency')):
        currency = company.get('currency')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What currency does {company_name} use?",
            "response": f"{company_name} uses {currency} as its currency.",
            "category": "financial_info",
            "source": "corporate"
        })
    
    # District information
    if is_meaningful(company.get('region')):
        district = company.get('region')
        company_name = company.get('name', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"In which district is {company_name} located?",
            "response": f"{company_name} is located in the district of {district}.",
            "category": "geographic_info",
            "source": "corporate"
        })
    
    return instructions

def generate_comparison_instructions(corporate_data: List[Dict]) -> List[Dict]:
    """Generate comparison-based instruction-response pairs."""
    instructions = []
    
    # Compare companies in same location
    location_companies = {}
    for company in corporate_data:
        location = company.get('location', 'Unknown')
        if location != "Unknown":
            if location not in location_companies:
                location_companies[location] = []
            location_companies[location].append(company)
    
    for location, companies in location_companies.items():
        if len(companies) >= 2:
            company_names = [c.get('name', 'Unknown') for c in companies[:3]]  # Limit to 3 companies
            company_names = [name for name in company_names if name != "Unknown"]
            
            if len(company_names) >= 2:
                instructions.append({
                    "instruction": f"Which companies are located in {location}?",
                    "response": f"The following companies are located in {location}: {', '.join(company_names)}.",
                    "category": "comparison",
                    "source": "corporate"
                })
                
                # Location-based business analysis
                instructions.append({
                    "instruction": f"Analyze the business landscape in {location}.",
                    "response": f"To analyze the business landscape in {location}, you would examine all companies in this area, their business activities, legal forms, and economic indicators to understand the local business environment.",
                    "category": "landscape_analysis",
                    "source": "corporate"
                })
    

    
    return instructions


@step(enable_cache=False)
def generate_instruction_dataset(corporate_data: List[Dict]) -> List[Dict]:
    """Generate comprehensive instruction dataset from corporate data."""
    all_instructions = []
    
    logger.info("Generating corporate instructions...")
    for company in corporate_data:
        company_instructions = generate_corporate_instructions(company)
        all_instructions.extend(company_instructions)
    
    logger.info("Generating comparison instructions...")
    comparison_instructions = generate_comparison_instructions(corporate_data)
    all_instructions.extend(comparison_instructions)

    
    # Add metadata to each instruction
    for instruction in all_instructions:
        instruction['generated_at'] = datetime.now().isoformat()
        instruction['dataset_version'] = '1.0'
    
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
    corporate_limit: int = 5,
    output_path: str = "datasets/instruction_dataset.jsonl"
):
    """Main pipeline for generating custom instruction dataset from corporate data."""
    
    # Fetch data from MongoDB
    corporate_data = fetch_corporate_data(corporate_limit)
    
    # Generate instruction dataset
    instructions = generate_instruction_dataset(corporate_data)
    
    # Save dataset
    saved_path = save_instruction_dataset(instructions, output_path)
    
    # Generate statistics
    stats = generate_dataset_statistics(instructions)
    
    # Log completion (without using len() on StepArtifact)
    logger.info(f"Pipeline completed successfully.")
    logger.info(f"Dataset saved to: {saved_path}")
    logger.info(f"Statistics generated.")
    
    return {
        "instructions_count": len(instructions) if hasattr(instructions, '__len__') else "Unknown",
        "saved_path": saved_path,
        "statistics": stats
    }


if __name__ == "__main__":
    instruction_dataset_pipeline() 