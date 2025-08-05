"""
Custom Instruction Dataset Generation Pipeline using ZenML
Generates diverse instruction-response pairs from corporate and procurement data
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
from mongo import get_mongodb_connection, get_database

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
        return []


@step(enable_cache=False)
def fetch_procurement_data(limit: int = 1000) -> List[Dict]:
    """Fetch procurement data from MongoDB."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, "openprocurement_albania")
        collection = db["tenders"]
        
        # Fetch documents with limit
        documents = list(collection.find().limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        client.close()
        logger.info(f"Fetched {len(documents)} procurement documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching procurement data: {e}")
        return []


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
    if is_meaningful(company.get('emri')):
        company_name = company.get('emri', '').strip('"')  # Remove quotes
        nipt = company.get('nipt', 'N/A')
        
        instructions.append({
            "instruction": f"What is the name of the company with NIPT {nipt}?",
            "response": f"The company name is {company_name}.",
            "category": "company_info",
            "source": "corporate"
        })
        
        # Company description
        description = company.get('përshkrimi', '')
        if is_meaningful(description):
            instructions.append({
                "instruction": f"Tell me about the company {company_name}.",
                "response": f"{company_name} is a company registered in Albania with NIPT {nipt}. {description}",
                "category": "company_description",
                "source": "corporate"
            })
        
        # Company summary with available data
        location = company.get('vendndodhja', 'Unknown location')
        reg_date = company.get('data_regjistrimit', 'Unknown date')
        status = company.get('statusi', 'Unknown')
        legal_form = company.get('forma_ligjore', 'Unknown')
        
        summary_parts = [f"{company_name} is a company registered in Albania with NIPT {nipt}"]
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
    if is_meaningful(company.get('vendndodhja')):
        location = company.get('vendndodhja')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Where is {company_name} located?",
            "response": f"{company_name} is located in {location}.",
            "category": "location",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Find companies located in {location}.",
            "response": f"To find companies located in {location}, you would search the corporate database for companies with location field matching '{location}'.",
            "category": "search_query",
            "source": "corporate"
        })
    
    # Registration date questions
    if is_meaningful(company.get('data_regjistrimit')):
        reg_date = company.get('data_regjistrimit')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"When was {company_name} registered?",
            "response": f"{company_name} was registered on {reg_date}.",
            "category": "registration",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"How old is {company_name}?",
            "response": f"To determine the age of {company_name}, you would calculate the difference between the current date and the registration date {reg_date}.",
            "category": "calculation",
            "source": "corporate"
        })
    
    # Status questions
    if is_meaningful(company.get('statusi')):
        status = company.get('statusi')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the current status of {company_name}?",
            "response": f"The current status of {company_name} is {status}.",
            "category": "status",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Show me all companies with status '{status}'.",
            "response": f"To find all companies with status '{status}', you would filter the corporate database by status field.",
            "category": "filter_query",
            "source": "corporate"
        })
    
    # Legal form questions
    if is_meaningful(company.get('forma_ligjore')):
        legal_form = company.get('forma_ligjore')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the legal form of {company_name}?",
            "response": f"The legal form of {company_name} is {legal_form}.",
            "category": "legal_form",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Compare {company_name} with other companies of the same legal form.",
            "response": f"To compare {company_name} with other companies of the same legal form ({legal_form}), you would filter the database by legal form and analyze similarities in capital, location, and business activities.",
            "category": "comparison_analysis",
            "source": "corporate"
        })
    
    # Capital questions
    if is_meaningful(company.get('kapitali')):
        capital = company.get('kapitali')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the capital of {company_name}?",
            "response": f"The capital of {company_name} is {capital}.",
            "category": "capital",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Find companies with capital similar to {company_name}.",
            "response": f"To find companies with similar capital to {company_name} ({capital}), you would search the database for companies within a similar capital range or category.",
            "category": "range_query",
            "source": "corporate"
        })
    
    # Address questions
    if is_meaningful(company.get('adresa')):
        address = company.get('adresa')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the address of {company_name}?",
            "response": f"The address of {company_name} is {address}.",
            "category": "address",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Analyze the geographic distribution of companies in the same area as {company_name}.",
            "response": f"To analyze geographic distribution around {company_name} located at {address}, you would search for companies in the same district or area and analyze their business activities and legal forms.",
            "category": "geographic_analysis",
            "source": "corporate"
        })
    
    # Business activity questions
    if is_meaningful(company.get('objekti_veprimtarisë')):
        business_activity = company.get('objekti_veprimtarisë')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What is the business activity of {company_name}?",
            "response": f"The business activity of {company_name} is: {business_activity}.",
            "category": "business_activity",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Find companies in the same industry as {company_name}.",
            "response": f"To find companies in the same industry as {company_name} which operates in {business_activity}, you would search the database for companies with similar business activity descriptions.",
            "category": "industry_search",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Classify the business activity of {company_name} into a standard industry category.",
            "response": f"Based on the business activity '{business_activity}', {company_name} would be classified into the appropriate industry category based on the nature of their operations.",
            "category": "classification",
            "source": "corporate"
        })
    
    # Administrator questions
    if company.get('administratorët') and len(company.get('administratorët', [])) > 0:
        admins = company.get('administratorët', [])
        admin_list = ", ".join(admins)
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Who are the administrators of {company_name}?",
            "response": f"The administrators of {company_name} are: {admin_list}.",
            "category": "management",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Find other companies managed by the same administrators as {company_name}.",
            "response": f"To find other companies managed by the same administrators ({admin_list}) as {company_name}, you would search the database for companies with matching administrator names.",
            "category": "network_analysis",
            "source": "corporate"
        })
    
    # Shareholder questions
    if company.get('zotëruesit') and len(company.get('zotëruesit', [])) > 0:
        shareholders = company.get('zotëruesit', [])
        shareholder_list = ", ".join(shareholders)
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"Who are the shareholders of {company_name}?",
            "response": f"The shareholders of {company_name} are: {shareholder_list}.",
            "category": "ownership",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Analyze the ownership network for {company_name}.",
            "response": f"To analyze the ownership network for {company_name}, you would examine the shareholders ({shareholder_list}) and potentially find other companies where these same individuals or entities are shareholders.",
            "category": "network_analysis",
            "source": "corporate"
        })
    
    # Currency information
    if is_meaningful(company.get('monedha')):
        currency = company.get('monedha')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"What currency does {company_name} use?",
            "response": f"{company_name} uses {currency} as its currency.",
            "category": "financial_info",
            "source": "corporate"
        })
    
    # District information
    if is_meaningful(company.get('rrethi')):
        district = company.get('rrethi')
        company_name = company.get('emri', 'this company').strip('"')
        
        instructions.append({
            "instruction": f"In which district is {company_name} located?",
            "response": f"{company_name} is located in the district of {district}.",
            "category": "geographic_info",
            "source": "corporate"
        })
    
    return instructions


def generate_procurement_instructions(tender: Dict) -> List[Dict]:
    """Generate instruction-response pairs from procurement data."""
    instructions = []
    
    # Helper function to check if value is meaningful
    def is_meaningful(value):
        if not value or value == "Unknown" or value == "-" or value == "":
            return False
        return True
    
    # Basic tender information
    if is_meaningful(tender.get('tender_id')):
        tender_id = tender.get('tender_id')
        
        # Tender object information
        tender_object = tender.get('Objekti i Tenderit', '')
        if is_meaningful(tender_object):
            instructions.append({
                "instruction": f"What is tender ID {tender_id} about?",
                "response": f"Tender ID {tender_id} is about: {tender_object}",
                "category": "tender_info",
                "source": "procurement"
            })
        
        # Tender summary with available data
        summary_parts = [f"Tender {tender_id}"]
        
        if is_meaningful(tender_object):
            summary_parts.append(f"is about {tender_object}")
        
        value = tender.get('Vlera / Fondi Limit', '')
        if is_meaningful(value):
            summary_parts.append(f"with a value of {value}")
        
        authority = tender.get('Autoritet Prokurues', '')
        if is_meaningful(authority):
            summary_parts.append(f"managed by {authority}")
        
        deadline = tender.get('Data e fundit per dorezimin e Dokumenteve', '')
        if is_meaningful(deadline):
            summary_parts.append(f"with submission deadline of {deadline}")
        
        status = tender.get('Statusi i Tenderit', '')
        if is_meaningful(status):
            summary_parts.append(f"and status {status}")
        
        contract_type = tender.get('Lloji i Kontrates Publike', '')
        if is_meaningful(contract_type):
            summary_parts.append(f"for {contract_type}")
        
        procedure = tender.get('Lloji i Procedures', '')
        if is_meaningful(procedure):
            summary_parts.append(f"using {procedure}")
        
        if len(summary_parts) > 1:
            instructions.append({
                "instruction": f"Provide a comprehensive summary of tender {tender_id}.",
                "response": ". ".join(summary_parts) + ".",
                "category": "tender_summary",
                "source": "procurement"
            })
    
    # Tender object questions
    if is_meaningful(tender.get('Objekti i Tenderit')):
        tender_object = tender.get('Objekti i Tenderit')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the object of tender {tender_id}?",
            "response": f"The object of tender {tender_id} is: {tender_object}",
            "category": "tender_object",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Find tenders with similar objects to '{tender_object}'.",
            "response": f"To find tenders with similar objects to '{tender_object}', you would search the procurement database for tenders containing similar keywords or phrases in their object descriptions.",
            "category": "similarity_search",
            "source": "procurement"
        })
    
    # Contracting authority questions
    if is_meaningful(tender.get('Autoritet Prokurues')):
        authority = tender.get('Autoritet Prokurues')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"Who is the contracting authority for tender {tender_id}?",
            "response": f"The contracting authority for tender {tender_id} is {authority}.",
            "category": "contracting_authority",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Find all tenders issued by {authority}.",
            "response": f"To find all tenders issued by {authority}, you would filter the procurement database by contracting authority field to see all tenders managed by this entity.",
            "category": "authority_analysis",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Analyze the procurement performance of {authority}.",
            "response": f"To analyze the procurement performance of {authority}, you would examine all tenders issued by this authority, their values, success rates, and compliance with procurement regulations.",
            "category": "performance_analysis",
            "source": "procurement"
        })
    
    # Tender value questions
    if is_meaningful(tender.get('Vlera / Fondi Limit')):
        value = tender.get('Vlera / Fondi Limit')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the value of tender {tender_id}?",
            "response": f"The value of tender {tender_id} is {value}.",
            "category": "tender_value",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Find tenders with values similar to {tender_id}.",
            "response": f"To find tenders with similar values to {value}, you would search the procurement database for tenders within a similar value range or category.",
            "category": "value_range_search",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Analyze the budget allocation for tenders in the same category as {tender_id}.",
            "response": f"To analyze budget allocation for tenders similar to {tender_id} with value {value}, you would examine the distribution of tender values in the same category or sector.",
            "category": "budget_analysis",
            "source": "procurement"
        })
    
    # Submission deadline questions
    if is_meaningful(tender.get('Data e fundit per dorezimin e Dokumenteve')):
        deadline = tender.get('Data e fundit per dorezimin e Dokumenteve')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the submission deadline for tender {tender_id}?",
            "response": f"The submission deadline for tender {tender_id} is {deadline}.",
            "category": "deadline",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Calculate how many days are left until the deadline for tender {tender_id}.",
            "response": f"To calculate the remaining days until the deadline for tender {tender_id} with deadline {deadline}, you would subtract the current date from the deadline date.",
            "category": "urgency_calculation",
            "source": "procurement"
        })
    
    # Tender status questions
    if is_meaningful(tender.get('Statusi i Tenderit')):
        status = tender.get('Statusi i Tenderit')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the status of tender {tender_id}?",
            "response": f"The status of tender {tender_id} is {status}.",
            "category": "tender_status",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Show me all tenders with status '{status}'.",
            "response": f"To find all tenders with status '{status}', you would filter the procurement database by status field.",
            "category": "status_filter",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Track the status changes for tender {tender_id}.",
            "response": f"To track status changes for tender {tender_id}, you would examine the tender's modification history to see how the status has evolved from initial publication to the current status of {status}.",
            "category": "status_tracking",
            "source": "procurement"
        })
    
    # Contract type questions
    if is_meaningful(tender.get('Lloji i Kontrates Publike')):
        contract_type = tender.get('Lloji i Kontrates Publike')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What type of contract is tender {tender_id}?",
            "response": f"Tender {tender_id} is for {contract_type}.",
            "category": "contract_type",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Compare {contract_type} tenders with other contract types.",
            "response": f"To compare {contract_type} tenders with other types, you would analyze differences in procedures, requirements, evaluation criteria, and typical values between different contract categories.",
            "category": "type_comparison",
            "source": "procurement"
        })
    
    # Procedure type questions
    if is_meaningful(tender.get('Lloji i Procedures')):
        procedure = tender.get('Lloji i Procedures')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the procedure type for tender {tender_id}?",
            "response": f"The procedure type for tender {tender_id} is {procedure}.",
            "category": "procedure_type",
            "source": "procurement"
        })
        
        instructions.append({
            "instruction": f"Explain the {procedure} procedure for tender {tender_id}.",
            "response": f"The {procedure} procedure for tender {tender_id} involves specific steps, evaluation criteria, and timeline requirements as defined by Albanian procurement regulations.",
            "category": "procedure_explanation",
            "source": "procurement"
        })
    
    # Contract duration
    if is_meaningful(tender.get('Kohezgjatja e kontrates')):
        duration = tender.get('Kohezgjatja e kontrates')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What is the contract duration for tender {tender_id}?",
            "response": f"The contract duration for tender {tender_id} is {duration}.",
            "category": "contract_duration",
            "source": "procurement"
        })
    
    # Winning offer
    if is_meaningful(tender.get('Oferta fituese  Leke pa TVSH')):
        winning_offer = tender.get('Oferta fituese  Leke pa TVSH')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"What was the winning offer for tender {tender_id}?",
            "response": f"The winning offer for tender {tender_id} was {winning_offer} Leke without VAT.",
            "category": "winning_offer",
            "source": "procurement"
        })
    
    # Contracting economic operator
    if is_meaningful(tender.get('Operator Ekonomik Kontraktues')):
        operator = tender.get('Operator Ekonomik Kontraktues')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"Who is the contracting economic operator for tender {tender_id}?",
            "response": f"The contracting economic operator for tender {tender_id} is {operator}.",
            "category": "contracting_operator",
            "source": "procurement"
        })
    
    # Number of competing operators
    if is_meaningful(tender.get('Nr', {}).get(' i Operatoreve Konkurues')):
        num_operators = tender.get('Nr', {}).get(' i Operatoreve Konkurues')
        tender_id = tender.get('tender_id', 'N/A')
        
        instructions.append({
            "instruction": f"How many operators competed for tender {tender_id}?",
            "response": f"{num_operators} operators competed for tender {tender_id}.",
            "category": "competition_analysis",
            "source": "procurement"
        })
    
    return instructions


def generate_comparison_instructions(corporate_data: List[Dict], procurement_data: List[Dict]) -> List[Dict]:
    """Generate comparison-based instruction-response pairs."""
    instructions = []
    
    # Compare companies in same location
    location_companies = {}
    for company in corporate_data:
        location = company.get('vendndodhja', 'Unknown')
        if location != "Unknown":
            if location not in location_companies:
                location_companies[location] = []
            location_companies[location].append(company)
    
    for location, companies in location_companies.items():
        if len(companies) >= 2:
            company_names = [c.get('emri', 'Unknown') for c in companies[:3]]  # Limit to 3 companies
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
    
    # Compare tenders by value
    tenders_with_value = [t for t in procurement_data if t.get('Vlera e tenderit') and t.get('Vlera e tenderit') != "Unknown"]
    if len(tenders_with_value) >= 2:
        # Find highest value tender
        highest_tender = max(tenders_with_value, key=lambda x: float(x.get('Vlera e tenderit', '0').replace(',', '').replace('€', '').replace(' ', '')) if x.get('Vlera e tenderit', '0').replace(',', '').replace('€', '').replace(' ', '').isdigit() else 0)
        
        instructions.append({
            "instruction": "Which tender has the highest value?",
            "response": f"Tender {highest_tender.get('tender_id', 'N/A')} has the highest value: {highest_tender.get('Vlera e tenderit', 'N/A')}.",
            "category": "comparison",
            "source": "procurement"
        })
        
        # Value distribution analysis
        instructions.append({
            "instruction": "Analyze the distribution of tender values in the database.",
            "response": f"To analyze tender value distribution, you would examine the range, median, and distribution patterns of tender values across different categories and authorities to understand procurement spending patterns.",
            "category": "distribution_analysis",
            "source": "procurement"
        })
    
    # Cross-domain comparisons
    if corporate_data and procurement_data:
        # Company vs Tender analysis
        instructions.append({
            "instruction": "Compare the geographic distribution of companies and tenders.",
            "response": "To compare geographic distribution, you would analyze how company locations correlate with tender locations, identifying areas with high business activity but low procurement activity or vice versa.",
            "category": "cross_domain_analysis",
            "source": "combined"
        })
        
        # Economic activity correlation
        instructions.append({
            "instruction": "Analyze the correlation between business activity and procurement activity.",
            "response": "To analyze this correlation, you would examine how business sectors represented by companies align with procurement categories, identifying sectors with high business presence but low procurement opportunities.",
            "category": "correlation_analysis",
            "source": "combined"
        })
    
    return instructions


def generate_analytical_instructions(corporate_data: List[Dict], procurement_data: List[Dict]) -> List[Dict]:
    """Generate analytical instruction-response pairs."""
    instructions = []
    
    # Company statistics
    total_companies = len(corporate_data)
    active_companies = len([c for c in corporate_data if c.get('statusi', '').lower() in ['aktiv', 'active']])
    
    instructions.append({
        "instruction": "How many companies are registered in the database?",
        "response": f"There are {total_companies} companies registered in the database.",
        "category": "analytics",
        "source": "corporate"
    })
    
    if active_companies > 0:
        instructions.append({
            "instruction": "How many companies are currently active?",
            "response": f"There are {active_companies} active companies out of {total_companies} total companies.",
            "category": "analytics",
            "source": "corporate"
        })
        
        # Market share analysis
        instructions.append({
            "instruction": "Calculate the market share of active companies.",
            "response": f"To calculate market share, you would analyze the proportion of active companies ({active_companies}) relative to total companies ({total_companies}), and further break this down by industry sectors and geographic regions.",
            "category": "market_analysis",
            "source": "corporate"
        })
    
    # Tender statistics
    total_tenders = len(procurement_data)
    open_tenders = len([t for t in procurement_data if t.get('Statusi', '').lower() in ['hapur', 'open', 'aktiv']])
    
    instructions.append({
        "instruction": "How many tenders are in the database?",
        "response": f"There are {total_tenders} tenders in the database.",
        "category": "analytics",
        "source": "procurement"
    })
    
    if open_tenders > 0:
        instructions.append({
            "instruction": "How many tenders are currently open?",
            "response": f"There are {open_tenders} open tenders out of {total_tenders} total tenders.",
            "category": "analytics",
            "source": "procurement"
        })
        
        # Procurement efficiency analysis
        instructions.append({
            "instruction": "Analyze procurement efficiency based on tender status distribution.",
            "response": f"To analyze procurement efficiency, you would examine the ratio of open tenders ({open_tenders}) to total tenders ({total_tenders}), along with completion rates, average processing times, and success rates across different authorities.",
            "category": "efficiency_analysis",
            "source": "procurement"
        })
    
    # Advanced analytics
    if corporate_data:
        # Industry distribution analysis
        business_activities = [c.get('objekti_veprimtarisë', 'Unknown') for c in corporate_data if c.get('objekti_veprimtarisë', 'Unknown') != "Unknown"]
        if business_activities:
            instructions.append({
                "instruction": "Analyze the distribution of business activities across companies.",
                "response": f"To analyze business activity distribution, you would categorize companies by their business activities, identify the most common sectors, and examine geographic and temporal patterns in business formation.",
                "category": "industry_analysis",
                "source": "corporate"
            })
        
        # Capital analysis
        companies_with_capital = [c for c in corporate_data if c.get('kapitali') and c.get('kapitali') != "Unknown"]
        if companies_with_capital:
            instructions.append({
                "instruction": "Analyze the capital distribution across companies.",
                "response": f"To analyze capital distribution, you would examine the range, median, and distribution of company capital, identify capital-intensive sectors, and analyze the relationship between capital and business activity.",
                "category": "capital_analysis",
                "source": "corporate"
            })
    
    if procurement_data:
        # Authority performance analysis
        authorities = [t.get('Autoriteti kontraktues', 'Unknown') for t in procurement_data if t.get('Autoriteti kontraktues', 'Unknown') != "Unknown"]
        if authorities:
            instructions.append({
                "instruction": "Analyze the performance of different contracting authorities.",
                "response": f"To analyze authority performance, you would examine tender success rates, average values, processing times, and compliance rates for each contracting authority to identify best practices and areas for improvement.",
                "category": "authority_performance",
                "source": "procurement"
            })
        
        # Temporal analysis
        instructions.append({
            "instruction": "Analyze procurement trends over time.",
            "response": f"To analyze temporal trends, you would examine tender publication patterns, seasonal variations, year-over-year growth, and the evolution of tender values and categories over different time periods.",
            "category": "temporal_analysis",
            "source": "procurement"
        })
    
    # Cross-domain analytics
    if corporate_data and procurement_data:
        # Economic impact analysis
        instructions.append({
            "instruction": "Analyze the economic impact of procurement on business development.",
            "response": "To analyze economic impact, you would examine how procurement opportunities correlate with business formation, identify sectors with high procurement activity but low business presence, and analyze the multiplier effects of procurement spending.",
            "category": "economic_impact",
            "source": "combined"
        })
        
        # Risk analysis
        instructions.append({
            "instruction": "Identify potential risks in the procurement and business ecosystem.",
            "response": "To identify risks, you would analyze concentration of procurement spending, dependency on specific authorities, geographic concentration of business activity, and potential conflicts of interest between companies and procurement opportunities.",
            "category": "risk_analysis",
            "source": "combined"
        })
    
    return instructions


def generate_complex_instructions(corporate_data: List[Dict], procurement_data: List[Dict]) -> List[Dict]:
    """Generate complex multi-step and scenario-based instruction-response pairs."""
    instructions = []
    
    # Multi-step analysis scenarios
    if corporate_data and procurement_data:
        # Business opportunity analysis
        instructions.append({
            "instruction": "A new company wants to enter the Albanian market. Help them identify business opportunities by analyzing both corporate and procurement data.",
            "response": "To help a new company identify opportunities, I would: 1) Analyze business activity distribution to identify underserved sectors, 2) Examine procurement categories to find high-value opportunities, 3) Identify geographic areas with business gaps, 4) Analyze successful companies in similar sectors, 5) Review procurement patterns to understand market demand, and 6) Provide recommendations based on capital requirements and competition levels.",
            "category": "business_consulting",
            "source": "combined"
        })
        
        # Investment analysis
        instructions.append({
            "instruction": "An investor wants to analyze the Albanian business and procurement landscape for investment opportunities. Provide a comprehensive analysis.",
            "response": "For investment analysis, I would: 1) Identify high-growth sectors based on business formation trends, 2) Analyze procurement spending patterns to understand government priorities, 3) Examine capital requirements and financial performance across sectors, 4) Identify geographic investment opportunities, 5) Analyze regulatory compliance and business risks, 6) Review competitive landscape and market concentration, and 7) Provide investment recommendations with risk assessment.",
            "category": "investment_analysis",
            "source": "combined"
        })
        
        # Policy recommendation scenario
        instructions.append({
            "instruction": "As a policy advisor, analyze the current business and procurement ecosystem and provide recommendations for improvement.",
            "response": "As a policy advisor, I would: 1) Analyze business formation barriers and success factors, 2) Examine procurement efficiency and transparency issues, 3) Identify sector-specific challenges and opportunities, 4) Analyze geographic disparities in business and procurement activity, 5) Review regulatory compliance and enforcement patterns, 6) Assess economic impact of current policies, and 7) Provide evidence-based policy recommendations for business development and procurement reform.",
            "category": "policy_analysis",
            "source": "combined"
        })
        
        # Due diligence scenario
        instructions.append({
            "instruction": "Perform due diligence on a potential business partner using the corporate and procurement databases.",
            "response": "For due diligence, I would: 1) Verify company registration and legal status, 2) Analyze financial performance and capital structure, 3) Review business activities and industry positioning, 4) Examine management and ownership structure, 5) Check for procurement participation and performance, 6) Analyze geographic and sector presence, 7) Review compliance history and regulatory standing, and 8) Provide a comprehensive risk assessment and recommendation.",
            "category": "due_diligence",
            "source": "combined"
        })
    
    # Complex query scenarios
    if corporate_data:
        # Market entry strategy
        instructions.append({
            "instruction": "Develop a market entry strategy for a technology company in Albania.",
            "response": "To develop a market entry strategy, I would: 1) Analyze existing technology companies and their performance, 2) Identify gaps in the technology sector, 3) Examine procurement opportunities in technology and digital services, 4) Analyze capital requirements and funding patterns, 5) Review regulatory requirements for technology businesses, 6) Identify potential partners and competitors, and 7) Provide a phased market entry plan with risk mitigation strategies.",
            "category": "strategy_development",
            "source": "corporate"
        })
        
        # Competitive intelligence
        instructions.append({
            "instruction": "Conduct competitive intelligence analysis for a specific industry sector.",
            "response": "For competitive intelligence, I would: 1) Identify all companies in the target sector, 2) Analyze their business activities and specializations, 3) Examine their financial performance and capital structure, 4) Review their geographic presence and market coverage, 5) Analyze their management and ownership structures, 6) Identify their procurement participation patterns, and 7) Provide competitive positioning analysis and strategic insights.",
            "category": "competitive_intelligence",
            "source": "corporate"
        })
    
    if procurement_data:
        # Procurement strategy development
        instructions.append({
            "instruction": "Develop a procurement strategy for a government agency to improve efficiency and transparency.",
            "response": "To develop a procurement strategy, I would: 1) Analyze current procurement patterns and performance metrics, 2) Identify bottlenecks and inefficiencies in the process, 3) Examine authority performance and best practices, 4) Analyze tender success rates and failure reasons, 5) Review value distribution and budget allocation patterns, 6) Identify opportunities for process improvement, and 7) Provide recommendations for enhanced transparency, efficiency, and compliance.",
            "category": "procurement_strategy",
            "source": "procurement"
        })
        
        # Supplier development
        instructions.append({
            "instruction": "Design a supplier development program based on procurement data analysis.",
            "response": "For supplier development, I would: 1) Analyze procurement categories with supply gaps, 2) Identify potential suppliers from the corporate database, 3) Examine their capabilities and financial capacity, 4) Analyze procurement requirements and qualification criteria, 5) Identify training and capacity building needs, 6) Review successful supplier patterns and best practices, and 7) Design a comprehensive supplier development program with support mechanisms.",
            "category": "supplier_development",
            "source": "procurement"
        })
    
    return instructions


@step(enable_cache=False)
def generate_instruction_dataset(corporate_data: List[Dict], procurement_data: List[Dict]) -> List[Dict]:
    """Generate comprehensive instruction dataset from corporate and procurement data."""
    all_instructions = []
    
    logger.info("Generating corporate instructions...")
    for company in corporate_data:
        company_instructions = generate_corporate_instructions(company)
        all_instructions.extend(company_instructions)
    
    logger.info("Generating procurement instructions...")
    for tender in procurement_data:
        tender_instructions = generate_procurement_instructions(tender)
        all_instructions.extend(tender_instructions)
    
    logger.info("Generating comparison instructions...")
    comparison_instructions = generate_comparison_instructions(corporate_data, procurement_data)
    all_instructions.extend(comparison_instructions)
    
    logger.info("Generating analytical instructions...")
    analytical_instructions = generate_analytical_instructions(corporate_data, procurement_data)
    all_instructions.extend(analytical_instructions)
    
    logger.info("Generating complex scenario instructions...")
    complex_instructions = generate_complex_instructions(corporate_data, procurement_data)
    all_instructions.extend(complex_instructions)
    
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
        return ""


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
    corporate_limit: int = 1000,
    procurement_limit: int = 1000,
    output_path: str = "datasets/instruction_dataset.jsonl"
):
    """Main pipeline for generating custom instruction dataset from corporate and procurement data."""
    
    # Fetch data from MongoDB
    corporate_data = fetch_corporate_data(corporate_limit)
    procurement_data = fetch_procurement_data(procurement_limit)
    
    # Generate instruction dataset
    instructions = generate_instruction_dataset(corporate_data, procurement_data)
    
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