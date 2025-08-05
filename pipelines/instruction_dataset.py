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


@step
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


@step
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
    
    # Basic company information questions
    if company.get('emri') and company.get('emri') != "Unknown":
        instructions.append({
            "instruction": f"What is the name of the company with NIPT {company.get('nipt', 'N/A')}?",
            "response": f"The company name is {company.get('emri')}.",
            "category": "company_info",
            "source": "corporate"
        })
        
        instructions.append({
            "instruction": f"Tell me about the company {company.get('emri')}.",
            "response": f"{company.get('emri')} is a company registered in Albania with NIPT {company.get('nipt', 'N/A')}. {company.get('përshkrimi', '')}",
            "category": "company_description",
            "source": "corporate"
        })
        
        # Company summary instruction
        instructions.append({
            "instruction": f"Provide a summary of {company.get('emri')} including key details.",
            "response": f"{company.get('emri')} is a company registered in Albania with NIPT {company.get('nipt', 'N/A')}. It is located in {company.get('vendndodhja', 'Unknown location')} and was registered on {company.get('data_regjistrimit', 'Unknown date')}. The company's status is {company.get('statusi', 'Unknown')} and its legal form is {company.get('forma_ligjore', 'Unknown')}. {company.get('përshkrimi', '')}",
            "category": "company_summary",
            "source": "corporate"
        })
    
    # Location-based questions
    if company.get('vendndodhja') and company.get('vendndodhja') != "Unknown":
        instructions.append({
            "instruction": f"Where is {company.get('emri', 'this company')} located?",
            "response": f"{company.get('emri', 'The company')} is located in {company.get('vendndodhja')}.",
            "category": "location",
            "source": "corporate"
        })
        
        # Location-based search instruction
        instructions.append({
            "instruction": f"Find companies located in {company.get('vendndodhja')}.",
            "response": f"To find companies located in {company.get('vendndodhja')}, you would search the corporate database for companies with location field matching '{company.get('vendndodhja')}'. {company.get('emri', 'This company')} is one such company located there.",
            "category": "search_query",
            "source": "corporate"
        })
    
    # Registration date questions
    if company.get('data_regjistrimit') and company.get('data_regjistrimit') != "Unknown":
        instructions.append({
            "instruction": f"When was {company.get('emri', 'this company')} registered?",
            "response": f"{company.get('emri', 'The company')} was registered on {company.get('data_regjistrimit')}.",
            "category": "registration",
            "source": "corporate"
        })
        
        # Age calculation instruction
        instructions.append({
            "instruction": f"How old is {company.get('emri', 'this company')}?",
            "response": f"To determine the age of {company.get('emri', 'the company')}, you would calculate the difference between the current date and the registration date {company.get('data_regjistrimit')}.",
            "category": "calculation",
            "source": "corporate"
        })
    
    # Status questions
    if company.get('statusi') and company.get('statusi') != "Unknown":
        instructions.append({
            "instruction": f"What is the current status of {company.get('emri', 'this company')}?",
            "response": f"The current status of {company.get('emri', 'the company')} is {company.get('statusi')}.",
            "category": "status",
            "source": "corporate"
        })
        
        # Status filter instruction
        instructions.append({
            "instruction": f"Show me all companies with status '{company.get('statusi')}'.",
            "response": f"To find all companies with status '{company.get('statusi')}', you would filter the corporate database by status field. {company.get('emri', 'This company')} is one example with this status.",
            "category": "filter_query",
            "source": "corporate"
        })
    
    # Legal form questions
    if company.get('forma_ligjore') and company.get('forma_ligjore') != "Unknown":
        instructions.append({
            "instruction": f"What is the legal form of {company.get('emri', 'this company')}?",
            "response": f"The legal form of {company.get('emri', 'the company')} is {company.get('forma_ligjore')}.",
            "category": "legal_form",
            "source": "corporate"
        })
        
        # Legal form comparison instruction
        instructions.append({
            "instruction": f"Compare {company.get('emri', 'this company')} with other companies of the same legal form.",
            "response": f"To compare {company.get('emri', 'the company')} with other companies of the same legal form ({company.get('forma_ligjore')}), you would filter the database by legal form and analyze similarities in capital, location, and business activities.",
            "category": "comparison_analysis",
            "source": "corporate"
        })
    
    # Capital questions
    if company.get('kapitali') and company.get('kapitali') != "Unknown":
        instructions.append({
            "instruction": f"What is the capital of {company.get('emri', 'this company')}?",
            "response": f"The capital of {company.get('emri', 'the company')} is {company.get('kapitali')}.",
            "category": "capital",
            "source": "corporate"
        })
        
        # Capital range instruction
        instructions.append({
            "instruction": f"Find companies with capital similar to {company.get('emri', 'this company')}.",
            "response": f"To find companies with similar capital to {company.get('emri', 'the company')} ({company.get('kapitali')}), you would search the database for companies within a similar capital range or category.",
            "category": "range_query",
            "source": "corporate"
        })
    
    # Contact information questions
    if company.get('telefoni') and company.get('telefoni') != "Unknown":
        instructions.append({
            "instruction": f"What is the phone number of {company.get('emri', 'this company')}?",
            "response": f"The phone number of {company.get('emri', 'the company')} is {company.get('telefoni')}.",
            "category": "contact",
            "source": "corporate"
        })
    
    if company.get('email') and company.get('email') != "Unknown":
        instructions.append({
            "instruction": f"What is the email address of {company.get('emri', 'this company')}?",
            "response": f"The email address of {company.get('emri', 'the company')} is {company.get('email')}.",
            "category": "contact",
            "source": "corporate"
        })
        
        # Contact verification instruction
        instructions.append({
            "instruction": f"Verify the contact information for {company.get('emri', 'this company')}.",
            "response": f"To verify contact information for {company.get('emri', 'the company')}, check the phone number: {company.get('telefoni', 'Not available')}, email: {company.get('email', 'Not available')}, and address: {company.get('adresa', 'Not available')}.",
            "category": "verification",
            "source": "corporate"
        })
    
    # Address questions
    if company.get('adresa') and company.get('adresa') != "Unknown":
        instructions.append({
            "instruction": f"What is the address of {company.get('emri', 'this company')}?",
            "response": f"The address of {company.get('emri', 'the company')} is {company.get('adresa')}.",
            "category": "address",
            "source": "corporate"
        })
        
        # Geographic analysis instruction
        instructions.append({
            "instruction": f"Analyze the geographic distribution of companies in the same area as {company.get('emri', 'this company')}.",
            "response": f"To analyze geographic distribution around {company.get('emri', 'the company')} located at {company.get('adresa', 'Unknown address')}, you would search for companies in the same district or area and analyze their business activities and legal forms.",
            "category": "geographic_analysis",
            "source": "corporate"
        })
    
    # Business activity questions
    if company.get('objekti_veprimtarisë') and company.get('objekti_veprimtarisë') != "Unknown":
        instructions.append({
            "instruction": f"What is the business activity of {company.get('emri', 'this company')}?",
            "response": f"The business activity of {company.get('emri', 'the company')} is: {company.get('objekti_veprimtarisë')}.",
            "category": "business_activity",
            "source": "corporate"
        })
        
        # Industry analysis instruction
        instructions.append({
            "instruction": f"Find companies in the same industry as {company.get('emri', 'this company')}.",
            "response": f"To find companies in the same industry as {company.get('emri', 'the company')} which operates in {company.get('objekti_veprimtarisë', 'Unknown industry')}, you would search the database for companies with similar business activity descriptions.",
            "category": "industry_search",
            "source": "corporate"
        })
        
        # Business activity classification instruction
        instructions.append({
            "instruction": f"Classify the business activity of {company.get('emri', 'this company')} into a standard industry category.",
            "response": f"Based on the business activity '{company.get('objekti_veprimtarisë', 'Unknown')}', {company.get('emri', 'the company')} would be classified into the appropriate industry category based on the nature of their operations.",
            "category": "classification",
            "source": "corporate"
        })
    
    # Administrator questions
    if company.get('administratorët') and company.get('administratorët'):
        admin_list = ", ".join(company.get('administratorët', []))
        instructions.append({
            "instruction": f"Who are the administrators of {company.get('emri', 'this company')}?",
            "response": f"The administrators of {company.get('emri', 'the company')} are: {admin_list}.",
            "category": "management",
            "source": "corporate"
        })
        
        # Management network instruction
        instructions.append({
            "instruction": f"Find other companies managed by the same administrators as {company.get('emri', 'this company')}.",
            "response": f"To find other companies managed by the same administrators ({admin_list}) as {company.get('emri', 'the company')}, you would search the database for companies with matching administrator names.",
            "category": "network_analysis",
            "source": "corporate"
        })
    
    # Shareholder questions
    if company.get('zotëruesit') and company.get('zotëruesit'):
        shareholder_list = ", ".join(company.get('zotëruesit', []))
        instructions.append({
            "instruction": f"Who are the shareholders of {company.get('emri', 'this company')}?",
            "response": f"The shareholders of {company.get('emri', 'the company')} are: {shareholder_list}.",
            "category": "ownership",
            "source": "corporate"
        })
        
        # Ownership network instruction
        instructions.append({
            "instruction": f"Analyze the ownership network for {company.get('emri', 'this company')}.",
            "response": f"To analyze the ownership network for {company.get('emri', 'the company')}, you would examine the shareholders ({shareholder_list}) and potentially find other companies where these same individuals or entities are shareholders.",
            "category": "network_analysis",
            "source": "corporate"
        })
    
    # Financial information questions
    if company.get('informacione_financiare') and company.get('informacione_financiare'):
        instructions.append({
            "instruction": f"What are the financial details for {company.get('emri', 'this company')}?",
            "response": f"The financial information for {company.get('emri', 'the company')} includes: {json.dumps(company.get('informacione_financiare', {}), ensure_ascii=False)}.",
            "category": "financial_analysis",
            "source": "corporate"
        })
        
        # Financial trend instruction
        instructions.append({
            "instruction": f"Analyze the financial trends for {company.get('emri', 'this company')} over the years.",
            "response": f"To analyze financial trends for {company.get('emri', 'the company')}, you would examine the financial data across different years: {json.dumps(company.get('informacione_financiare', {}), ensure_ascii=False)} and identify patterns in revenue, profit, or other financial metrics.",
            "category": "trend_analysis",
            "source": "corporate"
        })
    
    # Document analysis instructions
    if company.get('dokumenta_financiare') and company.get('dokumenta_financiare'):
        instructions.append({
            "instruction": f"What financial documents are available for {company.get('emri', 'this company')}?",
            "response": f"The following financial documents are available for {company.get('emri', 'the company')}: {company.get('dokumenta_financiare')}.",
            "category": "document_analysis",
            "source": "corporate"
        })
    
    return instructions


def generate_procurement_instructions(tender: Dict) -> List[Dict]:
    """Generate instruction-response pairs from procurement data."""
    instructions = []
    
    # Basic tender information
    if tender.get('tender_id'):
        instructions.append({
            "instruction": f"What is tender ID {tender.get('tender_id')} about?",
            "response": f"Tender ID {tender.get('tender_id')} is about: {tender.get('Titulli i tenderit', 'N/A')}",
            "category": "tender_info",
            "source": "procurement"
        })
        
        # Tender summary instruction
        instructions.append({
            "instruction": f"Provide a comprehensive summary of tender {tender.get('tender_id')}.",
            "response": f"Tender {tender.get('tender_id')} is titled '{tender.get('Titulli i tenderit', 'N/A')}' with a value of {tender.get('Vlera e tenderit', 'N/A')}. It is managed by {tender.get('Autoriteti kontraktues', 'N/A')} and has a submission deadline of {tender.get('Afati i dorëzimit', 'N/A')}. The tender status is {tender.get('Statusi', 'N/A')} and it is a {tender.get('Lloji i tenderit', 'N/A')} using {tender.get('Lloji i procedurës', 'N/A')} procedure.",
            "category": "tender_summary",
            "source": "procurement"
        })
    
    # Tender title questions
    if tender.get('Titulli i tenderit') and tender.get('Titulli i tenderit') != "Unknown":
        instructions.append({
            "instruction": f"What is the title of tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The title of tender {tender.get('tender_id', 'N/A')} is: {tender.get('Titulli i tenderit')}",
            "category": "tender_title",
            "source": "procurement"
        })
        
        # Title-based search instruction
        instructions.append({
            "instruction": f"Find tenders with similar titles to '{tender.get('Titulli i tenderit')}'.",
            "response": f"To find tenders with similar titles to '{tender.get('Titulli i tenderit')}', you would search the procurement database for tenders containing similar keywords or phrases in their titles.",
            "category": "similarity_search",
            "source": "procurement"
        })
    
    # Contracting authority questions
    if tender.get('Autoriteti kontraktues') and tender.get('Autoriteti kontraktues') != "Unknown":
        instructions.append({
            "instruction": f"Who is the contracting authority for tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The contracting authority for tender {tender.get('tender_id', 'N/A')} is {tender.get('Autoriteti kontraktues')}.",
            "category": "contracting_authority",
            "source": "procurement"
        })
        
        # Authority analysis instruction
        instructions.append({
            "instruction": f"Find all tenders issued by {tender.get('Autoriteti kontraktues')}.",
            "response": f"To find all tenders issued by {tender.get('Autoriteti kontraktues')}, you would filter the procurement database by contracting authority field to see all tenders managed by this entity.",
            "category": "authority_analysis",
            "source": "procurement"
        })
        
        # Authority performance instruction
        instructions.append({
            "instruction": f"Analyze the procurement performance of {tender.get('Autoriteti kontraktues')}.",
            "response": f"To analyze the procurement performance of {tender.get('Autoriteti kontraktues')}, you would examine all tenders issued by this authority, their values, success rates, and compliance with procurement regulations.",
            "category": "performance_analysis",
            "source": "procurement"
        })
    
    # Tender value questions
    if tender.get('Vlera e tenderit') and tender.get('Vlera e tenderit') != "Unknown":
        instructions.append({
            "instruction": f"What is the value of tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The value of tender {tender.get('tender_id', 'N/A')} is {tender.get('Vlera e tenderit')}.",
            "category": "tender_value",
            "source": "procurement"
        })
        
        # Value range instruction
        instructions.append({
            "instruction": f"Find tenders with values similar to {tender.get('tender_id', 'N/A')}.",
            "response": f"To find tenders with similar values to {tender.get('Vlera e tenderit')}, you would search the procurement database for tenders within a similar value range or category.",
            "category": "value_range_search",
            "source": "procurement"
        })
        
        # Budget analysis instruction
        instructions.append({
            "instruction": f"Analyze the budget allocation for tenders in the same category as {tender.get('tender_id', 'N/A')}.",
            "response": f"To analyze budget allocation for tenders similar to {tender.get('tender_id', 'N/A')} with value {tender.get('Vlera e tenderit')}, you would examine the distribution of tender values in the same category or sector.",
            "category": "budget_analysis",
            "source": "procurement"
        })
    
    # Submission deadline questions
    if tender.get('Afati i dorëzimit') and tender.get('Afati i dorëzimit') != "Unknown":
        instructions.append({
            "instruction": f"What is the submission deadline for tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The submission deadline for tender {tender.get('tender_id', 'N/A')} is {tender.get('Afati i dorëzimit')}.",
            "category": "deadline",
            "source": "procurement"
        })
        
        # Urgency analysis instruction
        instructions.append({
            "instruction": f"Calculate how many days are left until the deadline for tender {tender.get('tender_id', 'N/A')}.",
            "response": f"To calculate the remaining days until the deadline for tender {tender.get('tender_id', 'N/A')} with deadline {tender.get('Afati i dorëzimit')}, you would subtract the current date from the deadline date.",
            "category": "urgency_calculation",
            "source": "procurement"
        })
        
        # Deadline extension instruction
        instructions.append({
            "instruction": f"Check if tender {tender.get('tender_id', 'N/A')} has had any deadline extensions.",
            "response": f"To check for deadline extensions for tender {tender.get('tender_id', 'N/A')}, you would examine the tender's modification history and compare the original deadline with any updated deadlines.",
            "category": "modification_check",
            "source": "procurement"
        })
    
    # Tender status questions
    if tender.get('Statusi') and tender.get('Statusi') != "Unknown":
        instructions.append({
            "instruction": f"What is the status of tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The status of tender {tender.get('tender_id', 'N/A')} is {tender.get('Statusi')}.",
            "category": "tender_status",
            "source": "procurement"
        })
        
        # Status filter instruction
        instructions.append({
            "instruction": f"Show me all tenders with status '{tender.get('Statusi')}'.",
            "response": f"To find all tenders with status '{tender.get('Statusi')}', you would filter the procurement database by status field. Tender {tender.get('tender_id', 'N/A')} is one example with this status.",
            "category": "status_filter",
            "source": "procurement"
        })
        
        # Status transition instruction
        instructions.append({
            "instruction": f"Track the status changes for tender {tender.get('tender_id', 'N/A')}.",
            "response": f"To track status changes for tender {tender.get('tender_id', 'N/A')}, you would examine the tender's modification history to see how the status has evolved from initial publication to the current status of {tender.get('Statusi')}.",
            "category": "status_tracking",
            "source": "procurement"
        })
    
    # Tender type questions
    if tender.get('Lloji i tenderit') and tender.get('Lloji i tenderit') != "Unknown":
        instructions.append({
            "instruction": f"What type of tender is {tender.get('tender_id', 'N/A')}?",
            "response": f"Tender {tender.get('tender_id', 'N/A')} is a {tender.get('Lloji i tenderit')}.",
            "category": "tender_type",
            "source": "procurement"
        })
        
        # Type comparison instruction
        instructions.append({
            "instruction": f"Compare {tender.get('Lloji i tenderit')} tenders with other tender types.",
            "response": f"To compare {tender.get('Lloji i tenderit')} tenders with other types, you would analyze differences in procedures, requirements, evaluation criteria, and typical values between different tender categories.",
            "category": "type_comparison",
            "source": "procurement"
        })
    
    # Procedure type questions
    if tender.get('Lloji i procedurës') and tender.get('Lloji i procedurës') != "Unknown":
        instructions.append({
            "instruction": f"What is the procedure type for tender {tender.get('tender_id', 'N/A')}?",
            "response": f"The procedure type for tender {tender.get('tender_id', 'N/A')} is {tender.get('Lloji i procedurës')}.",
            "category": "procedure_type",
            "source": "procurement"
        })
        
        # Procedure analysis instruction
        instructions.append({
            "instruction": f"Explain the {tender.get('Lloji i procedurës')} procedure for tender {tender.get('tender_id', 'N/A')}.",
            "response": f"The {tender.get('Lloji i procedurës')} procedure for tender {tender.get('tender_id', 'N/A')} involves specific steps, evaluation criteria, and timeline requirements as defined by Albanian procurement regulations.",
            "category": "procedure_explanation",
            "source": "procurement"
        })
    
    # Additional tender fields
    if tender.get('Kategoria') and tender.get('Kategoria') != "Unknown":
        instructions.append({
            "instruction": f"What category does tender {tender.get('tender_id', 'N/A')} belong to?",
            "response": f"Tender {tender.get('tender_id', 'N/A')} belongs to the category: {tender.get('Kategoria')}.",
            "category": "tender_category",
            "source": "procurement"
        })
        
        # Category analysis instruction
        instructions.append({
            "instruction": f"Analyze tenders in the {tender.get('Kategoria')} category.",
            "response": f"To analyze tenders in the {tender.get('Kategoria')} category, you would examine all tenders in this category to understand typical values, contracting authorities, and success patterns.",
            "category": "category_analysis",
            "source": "procurement"
        })
    
    if tender.get('Vendndodhja') and tender.get('Vendndodhja') != "Unknown":
        instructions.append({
            "instruction": f"Where is tender {tender.get('tender_id', 'N/A')} located?",
            "response": f"Tender {tender.get('tender_id', 'N/A')} is located in {tender.get('Vendndodhja')}.",
            "category": "tender_location",
            "source": "procurement"
        })
        
        # Geographic analysis instruction
        instructions.append({
            "instruction": f"Find tenders in the same location as {tender.get('tender_id', 'N/A')}.",
            "response": f"To find tenders in the same location as {tender.get('tender_id', 'N/A')} ({tender.get('Vendndodhja', 'N/A')}), you would filter the procurement database by location field.",
            "category": "geographic_search",
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


@step
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


@step
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


@step
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