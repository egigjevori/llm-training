#!/usr/bin/env python3
"""
Specific test for company K31330047F - 4 VELLEZERIT NELA (ish "MARIO")
"""

import sys
import json
import re
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.open_corporate import scrape_company_details


def normalize_text(text):
    """Normalize text by removing extra whitespace and line breaks."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())


def test_company_k31330047f():
    """Test scraping the specific company K31330047F."""
    
    # Create company summary for the specific company
    company_summary = {
        'emri': '4 VELLEZERIT NELA (ish "MARIO")',
        'nipt': 'K31330047F',
        'përshkrimi': 'Test description',
        'vendndodhja': 'Tiranë',
        'monedha': 'ALL',
        'url_detaje': 'https://opencorporates.al/sq/nipt/k31330047f',
        'data_mbledhjes': '2025-07-22T16:30:00.000000'
    }

    print("Testing company K31330047F...")
    
    # Scrape the details
    result = scrape_company_details(company_summary)
    
    # Print the actual scraped data first
    print("\n📄 ACTUAL SCRAPED DATA:")
    print("=" * 50)
    print(f"Company Name: {result['emri']}")
    print(f"NIPT: {result['nipt']}")
    print(f"Administrators: {result.get('administratorët', [])}")
    print(f"Shareholders: {result.get('zotëruesit', [])}")
    print(f"Status: {result.get('statusi', 'Not found')}")
    print(f"Legal Form: {result.get('forma_ligjore', 'Not found')}")
    print(f"Registration Date: {result.get('data_regjistrimit', 'Not found')}")
    print(f"District: {result.get('rrethi', 'Not found')}")
    print(f"Capital: {result.get('kapitali', 'Not found')}")
    print(f"Business Activity: {result.get('objekti_veprimtarisë', 'Not found')}")
    print(f"Other Trade Names: {result.get('emërtime_tjera_tregtare', 'Not found')}")
    print(f"Licenses: {result.get('leje_licensa', 'Not found')}")
    
    # Expected data based on the actual scraped data
    expected_data = {
        'emri': '4 VELLEZERIT NELA (ish "MARIO")',
        'nipt': 'K31330047F',
        'administratorët': ['Avni Nela'],
        'zotëruesit': ['Avni Nela - 25%', 'Niko Nela - 25%', 'Rasim Nela - 25%', 'Irakli Nela - 25%'],
        'statusi': 'Aktiv',
        'forma_ligjore': 'Shoqëri me Përgjegjësi të Kufizuar SH.P.K',
        'data_regjistrimit': '11/04/2000',
        'rrethi': 'Tiranë',
        'kapitali': '100 000,00',
        'objekti_veprimtarisë': 'Prodhim dhe tregtim mielli dhe import export te mallrave ushqimore dhe industrial. Veprimtari ne fushen e nedrtimit dhe mulli bluarje drithrash.Furre Buke, prodhime brumi. Prodhim e tregtim embelsirash. Transport mallrash per nevojat e veta.',
        'emërtime_tjera_tregtare': '4 VELLEZERIT NELA',
        'leje_licensa': 'III.1.A - Leje Mjedisi e tipit CII.1.A - Prodhim, përpunim, shpërndarje me shumicë e ushqimeve'
    }
    
    # Assert the expected data
    assert result['emri'] == expected_data['emri'], f"Company name mismatch: expected {expected_data['emri']}, got {result['emri']}"
    assert result['nipt'] == expected_data['nipt'], f"NIPT mismatch: expected {expected_data['nipt']}, got {result['nipt']}"
    assert result['administratorët'] == expected_data['administratorët'], f"Administrators mismatch: expected {expected_data['administratorët']}, got {result['administratorët']}"
    assert result['zotëruesit'] == expected_data['zotëruesit'], f"Shareholders mismatch: expected {expected_data['zotëruesit']}, got {result['zotëruesit']}"
    assert result['statusi'] == expected_data['statusi'], f"Status mismatch: expected {expected_data['statusi']}, got {result['statusi']}"
    assert result['forma_ligjore'] == expected_data['forma_ligjore'], f"Legal form mismatch: expected {expected_data['forma_ligjore']}, got {result['forma_ligjore']}"
    assert result['data_regjistrimit'] == expected_data['data_regjistrimit'], f"Registration date mismatch: expected {expected_data['data_regjistrimit']}, got {result['data_regjistrimit']}"
    assert result['rrethi'] == expected_data['rrethi'], f"District mismatch: expected {expected_data['rrethi']}, got {result['rrethi']}"
    assert result['kapitali'] == expected_data['kapitali'], f"Capital mismatch: expected {expected_data['kapitali']}, got {result['kapitali']}"
    
    # Normalize business activity text for comparison
    expected_objekti = normalize_text(expected_data['objekti_veprimtarisë'])
    actual_objekti = normalize_text(result['objekti_veprimtarisë'])
    assert actual_objekti == expected_objekti, f"Business activity mismatch: expected {expected_objekti}, got {actual_objekti}"
    
    assert result['emërtime_tjera_tregtare'] == expected_data['emërtime_tjera_tregtare'], f"Other trade names mismatch: expected {expected_data['emërtime_tjera_tregtare']}, got {result['emërtime_tjera_tregtare']}"
    
    # Normalize license text for comparison
    expected_license = normalize_text(expected_data['leje_licensa'])
    actual_license = normalize_text(result['leje_licensa'])
    assert actual_license == expected_license, f"Licenses mismatch: expected {expected_license}, got {actual_license}"
    
    print("\n✅ All assertions passed! Company K31330047F data extracted correctly.")
    
    # Print the full JSON result
    print("\n📄 Full JSON Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


if __name__ == "__main__":
    test_company_k31330047f() 