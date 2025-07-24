"""
Tests for OpenCorporates Albania Web Scraping Pipeline
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup
import requests
from datetime import datetime

from pipelines.open_corporate import (
    get_total_pages,
    scrape_company_listings,
    scrape_company_details,
    combine_all_companies,
    save_to_csv,
    BASE_URL,
    SEARCH_URL,
    HEADERS
)


class TestGetTotalPages:
    @patch('pipelines.open_corporate.requests.get')
    def test_get_total_pages_success(self, mock_get):
        mock_html = '''
        <html>
            <span class="text-xs text-muted font-weight-normal text-normalcase op-7">
                Rezultate gjithsej: 36611
            </span>
        </html>
        '''
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = get_total_pages()
        
        assert result == 3662
        mock_get.assert_called_once_with(f"{SEARCH_URL}?page=1", headers=HEADERS, timeout=30)
    
    @patch('pipelines.open_corporate.requests.get')
    def test_get_total_pages_pagination_fallback(self, mock_get):
        mock_html = '''
        <html>
            <ul class="pagination">
                <a href="?page=1">1</a>
                <a href="?page=2">2</a>
                <a href="?page=100">100</a>
            </ul>
        </html>
        '''
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = get_total_pages()
        
        assert result == 100
        mock_get.assert_called_once()
    
    @patch('pipelines.open_corporate.requests.get')
    def test_get_total_pages_default_fallback(self, mock_get):
        mock_html = '<html><body>No pagination info</body></html>'
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = get_total_pages()
        
        assert result == 100
        mock_get.assert_called_once()
    
    @patch('pipelines.open_corporate.requests.get')
    def test_get_total_pages_request_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection error")
        
        result = get_total_pages()
        
        assert result == 100
        mock_get.assert_called_once()


class TestScrapeCompanyListings:
    @patch('pipelines.open_corporate.requests.get')
    def test_scrape_company_listings_success(self, mock_get):
        mock_html = '''
        <html>
            <div class="card px-3 py-4 mb-3 row-hover pos-relative">
                <div class="row align-items-center">
                    <div class="col-md-10">
                        <h4 class="mb-0">TEST COMPANY 1</h4>
                        <p class="text-muted mb-2 text-sm">
                            <a href="/sq/nipt/l12345678a" class="font-weight-bold text-muted">L12345678A</a>
                            <br/>(Test description)
                        </p>
                        <p class="text-muted mb-2 text-sm">
                            <span><i class="fa fa-map-marker"></i> Tiranë</span>
                            <span><i class="fa fa-money"></i> ALL</span>
                        </p>
                    </div>
                    <div class="col-md-2">
                        <a href="/sq/nipt/l12345678a" class="btn btn-danger text-uppercase font-weight-bold d-lg-block">Më shumë</a>
                    </div>
                </div>
            </div>
        </html>
        '''
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = scrape_company_listings(1)
        
        assert len(result) == 1
        company = result[0]
        assert company["emri"] == "TEST COMPANY 1"
        assert company["nipt"] == "L12345678A"
        assert "Test description" in company["përshkrimi"]
        assert company["vendndodhja"] == "Tiranë"
        assert company["monedha"] == "ALL"
        assert company["url_detaje"] == f"{BASE_URL}/sq/nipt/l12345678a"
        assert company["data_mbledhjes"] is not None
    
    @patch('pipelines.open_corporate.requests.get')
    def test_scrape_company_listings_multiple_companies(self, mock_get):
        mock_html = '''
        <html>
            <div class="card px-3 py-4 mb-3 row-hover pos-relative">
                <div class="row align-items-center">
                    <div class="col-md-10">
                        <h4 class="mb-0">COMPANY 1</h4>
                        <p class="text-muted mb-2 text-sm">
                            <a href="/sq/nipt/l11111111a" class="font-weight-bold text-muted">L11111111A</a>
                        </p>
                    </div>
                    <div class="col-md-2">
                        <a href="/sq/nipt/l11111111a" class="btn btn-danger text-uppercase font-weight-bold d-lg-block">Më shumë</a>
                    </div>
                </div>
            </div>
            <div class="card px-3 py-4 mb-3 row-hover pos-relative">
                <div class="row align-items-center">
                    <div class="col-md-10">
                        <h4 class="mb-0">COMPANY 2</h4>
                        <p class="text-muted mb-2 text-sm">
                            <a href="/sq/nipt/l22222222b" class="font-weight-bold text-muted">L22222222B</a>
                        </p>
                    </div>
                    <div class="col-md-2">
                        <a href="/sq/nipt/l22222222b" class="btn btn-danger text-uppercase font-weight-bold d-lg-block">Më shumë</a>
                    </div>
                </div>
            </div>
        </html>
        '''
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = scrape_company_listings(1)
        
        assert len(result) == 2
        assert result[0]["emri"] == "COMPANY 1"
        assert result[0]["nipt"] == "L11111111A"
        assert result[1]["emri"] == "COMPANY 2"
        assert result[1]["nipt"] == "L22222222B"
    
    @patch('pipelines.open_corporate.requests.get')
    def test_scrape_company_listings_no_companies(self, mock_get):
        mock_html = '<html><body>No company cards found</body></html>'
        
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = scrape_company_listings(1)
        
        assert len(result) == 0
    
    @patch('pipelines.open_corporate.requests.get')
    def test_scrape_company_listings_request_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection error")
        
        result = scrape_company_listings(1)
        
        assert len(result) == 0


class TestScrapeCompanyDetails:
    def test_scrape_company_details_basic_info(self):
        company_summary = {
            "emri": "TEST COMPANY",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        with patch('pipelines.open_corporate.requests.get') as mock_get:
            mock_html = '''
            <html>
                <body>
                    <table class="rwd-table">
                        <tr>
                            <th>Viti i Themelimit:</th>
                            <td>2020-01-01</td>
                        </tr>
                        <tr>
                            <th>Statusi:</th>
                            <td>Aktiv</td>
                        </tr>
                        <tr>
                            <th>Adresa:</th>
                            <td>Rruga Test, Tiranë</td>
                        </tr>
                    </table>
                </body>
            </html>
            '''
            
            mock_response = Mock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = scrape_company_details(company_summary)
            
            assert result["emri"] == "TEST COMPANY"
            assert result["nipt"] == "L12345678A"
            assert result["data_regjistrimit"] == "2020-01-01"
            assert result["statusi"] == "Aktiv"
            assert result["adresa"] == "Rruga Test, Tiranë"
    
    def test_scrape_company_details_with_directors(self):
        company_summary = {
            "emri": "TEST COMPANY",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        with patch('pipelines.open_corporate.requests.get') as mock_get:
            mock_html = '''
            <html>
                <body>
                    <table class="rwd-table">
                        <tr>
                            <th>Administrator: </th>
                            <td>
                                <a href="/person/1">John Doe</a>
                                <a href="/person/2">Jane Smith</a>
                            </td>
                        </tr>
                    </table>
                </body>
            </html>
            '''
            
            mock_response = Mock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = scrape_company_details(company_summary)
            
            assert "John Doe" in result["administratorët"]
            assert "Jane Smith" in result["administratorët"]
            assert len(result["administratorët"]) == 2
    
    def test_scrape_company_details_with_shareholders(self):
        company_summary = {
            "emri": "TEST COMPANY",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        with patch('pipelines.open_corporate.requests.get') as mock_get:
            mock_html = '''
            <html>
                <body>
                    <div class="sidebar-right">
                        <h3>Zotërues të Shoqërisë</h3>
                        <ul class="list-group">
                            <li class="list-group-item">
                                <a href="/person/1">John Doe - 60%</a>
                            </li>
                            <li class="list-group-item">
                                <a href="/person/2">Jane Smith - 40%</a>
                            </li>
                        </ul>
                    </div>
                </body>
            </html>
            '''
            
            mock_response = Mock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = scrape_company_details(company_summary)
            
            assert "John Doe" in result["zotëruesit"]
            assert "Jane Smith" in result["zotëruesit"]
            assert len(result["zotëruesit"]) == 2
    
    def test_scrape_company_details_request_error(self):
        company_summary = {
            "emri": "TEST COMPANY",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        with patch('pipelines.open_corporate.requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection error")
            
            result = scrape_company_details(company_summary)
            
            assert result["emri"] == "TEST COMPANY"
            assert result["nipt"] == "L12345678A"
            assert "data_regjistrimit" not in result


class TestCombineAllCompanies:
    def test_combine_all_companies_success(self):
        companies = [
            {
                "emri": "Company 1",
                "nipt": "L11111111A",
                "përshkrimi": "Description 1",
                "vendndodhja": "Tiranë",
                "monedha": "ALL",
                "url_detaje": f"{BASE_URL}/sq/nipt/l11111111a",
                "data_mbledhjes": datetime.now().isoformat()
            },
            {
                "emri": "Company 2",
                "nipt": "L22222222B",
                "përshkrimi": "Description 2",
                "vendndodhja": "Durrës",
                "monedha": "EUR",
                "url_detaje": f"{BASE_URL}/sq/nipt/l22222222b",
                "data_mbledhjes": datetime.now().isoformat()
            }
        ]
        
        result = combine_all_companies(companies)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result['emri']) == ["Company 1", "Company 2"]
        assert list(result['nipt']) == ["L11111111A", "L22222222B"]
        assert list(result['vendndodhja']) == ["Tiranë", "Durrës"]
    
    def test_combine_all_companies_empty_list(self):
        result = combine_all_companies([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSaveToCSV:
    def test_save_to_csv_success(self, tmp_path):
        df = pd.DataFrame({
            'emri': ['Company 1', 'Company 2'],
            'nipt': ['L11111111A', 'L22222222B'],
            'vendndodhja': ['Tiranë', 'Durrës']
        })
        
        filename = tmp_path / "test_companies.csv"
        result = save_to_csv(df, str(filename))
        
        assert result == str(filename)
        assert filename.exists()
        
        saved_df = pd.read_csv(filename)
        assert len(saved_df) == 2
        assert list(saved_df['emri']) == ['Company 1', 'Company 2']
    
    def test_save_to_csv_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        
        filename = tmp_path / "empty.csv"
        result = save_to_csv(df, str(filename))
        
        assert result == ""
        assert not filename.exists()


class TestLiveDataIntegration:
    @pytest.mark.integration
    def test_live_get_total_pages(self):
        result = get_total_pages()
        
        assert isinstance(result, int)
        assert result > 0
        assert result <= 10000
    
    @pytest.mark.integration
    def test_live_scrape_company_listings(self):
        result = scrape_company_listings(1)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        first_company = result[0]
        assert isinstance(first_company, dict)
        assert first_company["emri"]
        assert first_company["nipt"]
        assert first_company["url_detaje"].startswith(BASE_URL)
    
    @pytest.mark.integration
    def test_live_scrape_company_details(self):
        companies = scrape_company_listings(1)
        if companies:
            company_summary = companies[0]
            
            result = scrape_company_details(company_summary)
            
            assert isinstance(result, dict)
            assert result["emri"] == company_summary["emri"]
            assert result["nipt"] == company_summary["nipt"]
            assert result["url_detaje"] == company_summary["url_detaje"]
    
    @pytest.mark.integration
    def test_live_end_to_end_small_scale(self):
        total_pages = get_total_pages()
        
        companies = scrape_company_listings(1)
        assert len(companies) > 0
        
        company_details = []
        for company in companies[:2]:
            detail = scrape_company_details(company)
            company_details.append(detail)
        
        df = combine_all_companies(company_details)
        assert len(df) == 2
        
        required_columns = [
            'emri', 'nipt', 'përshkrimi', 'vendndodhja', 'monedha',
            'url_detaje', 'data_mbledhjes'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' not found in DataFrame"
        
        optional_columns = [
            'data_regjistrimit', 'statusi', 'forma_ligjore', 'kapitali',
            'adresa', 'telefoni', 'email', 'website', 'administratorët', 'zotëruesit'
        ]
        
        found_optional = sum(1 for col in optional_columns if col in df.columns)
        assert found_optional > 0, "No optional columns found in DataFrame"


class TestDataStructures:
    def test_company_summary_structure(self):
        company = {
            "emri": "Test Company",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        assert company["emri"] == "Test Company"
        assert company["nipt"] == "L12345678A"
        assert company["vendndodhja"] == "Tiranë"
    
    def test_company_detail_structure(self):
        company = {
            "emri": "Test Company",
            "nipt": "L12345678A",
            "përshkrimi": "Test description",
            "vendndodhja": "Tiranë",
            "monedha": "ALL",
            "data_regjistrimit": "2020-01-01",
            "statusi": "Aktiv",
            "adresa": "Test Address",
            "administratorët": ["John Doe", "Jane Smith"],
            "url_detaje": f"{BASE_URL}/sq/nipt/l12345678a",
            "data_mbledhjes": datetime.now().isoformat()
        }
        
        assert company["emri"] == "Test Company"
        assert company["data_regjistrimit"] == "2020-01-01"
        assert company["statusi"] == "Aktiv"
        assert len(company["administratorët"]) == 2
        assert "zotëruesit" not in company


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
