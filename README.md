# OpenCorporates Albania Web Scraping Pipeline

A ZenML-based web scraping pipeline for extracting comprehensive company information from the OpenCorporates Albania website.

## Features

- **Comprehensive Data Extraction**: Scrapes company listings and detailed information
- **Albanian Language Support**: All data fields use Albanian keys
- **Robust Error Handling**: Graceful handling of network errors and missing data
- **ZenML Integration**: Modular pipeline architecture with step-based processing
- **Comprehensive Testing**: Unit tests and live data integration tests

## Data Extracted

### Company Listings
- Company name (`emri`)
- NIPT (tax number) (`nipt`)
- Description (`përshkrimi`)
- Location (`vendndodhja`)
- Currency (`monedha`)
- Detail URL (`url_detaje`)

### Company Details
- Registration date (`data_regjistrimit`)
- Status (`statusi`)
- Legal form (`forma_ligjore`)
- Capital (`kapitali`)
- Address (`adresa`)
- Phone (`telefoni`)
- Email (`email`)
- Website (`website`)
- Directors (`administratorët`)
- Shareholders (`zotëruesit`)
- Business activity (`objekti_veprimtarisë`)
- Financial information (`informacione_financiare`)
- Licenses and permits (`leje_licensa`)
- Historical changes (`ndryshime`)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-training

# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
# Run in test mode (1 page, 2 companies)
python run_scraper.py --test-mode

# Run with custom parameters
python run_scraper.py --max-pages 10 --max-companies-per-page 5
```

### Command Line Options

- `--max-pages`: Maximum number of pages to scrape (default: 5)
- `--max-companies-per-page`: Maximum companies per page (default: all)
- `--test-mode`: Run with minimal data for testing

### Programmatic Usage

```python
from pipelines.open_corporate import opencorporates_scraping_pipeline

# Run the full pipeline
opencorporates_scraping_pipeline()
```

## Project Structure

```
llm-training/
├── pipelines/
│   └── open_corporate.py      # Main pipeline implementation
├── tests/
│   └── test_open_corporate.py # Comprehensive test suite
├── run_scraper.py             # Command-line runner
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Testing

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/ -m "not integration"

# Run only integration tests
pytest tests/ -m integration
```

## Output

The pipeline generates a CSV file with timestamped filename containing all extracted company data.

## Dependencies

- `zenml`: MLOps framework
- `requests`: HTTP requests
- `beautifulsoup4`: HTML parsing
- `pandas`: Data manipulation
- `pytest`: Testing framework

## License

This project is licensed under the MIT License.
