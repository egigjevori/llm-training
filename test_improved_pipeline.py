#!/usr/bin/env python3
"""
Test script for the Improved Instruction Dataset Pipeline
"""

import logging
from pipelines.instruction_dataset import instruction_dataset_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_improved_pipeline():
    """Test the improved instruction dataset pipeline."""
    
    logger.info("Testing improved instruction dataset pipeline...")
    
    try:
        # Run the pipeline with small limits for testing
        result = instruction_dataset_pipeline(
            corporate_limit=50,
            procurement_limit=50,
            output_path="test_improved_dataset.jsonl"
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Result: {result}")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise


if __name__ == "__main__":
    test_improved_pipeline() 