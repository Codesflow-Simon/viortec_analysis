#!/usr/bin/env python3
"""
Excel to CSV Converter

This script converts Excel files (.xlsx, .xls) to CSV format.
It handles multiple sheets and provides options for output formatting.
"""

import pandas as pd
import os
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_excel_to_csv(input_file, output_dir=None, sheet_name=None, encoding='utf-8'):
    """
    Convert an Excel file to CSV format.
    
    Args:
        input_file (str): Path to the input Excel file
        output_dir (str): Directory to save CSV files (defaults to same directory as input)
        sheet_name (str): Specific sheet to convert (None for all sheets)
        encoding (str): Encoding for output CSV files
    
    Returns:
        list: List of created CSV file paths
    """
    try:
        # Read the Excel file
        logger.info(f"Reading Excel file: {input_file}")
        
        if sheet_name:
            # Read specific sheet
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            if isinstance(df, dict):
                df = df[sheet_name]
            sheets = {sheet_name: df}
        else:
            # Read all sheets
            sheets = pd.read_excel(input_file, sheet_name=None)
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(input_file).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(input_file).stem
        
        created_files = []
        
        # Convert each sheet to CSV
        for sheet_name, df in sheets.items():
            if df is not None and not df.empty:
                # Clean sheet name for filename
                safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_sheet_name = safe_sheet_name.replace(' ', '_')
                
                # Create output filename
                if len(sheets) == 1:
                    output_file = output_dir / f"{base_name}.csv"
                else:
                    output_file = output_dir / f"{base_name}_{safe_sheet_name}.csv"
                
                # Convert to CSV
                df.to_csv(output_file, index=False, encoding=encoding)
                logger.info(f"Created CSV file: {output_file}")
                created_files.append(str(output_file))
                
                # Print basic info about the data
                logger.info(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"Columns: {list(df.columns)}")
                
        return created_files
        
    except Exception as e:
        logger.error(f"Error converting {input_file}: {str(e)}")
        raise

def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(description='Convert Excel files to CSV format')
    parser.add_argument('input_file', help='Path to the input Excel file')
    parser.add_argument('-o', '--output-dir', help='Output directory for CSV files')
    parser.add_argument('-s', '--sheet', help='Specific sheet name to convert')
    parser.add_argument('-e', '--encoding', default='utf-8', help='Output encoding (default: utf-8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        # Convert the file
        created_files = convert_excel_to_csv(
            args.input_file,
            args.output_dir,
            args.sheet,
            args.encoding
        )
        
        logger.info(f"Successfully converted {len(created_files)} sheet(s) to CSV")
        for file_path in created_files:
            logger.info(f"  - {file_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
