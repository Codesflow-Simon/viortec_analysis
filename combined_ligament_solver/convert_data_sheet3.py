#!/usr/bin/env python3
"""
Simple script to convert Data Sheet 3.XLSX to CSV format.
This script uses the excel_to_csv_converter module.
"""

import sys
import os

# Add src directory to path so we can import our converter
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from excel_to_csv_converter import convert_excel_to_csv

def main():
    # Path to your Excel file
    input_file = "data/raw/Data Sheet 3.XLSX"
    
    # Output directory (will create if it doesn't exist)
    output_dir = "data/processed"
    
    print(f"Converting {input_file} to CSV...")
    
    try:
        # Convert the file
        created_files = convert_excel_to_csv(
            input_file=input_file,
            output_dir=output_dir
        )
        
        print(f"\nSuccess! Created {len(created_files)} CSV file(s):")
        for file_path in created_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
