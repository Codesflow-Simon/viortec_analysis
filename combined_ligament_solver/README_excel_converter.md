# Excel to CSV Converter

This project provides Python scripts to convert Excel files (.xlsx, .xls) to CSV format.

## Files Created

- `src/excel_to_csv_converter.py` - Main converter module with full functionality
- `convert_data_sheet3.py` - Simple script specifically for converting Data Sheet 3.XLSX
- `requirements.txt` - Python dependencies

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Simple Conversion (Recommended for Data Sheet 3.XLSX)

Run the simple script to convert your Excel file:

```bash
python convert_data_sheet3.py
```

This will:
- Convert `data/raw/Data Sheet 3.XLSX` to CSV format
- Save all sheets as separate CSV files in `data/processed/`
- Handle multiple sheets automatically

### Option 2: Full-Featured Converter

Use the main converter with command-line options:

```bash
python src/excel_to_csv_converter.py "data/raw/Data Sheet 3.XLSX"
```

#### Command Line Options

- `-o, --output-dir`: Specify output directory (default: same as input file)
- `-s, --sheet`: Convert only a specific sheet
- `-e, --encoding`: Set output encoding (default: utf-8)
- `-v, --verbose`: Enable verbose logging

#### Examples

```bash
# Convert all sheets to a specific directory
python src/excel_to_csv_converter.py "data/raw/Data Sheet 3.XLSX" -o "data/csv_output"

# Convert only the "Master Summary" sheet
python src/excel_to_csv_converter.py "data/raw/Data Sheet 3.XLSX" -s "Master Summary"

# Convert with verbose logging
python src/excel_to_csv_converter.py "data/raw/Data Sheet 3.XLSX" -v
```

## Output

The conversion creates separate CSV files for each sheet in the Excel file:

- **Master Summary**: Overview data with 51 rows and 30 columns
- **Individual ligament data**: Multiple sheets for different specimens and ligament types (ACL, PCL, MCL, LCL)

Each CSV file is named with the pattern: `Data Sheet 3_[SheetName].csv`

## Data Structure

The Excel file contains:
- **Master Summary**: High-level specimen information including donor details, measurements, and mechanical properties
- **Individual ligament sheets**: Detailed stress-strain data, mechanical testing results, and specimen properties

## Features

- ✅ Handles multiple Excel formats (.xlsx, .xls)
- ✅ Converts all sheets automatically
- ✅ Preserves data structure
- ✅ Handles large files efficiently
- ✅ Provides detailed logging and progress information
- ✅ Command-line interface for automation
- ✅ Error handling and validation

## Troubleshooting

### Common Issues

1. **Memory errors with large files**: The script is optimized for large files but may require sufficient RAM
2. **Encoding issues**: Use the `-e` flag to specify different encodings if needed
3. **Sheet not found**: Use `-s` flag to specify exact sheet names

### Performance

- Large files (like Data Sheet 3.XLSX at 23MB) may take 1-2 minutes to process
- Memory usage scales with file size and number of sheets
- Progress is logged during conversion

## Dependencies

- `pandas`: Data manipulation and Excel reading
- `openpyxl`: Excel .xlsx file support
- `xlrd`: Excel .xls file support

## License

This script is provided as-is for data conversion purposes.








