# Text to Neural Network Weights

A Python tool that generates deterministic neural network weights from input text. The weights are initialized using Xavier/Glorot initialization and can be saved in multiple formats.

## Installation

### Requirements
- Python 3.x
- NumPy
- Pandas (for CSV output)

Install the required packages:
```bash
pip3 install numpy pandas
```

## Usage

### Basic Usage
```bash
python3 text_to_weights.py <input_text_file> [--format FORMAT] [--output-dir DIR]
```

### Arguments
- `input_text_file`: Path to the text file to generate weights from
- `--format`: Output format (choices: 'npy', 'txt', 'csv', default: 'npy')
- `--output-dir`: Output directory (default: 'weights')

### Output Formats
The program supports three output formats, organized in format-specific subdirectories:

1. **NPY Format** (weights/npy/)
   - Binary NumPy format
   - Most efficient for loading in Python
   - Example: `weights/npy/layer_0.npy`

2. **TXT Format** (weights/txt/)
   - Plain text, comma-separated values
   - Human-readable
   - Example: `weights/txt/layer_0.txt`

3. **CSV Format** (weights/csv/)
   - Comma-separated values with headers
   - Compatible with spreadsheet software
   - Example: `weights/csv/layer_0.csv`

### Examples

Generate weights in NPY format:
```bash
python3 text_to_weights.py input.txt
```

Generate weights in CSV format:
```bash
python3 text_to_weights.py input.txt --format csv
```

Use custom output directory:
```bash
python3 text_to_weights.py input.txt --format txt --output-dir my_weights
```

### Output Structure
The program generates three weight matrices with the following dimensions:
- Layer 1: 128 x 64
- Layer 2: 64 x 32
- Layer 3: 32 x 16

Each layer's weights are initialized using Xavier/Glorot initialization for better neural network training characteristics.

### Loading the Weights

```python
# Load NPY format
import numpy as np
weights = np.load('weights/npy/layer_0.npy')

# Load TXT format
weights = np.loadtxt('weights/txt/layer_0.txt', delimiter=',')

# Load CSV format
import pandas as pd
weights = pd.read_csv('weights/csv/layer_0.csv').values
```

## Features
- Deterministic weight generation (same input text produces same weights)
- Multiple output formats for different use cases
- Xavier/Glorot initialization for better neural network training
- Organized directory structure by format
- Configurable layer sizes

## Testing
Run the test suite:
```bash
python3 -m unittest test_text_to_weights.py -v
```
