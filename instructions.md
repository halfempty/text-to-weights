# Text to Neural Network Weights - Developer Guide

## Program Architecture

This program converts text into deterministic neural network weights using a hash-based seeding mechanism. Here's a detailed breakdown of how it works.

## Core Components

### 1. Main Function: `text_to_weights()`
```python
def text_to_weights(text, layer_sizes=[64, 32, 16], output_dir='weights', format='npy'):
    """
    Core function that converts text to neural network weights.
    - text: Input text to generate weights from
    - layer_sizes: List defining the size of each layer
    - output_dir: Where to save the weights
    - format: 'npy', 'txt', or 'csv'
    """
```

### 2. Weight Generation Process

1. **Text to Seed Conversion**:
```python
# Convert text to a deterministic seed
text_bytes = text.encode('utf-8')
hash_object = hashlib.sha256(text_bytes)
seed = int(hash_object.hexdigest(), 16) % (2**32)
np.random.seed(seed)
```

2. **Weight Matrix Creation**:
```python
# Initialize with input layer size of 128
prev_size = 128
for layer_size in layer_sizes:
    # Xavier/Glorot initialization
    weight_matrix = np.random.randn(prev_size, layer_size) * np.sqrt(2.0 / (prev_size + layer_size))
    prev_size = layer_size
```

### 3. File Output Formats

The program supports three output formats:

1. **NPY Format** (NumPy Binary):
```python
np.save(output_path, weight_matrix)
```

2. **TXT Format** (Plain Text):
```python
np.savetxt(output_path, weight_matrix, fmt='%.6f', delimiter=',')
```

3. **CSV Format** (With Headers):
```python
pd.DataFrame(weight_matrix).to_csv(output_path, index=False, float_format='%.6f')
```

## Implementation Steps

1. **Set Up Project Structure**:
```
project/
├── text_to_weights.py    # Main program
├── test_text_to_weights.py    # Test suite
├── README.md            # Basic documentation
└── instructions.md      # This detailed guide
```

2. **Required Dependencies**:
```python
import numpy as np
import hashlib
import os
import pandas as pd
```

3. **Neural Network Architecture**:
- Input layer: 128 nodes (fixed)
- Hidden layers: Configurable through layer_sizes parameter
- Default architecture: [128 → 64 → 32 → 16]

4. **Weight Initialization**:
- Uses Xavier/Glorot initialization
- Formula: `weight = random_normal() * sqrt(2 / (fan_in + fan_out))`
- Helps prevent vanishing/exploding gradients

## Key Implementation Details

### 1. Deterministic Generation
```python
# Same input text always produces the same weights
hash_object = hashlib.sha256(text.encode('utf-8'))
seed = int(hash_object.hexdigest(), 16) % (2**32)
np.random.seed(seed)
```

### 2. Directory Structure
```python
# Create format-specific subdirectories
format_dir = os.path.join(output_dir, format)
os.makedirs(format_dir, exist_ok=True)
```

### 3. File Naming Convention
```python
# Files are named layer_0, layer_1, etc.
output_path = os.path.join(format_dir, f'layer_{i}.{format}')
```

## Testing Implementation

1. **Test Cases to Include**:
- Output shape verification
- Deterministic output checking
- File format validation
- Error handling
- Directory structure verification

2. **Example Test**:
```python
def test_text_to_weights_deterministic(self):
    """Test if same input produces same weights"""
    text = "Hello world!"
    weights1 = text_to_weights(text)
    weights2 = text_to_weights(text)
    np.testing.assert_array_equal(weights1, weights2)
```

## Command Line Interface

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--format', choices=['npy', 'txt', 'csv'], default='npy')
    parser.add_argument('--output-dir', default='weights')
    args = parser.parse_args()
```

## Error Handling

1. **Input Validation**:
```python
if format not in ['npy', 'txt', 'csv']:
    raise ValueError("Format must be either 'npy', 'txt', or 'csv'")
```

2. **File Operations**:
```python
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find file: {file_path}")
```

## Performance Considerations

1. **Memory Usage**:
- Weights are generated and saved one layer at a time
- Memory usage is proportional to largest layer size

2. **File Size**:
- NPY: Most efficient (binary format)
- TXT/CSV: Larger due to text representation
- Each format stored in separate subdirectory

## Extension Points

1. **Custom Layer Sizes**:
- Modify layer_sizes parameter for different architectures
- Input layer size (128) can be made configurable

2. **Additional Formats**:
- Add new formats by extending the format handling logic
- Implement new save methods as needed

3. **Weight Initialization**:
- Xavier/Glorot can be replaced with other initialization methods
- Add initialization strategy as a parameter

This implementation provides a foundation that can be extended for specific use cases while maintaining the core functionality of deterministic weight generation from text.
