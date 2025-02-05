import numpy as np
import hashlib
import os
import pandas as pd

def text_to_weights(text, layer_sizes=[64, 32, 16], output_dir='weights', format='npy'):
    """
    Convert text into deterministic neural network weights and save to directory
    
    Args:
        text (str): Input text to generate weights from
        layer_sizes (list): List of layer sizes (excluding input layer)
        output_dir (str): Directory path to save the weight files
        format (str): Output format - either 'npy', 'txt', or 'csv'
        
    Returns:
        list: List of weight matrices for each layer
    """
    if format not in ['npy', 'txt', 'csv']:
        raise ValueError("Format must be either 'npy', 'txt', or 'csv'")
    
    # Create a seed from the text using SHA-256
    text_bytes = text.encode('utf-8')
    hash_object = hashlib.sha256(text_bytes)
    seed = int(hash_object.hexdigest(), 16) % (2**32)
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize weights list
    weights = []
    
    # Create format-specific subdirectory
    format_dir = os.path.join(output_dir, format)
    os.makedirs(format_dir, exist_ok=True)
    
    # Generate weights for each layer
    prev_size = 128  # Input layer size
    for i, layer_size in enumerate(layer_sizes):
        # Create weight matrix with Xavier/Glorot initialization
        weight_matrix = np.random.randn(prev_size, layer_size) * np.sqrt(2.0 / (prev_size + layer_size))
        weights.append(weight_matrix)
        
        # Save weight matrix to a file in the format-specific subdirectory
        if format == 'npy':
            output_path = os.path.join(format_dir, f'layer_{i}.npy')
            np.save(output_path, weight_matrix)
        elif format == 'txt':
            output_path = os.path.join(format_dir, f'layer_{i}.txt')
            np.savetxt(output_path, weight_matrix, fmt='%.6f', delimiter=',')
        else:  # csv format
            output_path = os.path.join(format_dir, f'layer_{i}.csv')
            pd.DataFrame(weight_matrix).to_csv(output_path, index=False, float_format='%.6f')
        
        prev_size = layer_size
    
    return weights

def read_text_file(file_path):
    """
    Read text from a file
    
    Args:
        file_path (str): Path to the input text file
        
    Returns:
        str: Contents of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate neural network weights from text')
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('--format', choices=['npy', 'txt', 'csv'], default='npy', 
                        help='Output format (npy, txt, or csv)')
    parser.add_argument('--output-dir', default='weights', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        input_text = read_text_file(args.input_file)
        weights = text_to_weights(input_text, output_dir=args.output_dir, format=args.format)
        
        # Print information about the generated weights
        print(f"Generated weights from file: '{args.input_file}'")
        print(f"Text length: {len(input_text)} characters")
        print(f"Weights saved to directory: '{os.path.abspath(os.path.join(args.output_dir, args.format))}'")
        print(f"Output format: {args.format}")
        for i, w in enumerate(weights):
            print(f"Layer {i+1} weights shape: {w.shape}")
            print(f"Layer {i+1} weights mean: {w.mean():.4f}")
            print(f"Layer {i+1} weights std: {w.std():.4f}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 