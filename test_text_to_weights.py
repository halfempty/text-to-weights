import unittest
import numpy as np
import os
import tempfile
import shutil
import pandas as pd
from text_to_weights import text_to_weights, read_text_file

class TestTextToWeights(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files and directories
        shutil.rmtree(self.test_dir)
    
    def test_text_to_weights_output_shape(self):
        """Test if the output weights have the correct shapes"""
        text = "Test text"
        layer_sizes = [32, 16, 8]
        weights = text_to_weights(text, layer_sizes, output_dir=os.path.join(self.test_dir, 'weights'))
        
        # Check number of layers
        self.assertEqual(len(weights), len(layer_sizes))
        
        # Check shapes of weight matrices
        prev_size = 128  # Input layer size
        for i, layer_size in enumerate(layer_sizes):
            self.assertEqual(weights[i].shape, (prev_size, layer_size))
            prev_size = layer_size
    
    def test_text_to_weights_deterministic(self):
        """Test if the same input text produces the same weights"""
        text = "Hello world!"
        output_dir1 = os.path.join(self.test_dir, 'weights1')
        output_dir2 = os.path.join(self.test_dir, 'weights2')
        
        weights1 = text_to_weights(text, output_dir=output_dir1)
        weights2 = text_to_weights(text, output_dir=output_dir2)
        
        for w1, w2 in zip(weights1, weights2):
            np.testing.assert_array_equal(w1, w2)
    
    def test_text_to_weights_file_output_npy(self):
        """Test if weights are correctly saved to directory in NPY format"""
        output_dir = os.path.join(self.test_dir, 'weights')
        text = "Test text"
        weights = text_to_weights(text, output_dir=output_dir, format='npy')
        
        # Check if format directory exists
        format_dir = os.path.join(output_dir, 'npy')
        self.assertTrue(os.path.exists(format_dir))
        
        # Load and verify weights
        for i, original_weight in enumerate(weights):
            loaded_weight = np.load(os.path.join(format_dir, f'layer_{i}.npy'))
            np.testing.assert_array_equal(loaded_weight, original_weight)
    
    def test_text_to_weights_file_output_txt(self):
        """Test if weights are correctly saved to directory in TXT format"""
        output_dir = os.path.join(self.test_dir, 'weights')
        text = "Test text"
        weights = text_to_weights(text, output_dir=output_dir, format='txt')
        
        # Check if format directory exists
        format_dir = os.path.join(output_dir, 'txt')
        self.assertTrue(os.path.exists(format_dir))
        
        # Load and verify weights
        for i, original_weight in enumerate(weights):
            loaded_weight = np.loadtxt(os.path.join(format_dir, f'layer_{i}.txt'), delimiter=',')
            np.testing.assert_array_almost_equal(loaded_weight, original_weight, decimal=6)
    
    def test_text_to_weights_file_output_csv(self):
        """Test if weights are correctly saved to directory in CSV format"""
        output_dir = os.path.join(self.test_dir, 'weights')
        text = "Test text"
        weights = text_to_weights(text, output_dir=output_dir, format='csv')
        
        # Check if format directory exists
        format_dir = os.path.join(output_dir, 'csv')
        self.assertTrue(os.path.exists(format_dir))
        
        # Load and verify weights
        for i, original_weight in enumerate(weights):
            loaded_weight = pd.read_csv(os.path.join(format_dir, f'layer_{i}.csv')).values
            np.testing.assert_array_almost_equal(loaded_weight, original_weight, decimal=6)
    
    def test_invalid_format(self):
        """Test if invalid format raises ValueError"""
        text = "Test text"
        output_dir = os.path.join(self.test_dir, 'weights')
        
        with self.assertRaises(ValueError):
            text_to_weights(text, output_dir=output_dir, format='invalid')
    
    def test_read_text_file(self):
        """Test reading text from file"""
        test_text = "Hello, this is a test!"
        test_file = os.path.join(self.test_dir, "test.txt")
        
        # Create test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
        
        # Test reading
        read_text = read_text_file(test_file)
        self.assertEqual(read_text, test_text)
    
    def test_read_text_file_nonexistent(self):
        """Test reading from non-existent file"""
        with self.assertRaises(FileNotFoundError):
            read_text_file("nonexistent_file.txt")
    
    def test_different_texts_different_weights(self):
        """Test if different input texts produce different weights"""
        text1 = "Hello world!"
        text2 = "Different text"
        output_dir1 = os.path.join(self.test_dir, 'weights1')
        output_dir2 = os.path.join(self.test_dir, 'weights2')
        
        weights1 = text_to_weights(text1, output_dir=output_dir1)
        weights2 = text_to_weights(text2, output_dir=output_dir2)
        
        for w1, w2 in zip(weights1, weights2):
            self.assertFalse(np.array_equal(w1, w2))
    
    def test_custom_layer_sizes(self):
        """Test if custom layer sizes are respected"""
        text = "Test text"
        custom_layers = [100, 50, 25]
        output_dir = os.path.join(self.test_dir, 'weights')
        weights = text_to_weights(text, layer_sizes=custom_layers, output_dir=output_dir)
        
        prev_size = 128
        for i, layer_size in enumerate(custom_layers):
            self.assertEqual(weights[i].shape, (prev_size, layer_size))
            prev_size = layer_size
    
    def test_format_subdirectories(self):
        """Test if format-specific subdirectories are created correctly"""
        text = "Test text"
        output_dir = os.path.join(self.test_dir, 'weights')
        
        # Test each format
        for format_type in ['npy', 'txt', 'csv']:
            text_to_weights(text, output_dir=output_dir, format=format_type)
            format_dir = os.path.join(output_dir, format_type)
            self.assertTrue(os.path.exists(format_dir))
            self.assertTrue(os.path.isdir(format_dir))

if __name__ == '__main__':
    unittest.main() 