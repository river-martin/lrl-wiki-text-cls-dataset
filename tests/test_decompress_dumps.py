import unittest
import os
from lrl_wiki_cls.decompress_dumps import decompress_file

class TestDecompression(unittest.TestCase):
    data_directory = "data/"
    decompressed_directory = "results/"

    def test_decompress_gz_file(self):
        """Test decompression of a .gz file"""
        compressed_file = os.path.join(self.data_directory, "test_file.gz")
        decompressed_file = os.path.join(self.decompressed_directory, "test_file")
        
        # Ensure the decompressed directory exists
        os.makedirs(self.decompressed_directory, exist_ok=True)
        
        # Run the decompression function
        decompress_file(compressed_file, decompressed_file)
        
        # Check if the decompressed file exists
        self.assertTrue(os.path.isfile(decompressed_file), f"Decompressed file {decompressed_file} does not exist")

        # Optionally, check the content of the decompressed file
        with open(decompressed_file, 'r') as f:
            content = f.read()
            self.assertEqual(content, "Test content\n", "Decompressed content does not match expected")

# Add similar tests for .bz2 files or other specific cases as needed.

if __name__ == "__main__":
    unittest.main()
