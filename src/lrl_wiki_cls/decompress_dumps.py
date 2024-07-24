import os
import gzip
import bz2
from pathlib import Path
from lrl_wiki_cls.config import DATA_DIR, RESULTS_DIR


def decompress_file(compressed_file_path, decompressed_file_path):
    """
    Decompresses a file. Supports .gz and .bz2 files.
    Adjust this function based on your actual decompression logic.
    """
    if compressed_file_path.endswith('.gz'):
        with gzip.open(compressed_file_path, 'rb') as f_in:
            with open(decompressed_file_path, 'wb') as f_out:
                f_out.write(f_in.read())
    elif compressed_file_path.endswith('.bz2'):
        with bz2.open(compressed_file_path, 'rb') as f_in:
            with open(decompressed_file_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise ValueError("Unsupported file format")


def decompress_all_in_directory(directory, to_directory):
    """
    Decompresses all .gz and .bz2 files in the specified directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.gz') or filename.endswith('.bz2'):
            output_filename = Path(filename).stem

            decompress_file(os.path.join(directory, filename), os.path.join(to_directory, output_filename))

# Example usage: Decompress all dump files in the data/ directory
if __name__ == '__main__':
    decompress_all_in_directory(DATA_DIR, RESULTS_DIR)
