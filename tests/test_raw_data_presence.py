import unittest
import os

class TestFilePresence(unittest.TestCase):
    data_directory = "data/"  # Assuming a data directory relative to this script's location

    def test_articles_file_presence(self):
        """Check that the articles file exists"""
        expected_file = os.path.join(self.data_directory, "zuwiki-latest-pages-articles.xml.bz2")
        self.assertTrue(os.path.isfile(expected_file), f"{expected_file} does not exist")

    def test_categorylinks_file_presence(self):
        """Check that the category links file exists"""
        expected_file = os.path.join(self.data_directory, "zuwiki-latest-categorylinks.sql.gz")
        self.assertTrue(os.path.isfile(expected_file), f"{expected_file} does not exist")

# Add similar tests for any other specific files you expect to process

if __name__ == "__main__":
    unittest.main()

