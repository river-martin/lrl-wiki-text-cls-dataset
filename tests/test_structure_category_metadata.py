from lrl_wiki_cls.structure_category_metadata import CategoryMetadata, create_metadata_json_files
from lrl_wiki_cls.config import RESULTS_DIR
import unittest
import json


class TestCategoryMetadata(unittest.TestCase):
    def setUp(self):
        create_metadata_json_files()
        self.title_to_id_file_path = f"{RESULTS_DIR}/zuwiki-title-to-id.json"
        self.id_to_title_file_path = f"{RESULTS_DIR}/zuwiki-id-to-title.json"
        self.subcat_links_file_path = f"{RESULTS_DIR}/zuwiki-subcat-links.json"
        self.category_metadata = CategoryMetadata(
            self.title_to_id_file_path, self.id_to_title_file_path, self.subcat_links_file_path
        )
        for key, value in self.category_metadata.id_to_title.items():
            assert type(key) == int
            assert type(value) == str
        for key, value in self.category_metadata.title_to_id.items():
            assert type(key) == str
            assert type(value) == int

    def test_get_id(self):
        # Test get_id method
        title = self.category_metadata.get_category_titles()[0]
        id = self.category_metadata.get_id(title)
        assert self.category_metadata.get_title(id) == title

    def test_get_title(self):
        # Test get_title method
        id = self.category_metadata.get_category_ids()[0]
        title = self.category_metadata.get_title(id)
        assert self.category_metadata.get_id(title) == id
        assert isinstance(title, str)

    def test_get_category_ids(self):
        cat_ids = self.category_metadata.get_category_ids()
        assert len(cat_ids) > 0
        assert all(isinstance(cat_id, int) for cat_id in cat_ids)

    def test_get_subcat_links(self):
        subcat_links = self.category_metadata.get_subcat_links()
        assert len(subcat_links) > 0
        assert all(isinstance(link, list) for link in subcat_links)

    def test_sanity(self):
        # Ensure that all the subcategories of a specific category are present
        subcat_links = self.category_metadata.get_subcat_links()
        expected_subcats_found = {"Amazwe_eYurophu": False, "Amazwe_e-Afrika": False}
        for link in subcat_links:
            src_str = self.category_metadata.get_title(link[0])
            dst_str = self.category_metadata.get_title(link[1])
            if src_str == "Amazwe":
                # We should not have duplicate category links
                assert not expected_subcats_found.get(dst_str, False)
                expected_subcats_found[dst_str] = True
        # We expect these to be subcategories of 'Amazwe'
        assert expected_subcats_found["Amazwe_eYurophu"]
        assert expected_subcats_found["Amazwe_e-Afrika"]

    def test_get_subcats_of(self):
        # Test get_subcats_of method
        cat = self.category_metadata.get_category_titles()[0]
        subcats = self.category_metadata.get_subcats_of(cat)
        assert isinstance(subcats, list)

    def test_get_supcats_of(self):
        # Test get_supcats_of method
        cat = self.category_metadata.get_category_titles()[0]
        supcats = self.category_metadata.get_supcats_of(cat)
        assert isinstance(supcats, list)

    def test_get_category_titles(self):
        cat_titles = self.category_metadata.get_category_titles()
        assert len(cat_titles) > 0
        assert all(isinstance(cat_title, str) for cat_title in cat_titles)


if __name__ == "__main__":
    unittest.main()
