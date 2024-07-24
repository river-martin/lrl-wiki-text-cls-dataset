import unittest
from lrl_wiki_cls.config import DATA_DIR, RESULTS_DIR, MIN_ENTRIES_PER_LABEL
from lrl_wiki_cls.wiki_text_cls_dataset import (
    WikiTextClsDataset,
    WikiTextClsDatasetCreator,
    count_labels,
)
import numpy as np
import xml.etree.ElementTree as ET
from lrl_wiki_cls.config import NAMESPACES, RESULTS_DIR, WIKI_CATEGORY_NAMESPACE
from lrl_wiki_cls.load_wiki_tables import load_table
import os
import pickle


class TestWikiTextClsDataset(unittest.TestCase):
    def setUp(self):
        """Set up a sample dataset for testing."""
        self.dataset_path = f"{DATA_DIR}/sample_data.json"
        self.dataset = WikiTextClsDataset(json_file=self.dataset_path)
        label_counts = count_labels(self.dataset.data)
        # If a category does not contain at least two entries per category label, the probability and entropy calculations will fail.
        for label, count in label_counts.items():
            assert count >= 2

    def test_dataset_initialization(self):
        """Test if the dataset can be initialized properly."""
        self.assertIsInstance(self.dataset, WikiTextClsDataset)

    def test_dataset_length(self):
        """Test if the dataset reports correct length."""
        expected_length = 2
        self.assertEqual(len(self.dataset), expected_length)

    def test_get_item(self):
        """Test retrieving an item from the dataset."""
        entry, cats = self.dataset[0]
        self.assertIn("text", entry.keys())
        self.assertIn("categories", entry.keys())
        self.assertIsInstance(entry["text"], str)
        self.assertIsInstance(entry["categories"], list)
        for i, label in enumerate(entry["categories"]):
            self.assertIsInstance(label, str)
            assert label != ""
            assert label == cats[i]


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


class TestWikiTextClsDatasetCreator(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset creator for testing."""
        self.categorylinks_df = load_table("categorylinks")
        self.page_df = load_table("page")
        self.cat_df = load_table("category")
        self.redirect_df = load_table("redirect")
        self.wiki_pages_articles_xml_path = f"{RESULTS_DIR}/zuwiki-latest-pages-articles.xml"
        self.dataset_creator = WikiTextClsDatasetCreator(
            wiki_pages_articles_xml_path=self.wiki_pages_articles_xml_path,
            categorylinks_df=self.categorylinks_df,
            page_df=self.page_df,
            cat_df=self.cat_df,
            redirect_df=self.redirect_df,
        )

    def test_add_parent_categories(self):
        redirect_df = load_table("redirect")
        page_df = load_table("page")
        categorylinks_df = load_table("categorylinks")
        cats = ["Amazwe_e-Afrika", "Amazwe_eYurophu"]
        updated_cats = self.dataset_creator.add_parent_categories(
            cats, redirect_df, page_df, categorylinks_df
        )
        assert "Amazwe" in updated_cats

    def test_dataset_creator_initialization(self):
        """Test if the dataset creator can be initialized properly."""
        self.assertIsInstance(self.dataset_creator, WikiTextClsDatasetCreator)

    def test_create_article_id_categories_map(self):
        preprocessed_articles = self.dataset_creator.read_and_preprocess_articles(
            self.wiki_pages_articles_xml_path
        )
        article_id_category_map = self.dataset_creator.create_article_id_categories_map(
            preprocessed_articles, self.categorylinks_df, self.cat_df
        )
        self.assertIsInstance(article_id_category_map, dict)
        self.assertGreater(len(article_id_category_map), 0)
        self.assertTrue(all(isinstance(page_id, int) for page_id in article_id_category_map.keys()))
        self.assertTrue(
            all(isinstance(categories, list) for categories in article_id_category_map.values())
        )
        for article_id, categories in article_id_category_map.items():
            self.assertTrue(isinstance(article_id, int))
            cat_counts = {}
            # Ensure that we do not have duplicate labels.
            for category in categories:
                self.assertIsInstance(category, str)
                assert category != ""
                cat_counts[category] = cat_counts.get(category, 0) + 1
            self.assertTrue(all(count == 1 for count in cat_counts.values()))

    def test_create_dataset_list(self):
        dataset = self.dataset_creator.create_dataset_list()
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        self.assertTrue(all(isinstance(sample, dict) for sample in dataset))
        self.assertTrue(all("text" in sample for sample in dataset))
        self.assertTrue(all("categories" in sample for sample in dataset))
        self.assertTrue(all("page_id" in sample for sample in dataset))
        self.assertTrue(all(isinstance(sample["text"], str) for sample in dataset))
        self.assertTrue(all(isinstance(sample["categories"], list) for sample in dataset))
        for sample in dataset:
            self.assertTrue(len(sample["categories"]) > 0)
            for label in sample["categories"]:
                self.assertIsInstance(label, str)
                assert label != ""

    def check_if_filtered(self, dataset_list, min_entries_per_label):
        assert isinstance(dataset_list, list)
        label_counts = count_labels(dataset_list)
        return all(count >= min_entries_per_label for count in label_counts.values())

    def test_filter_dataset_list(self):
        dataset = self.dataset_creator.create_dataset_list()
        filtered_dataset_list = self.dataset_creator.filter_dataset_list(
            dataset, self.categorylinks_df, min_entries_per_label=MIN_ENTRIES_PER_LABEL
        )
        assert isinstance(filtered_dataset_list, list)
        del dataset
        self.assertGreater(len(filtered_dataset_list), 0)
        self.assertTrue(all(isinstance(sample, dict) for sample in filtered_dataset_list))
        self.assertTrue(all("text" in sample for sample in filtered_dataset_list))
        self.assertTrue(all("categories" in sample for sample in filtered_dataset_list))
        self.assertTrue(all(isinstance(sample["text"], str) for sample in filtered_dataset_list))
        self.assertTrue(
            all(isinstance(sample["categories"], list) for sample in filtered_dataset_list)
        )
        self.assertTrue(self.check_if_filtered(filtered_dataset_list, MIN_ENTRIES_PER_LABEL))

    def test_is_redirect(self):
        assert self.dataset_creator.is_redirect(
            "Amazwe_eAfrika", self.dataset_creator.redirect_df, self.dataset_creator.page_df
        )
        assert not self.dataset_creator.is_redirect(
            "Amazwe_e-Afrika", self.dataset_creator.redirect_df, self.dataset_creator.page_df
        )

    def test_replace_redirects(self):
        category_titles = set(["Amazwe_eAfrika"])
        updated_category_titles = self.dataset_creator.replace_redirects(
            category_titles, self.dataset_creator.redirect_df, self.dataset_creator.page_df
        )
        assert isinstance(updated_category_titles, set)
        assert updated_category_titles == set(["Amazwe_e-Afrika"])
        category_titles = set(["Amazwe_eAfrika", "Amazwe_eYurophu"])
        updated_category_titles = self.dataset_creator.replace_redirects(
            category_titles, self.dataset_creator.redirect_df, self.dataset_creator.page_df
        )
        assert isinstance(updated_category_titles, set)
        assert updated_category_titles == set(["Amazwe_e-Afrika", "Amazwe_eYurophu"])

    def test_clean_article_text(self):
        # Ensure that the clean_article_text function removes file and image attachments from the wikitext
        article_text = "This is a sample article text with [[File:Sample.png|thumb|170px]] and [[Image:Sample.jpg|thumb|180px]]."
        cleaned_text = self.dataset_creator.clean_article_text(article_text)
        self.assertNotIn("File:", cleaned_text)
        self.assertNotIn("Image:", cleaned_text)
        self.assertNotIn("thumb", cleaned_text)
        self.assertNotIn("170px", cleaned_text)
        self.assertNotIn("180px", cleaned_text)

    def test_read_and_preprocess_articles(self):
        n_articles = 0
        for article_id, article in self.dataset_creator.read_and_preprocess_articles(
            self.wiki_pages_articles_xml_path
        ).items():
            # Only articles should be returned, not categories or files.
            self.assertFalse(article["title"].startswith("File:"))
            self.assertFalse(article["title"].startswith("Image:"))
            self.assertFalse(article["text"].startswith("Category:"))

            self.assertTrue(isinstance(article_id, int))
            self.assertIsInstance(article, dict)
            self.assertIn("title", article)
            self.assertIn("text", article)
            self.assertIsInstance(article["title"], str)
            self.assertIsInstance(article["text"], str)
            self.assertNotEqual(article["title"], "")
            self.assertNotEqual(article["text"], "")
            self.assertNotIn("<b>", article["text"])
            n_articles += 1
        self.assertGreater(n_articles, 0)

    def test_create_dataset_json(self):
        dataset_json_file_path = f"{RESULTS_DIR}/zuwiki-dataset.json"
        filter_based_on_label_counts = True
        dataset = self.dataset_creator.create_dataset_json(
            dataset_json_file_path, filter_based_on_label_counts=filter_based_on_label_counts
        )
        self.assertIsInstance(dataset, str)
        self.assertNotEqual(dataset, "")
        self.assertIn("text", dataset)
        self.assertIn("categories", dataset)
        self.assertIn("page_id", dataset)
        with open(dataset_json_file_path, "r") as f:
            dataset = f.read()
        assert dataset != ""
        dataset = WikiTextClsDataset(json_file=dataset_json_file_path)
        if filter_based_on_label_counts:
            is_filtered = self.check_if_filtered(dataset.data, MIN_ENTRIES_PER_LABEL)
            self.assertTrue(is_filtered)

        # We want to make sure that all the page_ids in the dataset are in the page_df (to ensure they are valid, and align with the categorylinks_df ids)
        valid_page_ids = set()
        for sample, label in dataset:
            valid_page_ids.add(sample["page_id"])
        page_df = self.page_df
        size_before = len(page_df)
        page_df = page_df[page_df["page_id"].isin(valid_page_ids)].query(
            "page_namespace == @WIKI_CATEGORY_NAMESPACE"
        )
        size_after = len(page_df)
        print(
            f"Filtered out {size_before - size_after} pages from page_df that are not in the dataset."
        )


if __name__ == "__main__":
    unittest.main()
