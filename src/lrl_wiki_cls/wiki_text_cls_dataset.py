from torch.utils.data import Dataset
import json
import mwparserfromhell
import numpy as np
import xml.etree.ElementTree as ET
from lrl_wiki_cls.config import (
    MIN_ENTRIES_PER_LABEL,
    NAMESPACES,
    RESULTS_DIR,
    WIKI_CATEGORY_NAMESPACE,
)
from lrl_wiki_cls.utils import check_if_is_junk_cat
from tqdm import tqdm


def count_labels(dataset_list):
    label_counts = {}
    for entry in dataset_list:
        for label in entry["categories"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


def get_cat_page_id(page_df, cat_title, redirect_df):
    """
    Returns the cat_id of the a category.
    """
    # Page titles don't contain spaces
    assert not " " in cat_title
    cat_title_bytes = cat_title.encode("utf-8")
    rd_title = None
    page_entry = page_df.query(
        "page_title==@cat_title_bytes and page_namespace==@WIKI_CATEGORY_NAMESPACE"
    )
    if len(page_entry) == 0:
        assert not cat_title == "Amazwe_eAfrika"
        assert not cat_title == "Amaze_eYurophu"
        return None, None
    if page_entry["page_is_redirect"].values[0]:
        redirect_info = redirect_df.query('rd_from==@page_entry["page_id"].values[0]')
        assert len(redirect_info) == 1
        rd_title = redirect_info["rd_title"].values[0].decode("utf-8")
        rd_title_bytes = rd_title.encode("utf-8")
        rd_id = page_df.query(
            "page_title==@rd_title_bytes and page_namespace==@WIKI_CATEGORY_NAMESPACE"
        )["page_id"]
        cat_page_id = rd_id
    else:
        cat_page_id = page_df.query(
            "page_title==@cat_title_bytes and page_namespace==@WIKI_CATEGORY_NAMESPACE"
        )["page_id"]
    return cat_page_id.values[0], rd_title


class WikiTextClsDataset(Dataset):
    """A Dataset for single-label multi-class text classification of Wikipedia articles in low-resource South African languages."""

    def __init__(self, json_file=None, json_dict=None, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - json_file: A string path to the JSON dataset file.
        - transform: Optional transform to be applied on a sample.
        """
        if json_dict:
            self.data = json_dict
        else:
            with open(json_file, "r", encoding="utf-8") as file:
                self.data = json.load(file)

        self.transform = transform
        self.labels = []
        for sample in self.data:
            self.labels.append(sample["categories"])

    def remove_category(self, category_id):
        """
        Remove a category from the dataset.

        Parameters:
        - category_id: The ID of the category to be removed.
        """
        category = self.page_id_title_map[category_id]
        for sample in self.data:
            sample["category_ids"] = [cat for cat in sample["categories"] if cat != category]
        self.data = [sample for sample in self.data if category not in sample["categories"]]

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "title": The title of the page.
                - "text": The text content of the page.
                - "page_id": The page ID of the article.
                - "categories": A list of categories associated with the page.
        """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]


class WikiTextClsDatasetCreator:
    """Processes a Wikipedia dump to produce single-label multi-class text classification
    of Wikipedia articles in low-resource South African languages, by using a label pair encoding algorithm,
    similar to the byte pair encoding tokenization algorithm."""

    def __init__(
        self, wiki_pages_articles_xml_path, categorylinks_df, page_df, cat_df, redirect_df
    ):
        """
        Initialize the dataset creator.

        Parameters:
        - wiki_pages_articles_xml_path: A string path to the Wikipedia pages-articles XML file.
        """
        self.wiki_pages_articles_xml_path = wiki_pages_articles_xml_path
        self.categorylinks_df = categorylinks_df
        self.page_df = page_df
        self.cat_df = cat_df
        self.redirect_df = redirect_df

    def is_redirect(self, cat_title, redirect_df, page_df):
        assert isinstance(cat_title, str)
        cat_page_id, rd_title = get_cat_page_id(page_df, cat_title, redirect_df)
        return rd_title is not None

    def replace_redirects(self, category_titles, redirect_df, page_df):
        assert isinstance(category_titles, set)
        for cat_title in category_titles:
            assert isinstance(cat_title, str)
            cat_page_id, rd_title = get_cat_page_id(page_df, cat_title, redirect_df)
            is_redirect = rd_title is not None
            if is_redirect:
                category_titles.remove(cat_title)
                category_titles.add(rd_title)
        return category_titles

    def add_parent_categories(self, categories, redirect_df, page_df, categorylinks_df):
        # Add parent categories to the list of categories associated with each article.
        assert isinstance(categories[0], str)
        categories = self.replace_redirects(set(categories), redirect_df, page_df)
        new_categories = categories
        processed_cats = set()
        final_category_list = []
        while len(new_categories) > 0:
            category = new_categories.pop()
            if category in processed_cats:
                continue
            cat_page_id, _ = get_cat_page_id(page_df, category, redirect_df)
            if cat_page_id is None:
                # All the categories below should should have pages, and thus category IDs.
                assert not category == "Amazwe_e-Afrika"
                assert not category == "Amazwe_eYurophu"
                assert not category == "Amazwe"
                processed_cats.add(category)
                continue

            parent_categories = categorylinks_df.query("cl_from==@cat_page_id")["cl_to"]
            # Most of our code assumes that the category titles are strings, not bytes.
            parent_categories = [cat.decode("utf-8") for cat in parent_categories]
            parent_categories = set(
                filter(lambda cat: not check_if_is_junk_cat(cat), parent_categories)
            )
            parent_categories = self.replace_redirects(parent_categories, redirect_df, page_df)
            parent_categories = set(
                filter(lambda cat: not check_if_is_junk_cat(cat), parent_categories)
            )
            if category == "Amazwe_e-Afrika":
                assert "Amazwe" in parent_categories
            elif category == "Amazwe_eYurophu":
                assert "Amazwe" in parent_categories
            new_categories = new_categories.union(parent_categories)
            final_category_list.append(category)
            processed_cats.add(category)
        return final_category_list

    def create_article_id_categories_map(self, preprocessed_articles, categorylinks_df, cat_df):
        """
        Load the article IDs and categories from the Wikipedia pages-categories XML file.

        Returns:
        A dictionary mapping article IDs to their categories.
        """
        article_id_categories_map = {}

        def check_if_junk_and_add_to_map(page_id, category):
            # Checks if the category is a junk category and adds it to the map if it is not (junk).
            is_junk_cat = check_if_is_junk_cat(category)
            if is_junk_cat:
                return
            if page_id in article_id_categories_map.keys():
                article_id_categories_map[page_id].append(category)
            else:
                article_id_categories_map[page_id] = [category]

        for index, row in tqdm(categorylinks_df.iterrows(), desc="Creating initial article_id_categories_map", total=len(categorylinks_df)):
            page_id = row["cl_from"]
            cl_type = row["cl_type"]
            is_link_from_missing_article = (
                cl_type.decode("utf-8") != "subcat" and page_id not in preprocessed_articles.keys()
            )
            if is_link_from_missing_article:
                # There is not point in adding categories for articles that don't have text.
                continue
            category = row["cl_to"].decode("utf-8")
            check_if_junk_and_add_to_map(page_id, category)


        # Add parent categories to the list of categories associated with each article.
        for page_id, categories in tqdm(
            article_id_categories_map.items(),
            desc="Adding parent categories to article_id_categories_map",
            total=len(article_id_categories_map),
        ):
            article_id_categories_map[page_id] = self.add_parent_categories(
                categories, self.redirect_df, self.page_df, categorylinks_df
            )
        # Remove entries without categories.
        article_id_categories_map = {
            page_id: categories
            for page_id, categories in article_id_categories_map.items()
            if len(categories) > 0
        }
        return article_id_categories_map

    def read_and_preprocess_articles(self, wiki_pages_articles_xml_path):
        """
        Read and preprocess the articles from the Wikipedia pages-articles XML file.

        Clean the article text by removing wiki markup and making it suitable for text classification.

        Returns:
        A dictionary mapping page IDs to their titles and texts.
        """
        preprocessed_articles = {}
        tree = ET.parse(wiki_pages_articles_xml_path)

        root = tree.getroot()
        all_pages = root.findall("mw:page", NAMESPACES)
        for page in tqdm(all_pages, desc="Reading and preprocessing articles", total=len(all_pages)):
            page_namespace = page.find("mw:ns", NAMESPACES).text
            if page_namespace != "0":
                # Skip non-article pages.
                continue
            page_id = int(page.find("mw:id", NAMESPACES).text)
            title = page.find("mw:title", NAMESPACES).text
            wikitext = page.find("mw:revision/mw:text", NAMESPACES).text
            clean_parsed_text = self.clean_article_text(wikitext)
            if len(clean_parsed_text) > 0:
                preprocessed_articles[page_id] = {"title": title, "text": clean_parsed_text}
        return preprocessed_articles

    def clean_article_text(self, article_text):
        """
        Clean the article text.

        Returns:
        A string of cleaned article text.
        """

        parsed_text = mwparserfromhell.parse(article_text)
        wikilinks = parsed_text.filter_wikilinks()
        for link in wikilinks:
            if parsed_text.contains(link):
                parsed_text.remove(link)
        templates = parsed_text.filter_templates()
        for template in templates:
            if parsed_text.contains(template):
                parsed_text.remove(template)
        return parsed_text.strip_code().strip()

    def create_dataset_json(
        self,
        dataset_json_file_path,
        filter_based_on_label_counts=False,
        min_entries_per_label=MIN_ENTRIES_PER_LABEL,
    ):
        """
        Create the dataset and save it to a JSON file.

        Returns:
        A string of the JSON dataset.
        """
        dataset = self.create_dataset_list()
        if filter_based_on_label_counts:
            dataset = self.filter_dataset_list(
                dataset, self.categorylinks_df, min_entries_per_label
            )
        with open(dataset_json_file_path, "w", encoding="utf-8") as file:
            json.dump(dataset, file, indent=2)
        return json.dumps(dataset, indent=2)

    def create_dataset_list(self):
        """
        Create the dataset.

        Returns:
        A SingleLabelWikiTextClsDataset object.
        """
        preprocessed_articles = self.read_and_preprocess_articles(self.wiki_pages_articles_xml_path)

        article_id_categories_map = self.create_article_id_categories_map(
            preprocessed_articles, self.categorylinks_df, self.cat_df
        )
        dataset = []

        pages_with_cat_links_that_dont_have_articles = 0
        for page_id, category_list in article_id_categories_map.items():
            if page_id not in preprocessed_articles.keys():
                pages_with_cat_links_that_dont_have_articles += 1
                continue
            page_title = preprocessed_articles[page_id]["title"]
            page_text = preprocessed_articles[page_id]["text"]
            dataset.append(
                {
                    "title": page_title,
                    "text": page_text,
                    "page_id": page_id,
                    "categories": category_list,
                }
            )
        print(
            f"Pages with category links that don't have articles: {pages_with_cat_links_that_dont_have_articles}"
        )
        return dataset

    def filter_dataset_list(
        self, dataset_list, categorylinks_df, min_entries_per_label=MIN_ENTRIES_PER_LABEL
    ):
        """
        Filter the dataset to remove labels with fewer than the specified minimum number of entries.

        Returns:
        A filtered dataset.
        """
        label_counts = count_labels(dataset_list)
        filtered_dataset = []
        for sample in tqdm(dataset_list, desc="Filtering dataset list based on label counts", total=len(dataset_list)):
            filtered_categories = []
            for i, label in enumerate(sample["categories"]):
                # We can simply filter out labels that have fewer than the minimum number of entries.
                # If the labels are subcategories of a larger category, the larger category should still be present (because we collected them all in create_article_id_categories_map())
                if label_counts[label] >= min_entries_per_label:
                    filtered_categories.append(label)
            if len(filtered_categories) > 0:
                entry_dict = sample.copy()
                entry_dict.update(
                    {
                        "categories": filtered_categories,
                    }
                )
                filtered_dataset.append(entry_dict)
        return filtered_dataset
