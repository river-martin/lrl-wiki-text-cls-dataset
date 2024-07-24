import pandas as pd
import json
from lrl_wiki_cls.config import WIKI_CATEGORY_NAMESPACE, RESULTS_DIR
from lrl_wiki_cls.load_wiki_tables import load_table
import networkx as nx
from lrl_wiki_cls.wiki_text_cls_dataset import WikiTextClsDataset
from lrl_wiki_cls.utils import check_if_is_junk_cat
import matplotlib.pyplot as plt

"""
This file (`structure_category_metadata.py`) contains the implementation of two classes: `CategoryMetadata` and `HierarchyGraph`. 

The `CategoryMetadata` class is responsible for managing category metadata, such as mapping category titles to IDs and vice versa.
It provides methods to retrieve category IDs, titles, subcategories, and supercategories.

The `HierarchyGraph` class represents a directed graph that represents the hierarchical classification of categories.
It takes a dataset and category metadata as input and constructs a graph using the dataset's category information.
It provides a method to check if the graph has cycles.

The file also includes utility functions for visualizing the hierarchical classification graph and obtaining the initial hierarchy metadata from the dataset.

The `load_dataset_and_cat_metadata` function loads the dataset and category metadata, and the `print_dataset_overview` function prints an overview of the dataset.

Finally, in the `__main__` block, the dataset and category metadata are loaded, the dataset overview is printed, and the hierarchy graph is obtained and plotted.
"""


class CategoryMetadata:
    def __init__(self, title_to_id_file_path, id_to_title_file_path, subcat_links_file_path):
        with open(title_to_id_file_path, "r") as file:
            self.title_to_id = json.load(file)
        with open(id_to_title_file_path, "r") as file:
            self.id_to_title = json.load(file)
        with open(subcat_links_file_path, "r") as file:
            self.subcat_links = json.load(file)
        dict_with_int_keys = {}
        for key, val in self.id_to_title.items():
            dict_with_int_keys[int(key)] = val
        self.id_to_title = dict_with_int_keys

    def get_id(self, title):
        assert type(title) == str
        val = self.title_to_id.get(title)
        assert type(val) == int
        return val

    def get_title(self, id):
        assert type(id) == int
        val = self.id_to_title.get(id)
        assert type(val) == str
        return val

    def get_category_ids(self):
        return list(self.title_to_id.values())

    def get_subcat_links(self):
        return self.subcat_links

    def get_category_titles(self):
        return list(self.title_to_id.keys())

    def get_subcats_of(self, cat):
        subcats = set()
        for link in self.subcat_links:
            if link[0] == self.title_to_id[cat]:
                subcats.add(self.id_to_title[link[1]])
        return list(subcats)

    def get_supcats_of(self, cat):
        supcats = set()
        for link in self.subcat_links:
            if link[1] == self.title_to_id[cat]:
                supcats.add(self.id_to_title[link[0]])
        return list(supcats)


class HierarchyGraph:
    def __init__(self, dataset, cat_metadata):
        self.dataset = dataset
        self.cat_metadata = cat_metadata

        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.cat_metadata.get_category_ids())
        self.G.add_edges_from(self.cat_metadata.get_subcat_links())

    def has_cycle(self):
        try:
            nx.algorithms.cycles.find_cycle(self.G)
        except nx.NetworkXNoCycle:
            return False
        return True


def visualize_hierarchy(graph, title="Category/subcategory network from Zulu Wikipedia"):
    """
    Visualizes the hierarchical classification graph.

    Args:
        hierarchy (nx.DiGraph): A directed graph representing the hierarchical classification.
    """
    from networkx.drawing.nx_agraph import graphviz_layout

    pos = graphviz_layout(graph, prog="dot")
    nx.draw(
        graph,
        pos,
        with_labels=False,
        arrows=True,
        node_size=1,
        font_size=8,
        node_color="blue",
        edge_color="gray",
    )
    plt.title(title)
    plt.show(block=True)


def get_and_plot_hierarchy_graph(dataset, cat_metadata, root="root"):
    hierarchy_graph = HierarchyGraph(dataset, cat_metadata)
    if root == "root":
        # Add artifical root that points to all the top-level categories
        hierarchy_graph.G.add_node(root)
        for node in hierarchy_graph.G.nodes:
            if not list(hierarchy_graph.G.predecessors(node)) and node != root:
                hierarchy_graph.G.add_edge(root, node)
    else:
        root = cat_metadata.get_id(root)
        assert root in hierarchy_graph.G.nodes
    nodes_in_subgraph = nx.descendants(hierarchy_graph.G, root)
    nodes_in_subgraph.add(root)
    graph = hierarchy_graph.G.subgraph(nodes_in_subgraph)
    title = (
        "Category/subcategory network from Zulu Wikipedia"
        if root == "root"
        else f"Subgraph rooted at {cat_metadata.get_title(root)}"
    )
    visualize_hierarchy(graph, title=title)
    plt.show(block=True)
    assert not hierarchy_graph.has_cycle()
    return hierarchy_graph


def get_initial_hierarchy_metadata_and_save_to_json_files(dataset):
    set_of_cats_associated_with_articles = set()
    for entry, list_of_categories in dataset:
        for cat in list_of_categories:
            set_of_cats_associated_with_articles.add(cat)
    # Load the categorylinks and page dataframes
    categorylinks_df = load_table("categorylinks")
    page_df = load_table("page")

    # Create an empty dictionary to store the category title-id mapping
    category_title_id_map = {}
    category_id_title_map = {}
    category_strings = set()
    # Links from the parent category to the subcategory
    subcat_links = set()

    def make_new_cat(category_strings, cat_title_str, category_title_id_map, category_id_title_map):
        # We create our own IDs for the categories, we don't use the page_ids, because not all categories have pages/page-ids.
        cat_id = len(category_strings)
        category_title_id_map[cat_title_str] = int(cat_id)
        category_id_title_map[int(cat_id)] = cat_title_str
        category_strings.add(cat_title_str)
        return cat_id

    def get_cat_title(page_id):
        return page_df.query("page_id==@page_id and page_namespace==@WIKI_CATEGORY_NAMESPACE")[
            "page_title"
        ]

    for index, row in categorylinks_df.iterrows():
        cat_title_str = row["cl_to"].decode("utf-8")
        cat_is_junk = check_if_is_junk_cat(cat_title_str)
        if cat_is_junk or not cat_title_str in set_of_cats_associated_with_articles:
            continue
        if not cat_title_str in category_strings:
            make_new_cat(
                category_strings, cat_title_str, category_title_id_map, category_id_title_map
            )
        if row["cl_type"].decode("utf-8") == "subcat":
            subcat_page_id = row["cl_from"]
            subcat_title = get_cat_title(subcat_page_id)
            if not subcat_title.empty:
                # Add the subcategory to the mapping
                subcat_title = subcat_title.values[0]
                subcat_title_str = subcat_title.decode("utf-8")
                subcat_is_junk = check_if_is_junk_cat(subcat_title_str)
                if subcat_is_junk:
                    continue
                if not subcat_title_str in category_strings:
                    make_new_cat(
                        category_strings,
                        subcat_title_str,
                        category_title_id_map,
                        category_id_title_map,
                    )
                # Add the subcategory link to the set
                link = (
                    category_title_id_map[cat_title_str],
                    category_title_id_map[subcat_title_str],
                )
                is_self_link = link[0] == link[1]
                if not is_self_link:
                    assert category_id_title_map[link[0]] == cat_title_str
                    assert category_id_title_map[link[1]] == subcat_title_str
                    subcat_links.add(link)
    with open(f"{RESULTS_DIR}/zuwiki-title-to-id.json", "w") as file:
        json.dump(category_title_id_map, file, indent=4)
    with open(f"{RESULTS_DIR}/zuwiki-id-to-title.json", "w") as file:
        json.dump(category_id_title_map, file, indent=4)
    with open(f"{RESULTS_DIR}/zuwiki-subcat-links.json", "w") as file:
        json.dump(list(subcat_links), file, indent=4)


def print_dataset_overview(dataset):
    max_num_labels_per_article = 0
    unique_labels = set()
    doc_stats = []

    for entry, categories in dataset:
        doc_stats.append((len(entry["text"]), len(categories)))
        assert len(categories) > 0
        contains_junk_cat = any(check_if_is_junk_cat(cat) for cat in categories)
        assert not contains_junk_cat
        assert len(entry["text"]) > 0
        max_num_labels_per_article = max(max_num_labels_per_article, len(categories))
        unique_labels = unique_labels.union(categories)
    doc_stat_df = pd.DataFrame(doc_stats, columns=["Text Length", "Num. Categories"])

    print(f"Num. documents: {len(dataset.data)}")
    print(f"Num. unique categories: {len(unique_labels)}")
    print()
    print(doc_stat_df.describe())


def load_dataset_and_cat_metadata():
    dataset = WikiTextClsDataset(f"{RESULTS_DIR}/zuwiki-dataset.json")

    get_initial_hierarchy_metadata_and_save_to_json_files(dataset)

    # get_initial_hierarchy_metadata creates the following JSON files.
    title_to_id_file = f"{RESULTS_DIR}/zuwiki-title-to-id.json"
    category_to_id_file = f"{RESULTS_DIR}/zuwiki-id-to-title.json"
    subcat_links_file = f"{RESULTS_DIR}/zuwiki-subcat-links.json"
    cat_metadata = CategoryMetadata(
        title_to_id_file, category_to_id_file, subcat_links_file_path=subcat_links_file
    )
    return dataset, cat_metadata


def create_metadata_json_files():
    dataset = WikiTextClsDataset(f"{RESULTS_DIR}/zuwiki-dataset.json")
    get_initial_hierarchy_metadata_and_save_to_json_files(dataset)


if __name__ == "__main__":
    dataset, cat_metadata = load_dataset_and_cat_metadata()
    print_dataset_overview(dataset)
    hierarchy_graph = get_and_plot_hierarchy_graph(dataset, cat_metadata)
