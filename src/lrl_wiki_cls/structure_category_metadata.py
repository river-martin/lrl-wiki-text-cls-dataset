"""Category metadata and hierarchy graph utilities for the dataset.

Everything here is derived from the two JSON artifacts produced by the build step
(``zuwiki-dataset.json`` and ``zuwiki-hierarchy.json``); no database or dump
access is required.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from lrl_wiki_cls.config import RESULTS_DIR
from lrl_wiki_cls.utils import check_if_is_junk_cat
from lrl_wiki_cls.wiki_text_cls_dataset import WikiTextClsDataset

DATASET_JSON = RESULTS_DIR / "zuwiki-dataset.json"
HIERARCHY_JSON = RESULTS_DIR / "zuwiki-hierarchy.json"
TITLE_TO_ID_JSON = RESULTS_DIR / "zuwiki-title-to-id.json"
ID_TO_TITLE_JSON = RESULTS_DIR / "zuwiki-id-to-title.json"
SUBCAT_LINKS_JSON = RESULTS_DIR / "zuwiki-subcat-links.json"


class CategoryMetadata:
    def __init__(self, title_to_id_file_path, id_to_title_file_path, subcat_links_file_path):
        with open(title_to_id_file_path, "r") as file:
            self.title_to_id = json.load(file)
        with open(id_to_title_file_path, "r") as file:
            self.id_to_title = {int(key): value for key, value in json.load(file).items()}
        with open(subcat_links_file_path, "r") as file:
            self.subcat_links = json.load(file)

    def get_id(self, title):
        assert isinstance(title, str)
        val = self.title_to_id.get(title)
        assert isinstance(val, int)
        return val

    def get_title(self, id):
        assert isinstance(id, int)
        val = self.id_to_title.get(id)
        assert isinstance(val, str)
        return val

    def get_category_ids(self):
        return list(self.title_to_id.values())

    def get_subcat_links(self):
        return self.subcat_links

    def get_category_titles(self):
        return list(self.title_to_id.keys())

    def get_subcats_of(self, cat):
        cat_id = self.title_to_id[cat]
        return list({self.id_to_title[link[1]] for link in self.subcat_links if link[0] == cat_id})

    def get_supcats_of(self, cat):
        cat_id = self.title_to_id[cat]
        return list({self.id_to_title[link[0]] for link in self.subcat_links if link[1] == cat_id})


class HierarchyGraph:
    def __init__(self, dataset, cat_metadata):
        self.dataset = dataset
        self.cat_metadata = cat_metadata
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.cat_metadata.get_category_ids())
        self.G.add_edges_from([tuple(link) for link in self.cat_metadata.get_subcat_links()])

    def has_cycle(self):
        try:
            nx.find_cycle(self.G)
        except nx.NetworkXNoCycle:
            return False
        return True


def _hierarchical_layout(graph):
    """A layered tree layout computed with pure NetworkX (no Graphviz/fontconfig).

    Each node's layer is its longest distance from a root; nodes in a layer are
    spread horizontally. Returns a spring layout if the graph is not a DAG.
    """
    if not nx.is_directed_acyclic_graph(graph):
        return nx.spring_layout(graph, seed=42)

    depths = {}
    for node in nx.topological_sort(graph):
        predecessors = list(graph.predecessors(node))
        depths[node] = 0 if not predecessors else 1 + max(depths[p] for p in predecessors)

    layers: dict[int, list] = {}
    for node, depth in depths.items():
        layers.setdefault(depth, []).append(node)

    pos = {}
    for depth, nodes in layers.items():
        ordered = sorted(nodes, key=str)
        for index, node in enumerate(ordered):
            pos[node] = (index - (len(ordered) - 1) / 2, -depth)
    return pos


def visualize_hierarchy(
    graph,
    title="Category/subcategory network from Zulu Wikipedia",
    show=True,
    save_path=None,
):
    plt.figure(figsize=(14, 10))
    pos = _hierarchical_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=1,
        font_size=8,
        node_color="blue",
        edge_color="gray",
    )
    plt.title(title)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    elif show:
        plt.show(block=True)


def get_and_plot_hierarchy_graph(dataset, cat_metadata, root="root", show=True, save_path=None):
    hierarchy_graph = HierarchyGraph(dataset, cat_metadata)
    if root == "root":
        hierarchy_graph.G.add_node(root)
        for node in list(hierarchy_graph.G.nodes):
            if node != root and not list(hierarchy_graph.G.predecessors(node)):
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
    visualize_hierarchy(graph, title=title, show=show, save_path=save_path)
    assert not hierarchy_graph.has_cycle()
    return hierarchy_graph


def get_initial_hierarchy_metadata_and_save_to_json_files(
    dataset,
    hierarchy_edges,
    title_to_id_path=TITLE_TO_ID_JSON,
    id_to_title_path=ID_TO_TITLE_JSON,
    subcat_links_path=SUBCAT_LINKS_JSON,
):
    """Assign integer ids to the labels and save the title/id maps and subcat links.

    ``hierarchy_edges`` is a list of ``{"parent": ..., "child": ...}`` rows, as
    written by the build step.
    """
    labels = set()
    for entry, categories in dataset:
        labels.update(categories)

    category_title_id_map = {}
    category_id_title_map = {}

    def ensure_category(title):
        if title not in category_title_id_map:
            cat_id = len(category_title_id_map)
            category_title_id_map[title] = cat_id
            category_id_title_map[cat_id] = title
        return category_title_id_map[title]

    for title in sorted(labels):
        ensure_category(title)

    subcat_links = set()
    for edge in hierarchy_edges:
        parent, child = edge["parent"], edge["child"]
        if parent not in labels or child not in labels:
            continue
        if check_if_is_junk_cat(parent) or check_if_is_junk_cat(child):
            continue
        parent_id, child_id = ensure_category(parent), ensure_category(child)
        if parent_id != child_id:
            subcat_links.add((parent_id, child_id))

    with open(title_to_id_path, "w") as file:
        json.dump(category_title_id_map, file, indent=4, ensure_ascii=False)
    with open(id_to_title_path, "w") as file:
        json.dump(category_id_title_map, file, indent=4, ensure_ascii=False)
    with open(subcat_links_path, "w") as file:
        json.dump(sorted([list(link) for link in subcat_links]), file, indent=4, ensure_ascii=False)


def print_dataset_overview(dataset):
    max_num_labels_per_article = 0
    unique_labels = set()
    doc_stats = []

    for entry, categories in dataset:
        doc_stats.append((len(entry["text"]), len(categories)))
        assert len(categories) > 0
        assert not any(check_if_is_junk_cat(cat) for cat in categories)
        assert len(entry["text"]) > 0
        max_num_labels_per_article = max(max_num_labels_per_article, len(categories))
        unique_labels = unique_labels.union(categories)

    doc_stat_df = pd.DataFrame(
        doc_stats, columns=pd.Index(["Text Length", "Num. Categories"])
    )
    print(f"Num. documents: {len(dataset)}")
    print(f"Num. unique categories: {len(unique_labels)}")
    print(f"Max labels on one document: {max_num_labels_per_article}")
    print()
    print(doc_stat_df.describe())


def load_dataset_and_cat_metadata():
    dataset = WikiTextClsDataset(DATASET_JSON)
    create_metadata_json_files()
    cat_metadata = CategoryMetadata(TITLE_TO_ID_JSON, ID_TO_TITLE_JSON, SUBCAT_LINKS_JSON)
    return dataset, cat_metadata


def create_metadata_json_files():
    dataset = WikiTextClsDataset(DATASET_JSON)
    with open(HIERARCHY_JSON, "r", encoding="utf-8") as file:
        hierarchy_edges = json.load(file)
    get_initial_hierarchy_metadata_and_save_to_json_files(dataset, hierarchy_edges)


if __name__ == "__main__":
    dataset, cat_metadata = load_dataset_and_cat_metadata()
    print_dataset_overview(dataset)
    get_and_plot_hierarchy_graph(dataset, cat_metadata)
