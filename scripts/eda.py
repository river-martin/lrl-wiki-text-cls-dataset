"""Exploratory analysis of the Zulu Wikipedia multi-label classification dataset.

This file uses the Jupytext "percent" format: ``# %%`` starts a code cell and
``# %% [markdown]`` starts a markdown cell. It is a normal Python script that can
be run end-to-end:

    uv run python scripts/eda.py

Figures are written to ``results/figures/`` instead of being displayed, so the
script works headlessly (CI, SSH, Zed's terminal, etc.).

If you want a notebook again (e.g. for JupyterLab or an editor with a kernel):

    uvx jupytext --to ipynb scripts/eda.py
"""

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lrl_wiki_cls.config import PROJECT_ROOT, RESULTS_DIR
from lrl_wiki_cls.structure_category_metadata import (
    get_and_plot_hierarchy_graph,
    load_dataset_and_cat_metadata,
    print_dataset_overview,
)
from lrl_wiki_cls.wiki_text_cls_dataset import (
    WikiTextClsDataset,
)
from lrl_wiki_cls.wiki_text_cls_dataset import (
    count_labels as get_dict_of_label_counts,
)

# Write figures instead of opening a GUI window (works headlessly).
plt.switch_backend("Agg")

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(name):
    """Save the current matplotlib figure to ``results/figures/<name>.png``."""
    path = FIGURES_DIR / f"{name}.png"
    plt.gcf().savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"Saved figure: {path.relative_to(PROJECT_ROOT)}")
    return path


def plot_hierarchy(dataset, cat_metadata, name, root="root"):
    """Plot and save a hierarchy graph to ``results/figures/<name>.png``."""
    path = FIGURES_DIR / f"{name}.png"
    result = get_and_plot_hierarchy_graph(
        dataset, cat_metadata, root=root, show=False, save_path=str(path)
    )
    print(f"Saved figure: {path.relative_to(PROJECT_ROOT)}")
    return result


dataset, cat_metadata = load_dataset_and_cat_metadata()

# %% [markdown]
# # A high-level overview of the dataset

# %%
# The text length is the number of characters in the text.
print_dataset_overview(dataset)

# %% [markdown]
# # Exploring the structure of the category/subcategory network
#
# ## Visualizing the graph
#
# Each edge in the graph is from a category to one of its subcategories. Hence,
# the root node (which corresponds to the highest level) is at the top of the
# plot.
#
# The root node does not correspond to a category; it is added to link all the
# top-level categories for the visualization.
#
# Furthermore, the parent categories of each category associated with a document
# were collected and added to the set of labels for an article (see the example
# in the next section for further explanation).

# %%
_ = plot_hierarchy(dataset, cat_metadata, "hierarchy_full")

# %% [markdown]
# ## A concrete example
#
# Below we consider the labelling of the article
# "[Misuzulu Zulu](https://zu.wikipedia.org/wiki/Misuzulu_Zulu)" (about Misuzulu
# Zulu, king of the Zulu nation as of 2021).
#
# The article is explicitly categorized into "Abantu baphilayo" ("Living
# people"), "Abazalwa nge-1974" ("Born in 1974") and "Amakhosi akwaZulu" ("Kings
# of Zululand"), but the dataset also includes all parent categories, so the
# article is associated with more than just these three categories.

# %%
example_entry = None
for entry, labels in dataset:
    if entry["title"] == "Misuzulu Zulu":
        example_entry = entry

if example_entry is None:
    print("Example article 'Misuzulu Zulu' not found in this snapshot.")
else:
    print(json.dumps(example_entry["labels"], indent=4))

# %% [markdown]
# For the rest of this example we focus on "Abantu baphilayo" and "Amakhosi
# akwaZulu". Neither is a subcategory of the other and they are not mutually
# exclusive (both are associated with the same article).
#
# However, "Amakhosi akwaZulu" is a subcategory of "AmaZulu" ("The Zulus"), i.e.
# Zulu kings are Zulu.

# %%
if example_entry is not None:
    for cat in example_entry["labels"]:
        if cat in {"Abantu_baphilayo", "Amakhosi_akwaZulu", "AmaZulu"}:
            print(f"Subcategories of {cat}: {cat_metadata.get_subcats_of(cat)}")
            print(f"Supercategories of {cat}: {cat_metadata.get_supcats_of(cat)}")
            print()

# %% [markdown]
# This type of labelling is not possible in the hierarchical-classification
# formulation of Silla and Freitas (2011), which assumes classes are organized
# into a hierarchy of levels that take on one of a set of mutually exclusive
# values. In other words, that framework does not allow a single document to be
# associated with multiple classes from the same level.
#
# In our example the article is associated with at most two top-level classes,
# since "Abantu_baphilayo" and "AmaZulu" are both classes in the highest layer.
#
# **Reference:** Silla, C. and Freitas, A. (2011). A survey of hierarchical
# classification across different application domains. Data Mining and Knowledge
# Discovery, vol. 22, pp. 31-72.

# %% [markdown]
# ## Visualization of the frequency of occurrence of each label

# %%
def build_label_count_df(dataset):
    assert isinstance(dataset, WikiTextClsDataset)
    label_counts = get_dict_of_label_counts(dataset.data)
    count_df = pd.DataFrame(
        label_counts.items(),
        columns=pd.Index(["Category", "Num. docs associated with label"]),
    )
    return (
        count_df.sort_values(by="Num. docs associated with label", ascending=False),
        len(label_counts),
    )


def visualize_label_distribution(count_df, title="Label distribution"):
    sns.displot(
        count_df,
        x="Num. docs associated with label",
        kind="hist",
        kde=True,
        height=5,
        aspect=2,
    )
    plt.title(title)


count_df, num_unique_labels = build_label_count_df(dataset)

# %% [markdown]
# The published dataset keeps **all** labels, so the distribution below has a
# long tail of rare labels.
#
# The `min_entries_per_label=20` threshold comes from Gao et al. (2021), who use
# 16 training examples per class for their few-shot scenario. Assuming an 80/20
# train/test split, that needs at least 20 examples per class (0.8 * 20 = 16),
# so filtering at 20 reproduces a model-ready subset.
#
# **Reference:** [Making Pre-trained Language Models Better Few-shot Learners](https://aclanthology.org/2021.acl-long.295)
# (Gao et al., ACL-IJCNLP 2021).

# %%
threshold = 20
filtered = count_df[count_df["Num. docs associated with label"] >= threshold]
print(f"{len(filtered)} / {len(count_df)} labels have >= {threshold} documents")

visualize_label_distribution(
    count_df,
    title="Num. labels that are associated with $x$ documents",
)
save_current_figure("label_distribution")
count_df.describe()

# %% [markdown]
# # Finding a promising category hierarchy for a single-hierarchy classification problem

# %%
count_df.head(10)

# %%
cat = "Amazwe"
subcats = list(cat_metadata.get_subcats_of(cat))
print(f"Num. subcategories of {cat}: {len(subcats)}")
print(f"Subcategories of {cat}: {json.dumps(subcats, indent=4)}")
print(f"Supercategories of {cat}: {cat_metadata.get_supcats_of(cat)}")

# %%
_ = plot_hierarchy(dataset, cat_metadata, "hierarchy_amazwe", root="Amazwe")

# %% [markdown]
# # Checking that our documents have reasonable lengths

# %%
def count_words(entry):
    return len(entry["text"].split())


unordered_list_of_word_counts = [count_words(entry) for entry in dataset.data]
words_per_entry = sorted(unordered_list_of_word_counts)
word_count_df = pd.DataFrame(words_per_entry, columns=pd.Index(["word count"]))
word_count_df.describe()

# %%
# For this plot we assume models will be limited to 128 tokens (and that every
# word maps to a single token, which is unlikely). Documents are not truncated in
# the dataset itself, but will most likely be truncated during training.
max_seq_len = 128

word_count_df["Word count (after truncation)"] = word_count_df["word count"].apply(
    lambda x: min(x, max_seq_len)
)
sns.displot(
    word_count_df,
    x="Word count (after truncation)",
    kind="hist",
    kde=True,
    height=5,
    aspect=2,
)
plt.title(
    "Num. documents with a given word count (after hypothetically truncating "
    "the articles to 128 words)"
)
save_current_figure("word_count_after_truncation")

# %% [markdown]
# ## Sanity checks / simple fuzz-tests

# %%
index_of_article_with_most_words = int(np.argmax(unordered_list_of_word_counts))
article_with_most_words = dataset.data[index_of_article_with_most_words]
print(f'Title: {article_with_most_words["title"]}')
print(f"Number of words: {count_words(article_with_most_words)}")

# %%
index_of_article_with_least_words = int(np.argmin(unordered_list_of_word_counts))
article_with_least_words = dataset.data[index_of_article_with_least_words]
print(
    f'Title: {article_with_least_words["title"]}\n'
    f'Text: {article_with_least_words["text"]}'
)
print(f"Number of words: {count_words(article_with_least_words)}")

# %% [markdown]
# # Final notes
#
# 1. Pages in the category namespace (such as
#    [Category:Amakhosi_akwaZulu](https://zu.wikipedia.org/wiki/Category:Amakhosi_akwaZulu))
#    are excluded from the dataset.
# 2. Some categories redirect to other categories (e.g. "Amazwe_eAfrika"
#    redirects to "Amazwe_e-Afrika"). Redirects are resolved, so articles
#    labelled with the redirect are labelled with the canonical category.
# 3. Many Zulu Wikipedia articles are very short (the shortest here is a
#    single-word stub like "Mdlani"), so some documents carry little text signal.
