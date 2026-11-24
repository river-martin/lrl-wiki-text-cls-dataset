# LRL Wiki Text Classification Dataset

A reproducible pipeline that turns the **Zulu Wikipedia** into a multi-label
text classification dataset, and publishes it to the Hugging Face Hub.

Each article becomes one row with:

- the article title and full (wikitext-stripped) text,
- the article's **direct categories** (as recorded by Wikipedia), and
- **labels**: those categories plus every ancestor category in the category tree
  (hierarchical multi-label classification).

The dataset also ships the **category hierarchy** itself, so users can work at
the leaf, hierarchical, or top-level granularity.

## What the pipeline needs

The build downloads a pinned Wikimedia dump snapshot and needs no database
server:

| file | why |
| --- | --- |
| `pages-articles.xml.bz2` | article text, titles, page/revision ids |
| `categorylinks.sql.gz` | which categories each page belongs to |
| `linktarget.sql.gz` | resolves `categorylinks.cl_target_id` to a title |
| `page.sql.gz` | page namespaces, titles, redirects |
| `redirect.sql.gz` | category redirect normalisation |
| `category.sql.gz`, `langlinks.sql.gz` | loaded for completeness/tests |

The dump date is pinned in `src/lrl_wiki_cls/config.py` (`DUMP_DATE`) for
reproducibility. The previous README referenced April 2024 data, but Wikimedia
purges old dumps, so the current build pins the latest available snapshot. If
the pinned date is purged, update `DUMP_DATE` and rebuild (results will differ).

## Installation (Ubuntu)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

    # Run these commands from the project's root directory.
    uv sync
`uv sync` creates a `.venv`, installs this project in editable mode (so code edits take effect immediately), and installs the exact versions recorded in `uv.lock` (including the `dev` dependency group).

No system Graphviz is required: hierarchy plots use a pure-NetworkX layered
layout.

Run anything inside the environment with `uv run`, e.g. `uv run pytest tests`.

## Building the dataset

    # 1. Download the pinned dump files (resumable, size- and CRC-validated).
    uv run python -m lrl_wiki_cls.download_dumps --verify

    # 2. Parse the dumps and write results/zuwiki-dataset.json
    #    and results/zuwiki-hierarchy.json.
    uv run python -m lrl_wiki_cls.wiki_text_cls_dataset

    # Optional: drop very short articles (default: keep everything).
    # Zulu Wikipedia is stub-heavy (median ~8 words), so 0 is the recommended
    # default; a threshold like 10 removes most of the corpus, not just stubs.
    uv run python -m lrl_wiki_cls.wiki_text_cls_dataset --min-words 5

The SQL dumps are parsed directly into pandas (cached under `data/cache/`); the
XML dump is streamed from `.bz2` with `iterparse`, so it is never decompressed to
disk and never fully loaded into memory.

### Optional: model-ready subset

The published dataset keeps *all* labels. To reproduce the paper-style setup
that keeps only labels with at least 16 training examples (20 documents), use
the provided helper:

    from lrl_wiki_cls.wiki_text_cls_dataset import filter_records_by_label_count
    subset = filter_records_by_label_count(records, min_entries_per_label=20)

## Using the dataset

    from lrl_wiki_cls.wiki_text_cls_dataset import WikiTextClsDataset
    from lrl_wiki_cls.config import RESULTS_DIR

    dataset = WikiTextClsDataset(f"{RESULTS_DIR}/zuwiki-dataset.json")
    for entry, labels in dataset:
        print(entry["title"])
        print(entry["text"])
        print(entry["direct_categories"])  # categories on the page
        print(labels)                      # + all ancestors

### Category hierarchy and metadata

    from lrl_wiki_cls.structure_category_metadata import (
        load_dataset_and_cat_metadata,
        get_and_plot_hierarchy_graph,
        print_dataset_overview,
    )
    dataset, cat_metadata = load_dataset_and_cat_metadata()
    print_dataset_overview(dataset)
    networkx_digraph = get_and_plot_hierarchy_graph(dataset, cat_metadata)

## Publishing to Hugging Face

After building, publish the two dataset configs (`default` and `hierarchy`):

    # Preview: builds the DatasetDict objects and writes results/README.md
    # without uploading (requires the `datasets` dependency from `uv sync`).
    uv run python -m lrl_wiki_cls.publish_hf_dataset --dry-run

    # Authenticate once, then publish. The dataset card is uploaded as the repo's
    # README.md (merged with the Hub-generated `dataset_info`, so the viewer keeps
    # discovering the configs).
    uv run python -c "from huggingface_hub import login; login()"
    uv run python -m lrl_wiki_cls.publish_hf_dataset --repo-id <user>/<dataset-name>

    # Already pushed the data but the README renders empty? Re-upload just the card:
    uv run python -m lrl_wiki_cls.publish_hf_dataset --card-only --repo-id <user>/<dataset-name>

The published schema:

- **`default`**: `id`, `language`, `page_id`, `revision_id`, `title`, `url`,
  `text`, `lead`, `num_words`, `direct_categories`, `labels`
  (`Sequence(ClassLabel)`), split into fixed, seeded, multi-label-stratified
  `train`/`validation`/`test`.
- **`hierarchy`**: `category_id`, `title`, `parent_titles`, `depth`,
  `num_documents`. This is a split-independent lookup table; load it with
  `split="metadata"` (e.g. `load_dataset(repo, "hierarchy", split="metadata")`).

## Running the tests

    uv run pytest tests -s

The test suite runs on small in-repo fixtures and does not need network access, a
database, or the full dumps.

## Exploratory analysis

The analysis lives in `scripts/eda.py` as a normal Python script in Jupytext
"percent" format. Run it end-to-end with:

    uv run python scripts/eda.py

It writes figures to `results/figures/` (no GUI needed), which works in a plain
terminal and in editors without notebook support (e.g. Zed).

If you want a notebook again (JupyterLab or an editor with a kernel), convert it:

    uvx jupytext --to ipynb scripts/eda.py

To make the uv environment available as a Jupyter kernel:

    uv run python -m ipykernel install --user --name lrl-wiki-cls

## Notes

- The original Zulu data was collected on 1–2 April 2024. Those dumps are no
  longer served by Wikimedia, so a fresh build uses the currently pinned
  snapshot (see `config.DUMP_DATE`). Archive `data/` (or the published Parquet
  files) if you need to freeze a specific snapshot.
- Labels are derived ("silver") labels from Wikipedia categories and contain
  noise. Many Zulu Wikipedia articles are very short (the median is well under
  100 characters), so many documents carry little text signal.

## License

This repository contains two separately licensed works:

- **Source code:** GNU General Public License v3.0 or later (`GPL-3.0-or-later`).
  See [`LICENSE`](LICENSE).
- **Generated dataset:** Creative Commons Attribution-ShareAlike 4.0
  International (`CC BY-SA 4.0`), because it is a derivative work of the Zulu
  Wikipedia. See [`LICENSE-CC-BY-SA-4.0.txt`](LICENSE-CC-BY-SA-4.0.txt) and
  [`ATTRIBUTION.md`](ATTRIBUTION.md) for the required attribution.
