"""Convert the built dataset into Hugging Face ``datasets`` and (optionally) publish it.

Run ``build_dataset`` (or the notebook) first to produce
``results/zuwiki-dataset.json`` and ``results/zuwiki-hierarchy.json``.

The published repo has two configs:

* ``default``   – one row per article, multi-label ``labels`` (``Sequence(ClassLabel)``).
* ``hierarchy`` – one row per category with parents, depth and document counts.

Splits are fixed, seeded and multi-label stratified so results are comparable
across users. ``datasets``/``huggingface_hub`` are imported lazily so the schema
and split helpers can be used (and tested) without the extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from lrl_wiki_cls.config import DUMP_DATE, RESULTS_DIR

DEFAULT_REPO_ID = "river-martin/zuwiki-multilabel-cls"
DATASET_JSON = RESULTS_DIR / "zuwiki-dataset.json"
HIERARCHY_JSON = RESULTS_DIR / "zuwiki-hierarchy.json"
CARD_PATH = RESULTS_DIR / "README.md"


# --- splitting --------------------------------------------------------------


def iterative_stratified_split(records, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split multi-label records so each label appears in roughly ``ratios`` proportions.

    A simple iterative-stratification approach (Sechidis et al., 2011): labels
    are processed rarest-first and each record is assigned to the split that
    currently "wants" that label most.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    split_names = ("train", "validation", "test")
    rng = random.Random(seed)
    n = len(records)

    label_to_indices: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        for label in record["labels"]:
            label_to_indices.setdefault(label, []).append(index)

    assigned: list[int | None] = [None] * n
    capacity = [n * ratio for ratio in ratios]
    desired = {
        label: [len(indices) * ratio for ratio in ratios]
        for label, indices in label_to_indices.items()
    }

    for label in sorted(label_to_indices, key=lambda lab: len(label_to_indices[lab])):
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        for index in indices:
            if assigned[index] is not None:
                continue
            best = max(range(len(split_names)), key=lambda s: (desired[label][s], capacity[s]))
            if capacity[best] <= 0:
                best = max(range(len(split_names)), key=lambda s: capacity[s])
            assigned[index] = best
            capacity[best] -= 1
            for other in records[index]["labels"]:
                desired[other][best] -= 1

    # Records with no labels (should not occur) fall back to remaining capacity.
    for index, split_index in enumerate(assigned):
        if split_index is None:
            best = max(range(len(split_names)), key=lambda s: capacity[s])
            assigned[index] = best
            capacity[best] -= 1

    splits = {name: [] for name in split_names}
    for index, split_index in enumerate(assigned):
        assert split_index is not None
        splits[split_names[split_index]].append(records[index])
    return splits


# --- schema -----------------------------------------------------------------


def label_names(records) -> list[str]:
    names = set()
    for record in records:
        names.update(record["labels"])
    return sorted(names)


def direct_category_names(records) -> list[str]:
    names = set()
    for record in records:
        names.update(record["direct_categories"])
    return sorted(names)


def default_features(records):
    import datasets

    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "language": datasets.Value("string"),
            "page_id": datasets.Value("int32"),
            "revision_id": datasets.Value("int64"),
            "title": datasets.Value("string"),
            "url": datasets.Value("string"),
            "text": datasets.Value("string"),
            "lead": datasets.Value("string"),
            "num_words": datasets.Value("int32"),
            "direct_categories": datasets.Sequence(datasets.Value("string")),
            "labels": datasets.Sequence(datasets.ClassLabel(names=label_names(records))),
        }
    )


def hierarchy_features():
    import datasets

    return datasets.Features(
        {
            "category_id": datasets.Value("int32"),
            "title": datasets.Value("string"),
            "parent_titles": datasets.Sequence(datasets.Value("string")),
            "depth": datasets.Value("int32"),
            "num_documents": datasets.Value("int32"),
        }
    )


def build_hierarchy_rows(records, hierarchy_edges):
    """Build the hierarchy config rows from the dataset labels and parent edges."""
    names = label_names(records)
    name_set = set(names)
    counts = Counter(label for record in records for label in record["labels"])

    parents: dict[str, set[str]] = {name: set() for name in names}
    for edge in hierarchy_edges:
        parent, child = edge["parent"], edge["child"]
        if parent in name_set and child in name_set and parent != child:
            parents[child].add(parent)

    depth_cache: dict[str, int] = {}

    def depth(name, stack=frozenset()):
        if name in depth_cache:
            return depth_cache[name]
        if name in stack:  # cycle guard
            return 0
        if not parents[name]:
            depth_cache[name] = 0
            return 0
        value = 1 + max(depth(parent, stack | {name}) for parent in parents[name])
        depth_cache[name] = value
        return value

    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "category_id": index,
                "title": name,
                "parent_titles": sorted(parents[name]),
                "depth": depth(name),
                "num_documents": counts.get(name, 0),
            }
        )
    return rows


# --- dataset card -----------------------------------------------------------


def build_dataset_metadata() -> dict:
    """YAML front-matter metadata for the dataset card."""
    return {
        "license": "cc-by-sa-4.0",
        "language": ["zu"],
        "pretty_name": "Zulu Wikipedia Multi-label Topic Classification",
        "task_categories": ["text-classification"],
        "tags": [
            "multi-label-classification",
            "hierarchical-classification",
            "wikipedia",
            "zulu",
            "low-resource",
        ],
        "size_categories": ["1K<n<10K"],
        "source_datasets": ["wikipedia"],
    }


def build_card_body(records, splits) -> str:
    label_counts = Counter(label for record in records for label in record["labels"])
    num_words = sorted(record["num_words"] for record in records)
    median_words = num_words[len(num_words) // 2] if num_words else 0
    split_lines = "\n".join(
        f"| {name} | {len(rows):,} |" for name, rows in splits.items()
    )
    return f"""# Dataset Card for Zulu Wikipedia Multi-label Topic Classification

## Dataset Details

### Dataset Description

Multi-label topic classification over the Zulu Wikipedia. Each row is an article
whose `labels` are its Wikipedia categories plus every ancestor in the category
hierarchy. The category hierarchy is also published as a separate `hierarchy`
config.

- **Curated by:** River Martin
- **Language(s) (NLP):** Zulu (`zu`)
- **License:** CC BY-SA 4.0

### Dataset Sources

- **Source data:** Wikimedia `zuwiki` dump snapshot `{DUMP_DATE}`
  (https://dumps.wikimedia.org/zuwiki/), © the contributors of the Zulu Wikipedia.
- **Original snapshot:** the first version of this dataset used dumps dated
  2024-04-01/02. Wikimedia no longer serves those files, so rebuilds use the
  pinned snapshot and may differ.
- **Repository:** https://github.com/river-martin/lrl-wiki-text-cls-dataset

### Attribution and License

This dataset is derived from the **Zulu Wikipedia**
(https://zu.wikipedia.org), © the contributors of the Zulu Wikipedia, and is
licensed under CC BY-SA 4.0. Markup was removed, categories were derived and
expanded along the category hierarchy, and labels with too few associated
articles may be rare.

## Uses

### Direct Use

- Training and evaluating multi-label (and hierarchical) text classifiers for Zulu.
- Benchmarking multilingual encoders on a low-resource African language.
- Few-shot experiments: filter labels by `num_documents` (e.g. >= 20).

### Out-of-Scope Use

- The labels are **derived ("silver") labels**, not gold human annotation; do not
  treat them as ground truth.
- Not suitable for high-stakes decisions about people, places or organisations.
- Not a factual source; the text is a cleaned derivative of Wikipedia.

## Dataset Structure

### Configs and Fields

#### `default`

| field | type | description |
| --- | --- | --- |
| `id` | string | Stable identifier (`zu-<page_id>`) |
| `language` | string | ISO 639-1 code (`zu`) |
| `page_id` | int32 | MediaWiki page id |
| `revision_id` | int64 | MediaWiki revision id |
| `title` | string | Article title |
| `url` | string | Link to the source article |
| `text` | string | Article prose (wikitext markup removed) |
| `lead` | string | Intro section (text before the first heading) |
| `num_words` | int32 | Word count of `text` |
| `direct_categories` | list[str] | Categories as recorded on the article |
| `labels` | list[ClassLabel] | Direct categories + all ancestors |

#### `hierarchy`

This config is a **single, split-independent lookup table** (one row per
category), not a training/validation/test split. It is published under the
`metadata` split (the Hub requires every config to name a split) and is the same
for all splits of the `default` config.

| field | type | description |
| --- | --- | --- |
| `category_id` | int32 | Stable integer id (index into sorted label names) |
| `title` | string | Category title |
| `parent_titles` | list[str] | Parent categories present in the dataset |
| `depth` | int32 | Distance from a root category |
| `num_documents` | int32 | Number of articles carrying this label |

### Splits

Fixed, seeded, multi-label stratified splits (no article appears twice):

| split | rows |
| --- | --- |
{split_lines}

### Statistics

- Documents: **{len(records):,}**
- Unique labels: **{len(label_counts):,}**
- Median article length: **{median_words:,} words**
- Longest article: **{max(num_words) if num_words else 0:,} words**

## Dataset Creation

### Curation Rationale

The goal is a reproducible, packaged benchmark for multi-label topic
classification in a low-resource South African language. Labels come from the
same category system Wikipedia itself uses (via `categorylinks`/`linktarget`),
not from hand annotation.

### Source Data

#### Data Collection and Processing

The build downloads a pinned `zuwiki` dump snapshot and parses the article XML
(streamed from `.bz2`) plus the `categorylinks`, `linktarget`, `page` and
`redirect` SQL dumps. Redirects are resolved, maintenance categories removed,
and each article's categories are expanded to include all ancestors.

#### Changes made

This dataset is a derivative of the source articles; the text was transformed as
follows:

- Wikitext markup was stripped to plain text (link display text preserved).
- **References/sources, "see also"/related, external-links, and image-gallery
  sections are removed entirely**, along with templates, tables, citations and
  stray HTML tags. Only article prose and content sections remain.
- Content section headings (e.g. "History"/"Umlando") are retained as plain text.
- Redirects were resolved and maintenance categories filtered; `labels` are
  expanded along the category hierarchy.
- Pages with empty text or no usable categories are omitted.

#### Who are the source data producers?

The contributors of the Zulu Wikipedia (https://zu.wikipedia.org).

### Annotations

#### Annotation process

Labels are derived automatically from MediaWiki category links and the category
hierarchy. No human annotation was performed. The resulting labels are therefore
"silver" labels and contain noise.

## Bias, Risks, and Limitations

- Labels are derived from Wikipedia categories, not human annotation, and
  contain noise and some non-topical categories.
- Zulu Wikipedia is stub-heavy (median article length is a few words), so many
  documents carry little text signal.
- Many labels are rare; see `num_documents` in the `hierarchy` config.
- The dataset reflects the Zulu Wikipedia snapshot `{DUMP_DATE}`; it is not a
  balanced or representative sample of Zulu-language text in general.

### Recommendations

- Filter labels by `num_documents` for a model-ready subset (the project also
  documents a `min_entries_per_label=20` recipe).
- Treat the labels as weak supervision and validate downstream decisions.
- Keep the CC BY-SA 4.0 license and attribution in redistributions.

## Glossary

- `labels`: direct categories plus all ancestor categories.
- `direct_categories`: categories recorded on the article itself.
- `hierarchy`: category lookup table with parents, depth and document counts.

## More Information

The build and publishing pipeline lives in the project repository. Dumps are
pinned in `config.DUMP_DATE` for reproducibility.

## Dataset Card Authors

River Martin
"""


def build_dataset_card(records, splits, repo_id=DEFAULT_REPO_ID) -> str:
    """Full dataset card (YAML front matter + template body), for local preview."""
    yaml_text = yaml.safe_dump(
        build_dataset_metadata(), sort_keys=False, allow_unicode=True
    ).strip()
    return f"---\n{yaml_text}\n---\n\n{build_card_body(records, splits)}"


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def split_front_matter(text: str):
    """Split a card into (yaml_front_matter, body). Returns (None, text) if absent."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end():]


def merge_readme(existing_readme: str, body: str, metadata: dict) -> str:
    """Merge our card body/metadata into an existing README, preserving its YAML.

    ``push_to_hub`` writes a README containing a ``dataset_info`` block. This
    keeps that block (so configs/features stay discoverable) while replacing the
    typically-empty body with our dataset card.
    """
    existing_yaml = None
    if existing_readme:
        existing_yaml, _ = split_front_matter(existing_readme)
    merged = {}
    if existing_yaml:
        try:
            loaded = yaml.safe_load(existing_yaml)
            if isinstance(loaded, dict):
                merged = loaded
        except yaml.YAMLError:
            merged = {}
    merged.update(metadata)
    yaml_text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{yaml_text}\n---\n\n{body.lstrip()}"


def upload_dataset_card(repo_id, body, metadata):
    """Upload the card as README.md, preserving the Hub-generated dataset_info."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    existing = ""
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type="dataset", force_download=True
        )
        existing = Path(local).read_text(encoding="utf-8")
    except (HfHubHTTPError, OSError):
        # README not present yet (or unreachable) — start from our own metadata.
        existing = ""
    merged = merge_readme(existing, body, metadata)
    HfApi().upload_file(
        path_or_fileobj=merged.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    print("Uploaded README.md (dataset card)")


# --- publishing -------------------------------------------------------------


def load_artifacts(dataset_json=DATASET_JSON, hierarchy_json=HIERARCHY_JSON):
    with open(dataset_json, "r", encoding="utf-8") as file:
        records = json.load(file)
    with open(hierarchy_json, "r", encoding="utf-8") as file:
        hierarchy_edges = json.load(file)
    return records, hierarchy_edges


def publish(repo_id=DEFAULT_REPO_ID, private=False, seed=42, dataset_json=DATASET_JSON,
            hierarchy_json=HIERARCHY_JSON, dry_run=False):
    """Build ``datasets.DatasetDict`` objects and push them to the Hugging Face Hub."""
    import datasets

    records, hierarchy_edges = load_artifacts(dataset_json, hierarchy_json)
    splits = iterative_stratified_split(records, seed=seed)

    default_ds = datasets.DatasetDict(
        {
            name: datasets.Dataset.from_list(rows, features=default_features(records))
            for name, rows in splits.items()
        }
    )
    hierarchy_ds = datasets.DatasetDict(
        {
            "metadata": datasets.Dataset.from_list(
                build_hierarchy_rows(records, hierarchy_edges), features=hierarchy_features()
            )
        }
    )

    body = build_card_body(records, splits)
    metadata = build_dataset_metadata()
    card = build_dataset_card(records, splits, repo_id=repo_id)
    CARD_PATH.write_text(card, encoding="utf-8")
    print(f"Wrote dataset card draft to {CARD_PATH}")

    if dry_run:
        print("Dry run: skipping push_to_hub.")
        return default_ds, hierarchy_ds

    default_ds.push_to_hub(repo_id, config_name="default", private=private)
    hierarchy_ds.push_to_hub(repo_id, config_name="hierarchy", private=private)
    upload_dataset_card(repo_id, body, metadata)
    print(f"Published to https://huggingface.co/datasets/{repo_id}")
    return default_ds, hierarchy_ds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build locally without pushing")
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="only (re)upload README.md, without re-pushing the data",
    )
    parser.add_argument("--dataset-json", default=str(DATASET_JSON))
    parser.add_argument("--hierarchy-json", default=str(HIERARCHY_JSON))
    args = parser.parse_args(argv)

    if args.card_only:
        records, _ = load_artifacts(args.dataset_json, args.hierarchy_json)
        splits = iterative_stratified_split(records)
        upload_dataset_card(
            args.repo_id, build_card_body(records, splits), build_dataset_metadata()
        )
        return

    publish(
        repo_id=args.repo_id,
        private=args.private,
        dry_run=args.dry_run,
        dataset_json=args.dataset_json,
        hierarchy_json=args.hierarchy_json,
    )


if __name__ == "__main__":
    main()
