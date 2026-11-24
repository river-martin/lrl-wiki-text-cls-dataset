"""Build and represent the Zulu Wikipedia multi-label text classification dataset."""

from __future__ import annotations

import bz2
import json
import re
import xml.etree.ElementTree as ET

import mwparserfromhell
from mwparserfromhell.nodes import Heading
from mwparserfromhell.wikicode import Wikicode
from tqdm import tqdm

from lrl_wiki_cls.config import (
    LANGUAGE_CODE,
    MIN_ENTRIES_PER_LABEL,
    WIKI_CATEGORY_NAMESPACE,
    WIKI_DOMAIN,
    dump_path,
)
from lrl_wiki_cls.utils import check_if_is_junk_cat


def count_labels(dataset_list):
    """Count how many articles are associated with each label."""
    label_counts: dict[str, int] = {}
    for entry in dataset_list:
        for label in entry["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


def filter_records_by_label_count(dataset_list, min_entries_per_label=MIN_ENTRIES_PER_LABEL):
    """Return a copy of ``dataset_list`` keeping only labels seen at least ``min_entries_per_label`` times.

    Published datasets are *not* filtered by default; this is provided as the
    documented recipe for producing a model-ready (e.g. few-shot) subset.
    """
    label_counts = count_labels(dataset_list)
    filtered = []
    for sample in dataset_list:
        kept = [label for label in sample["labels"] if label_counts[label] >= min_entries_per_label]
        if kept:
            entry = dict(sample)
            entry["labels"] = kept
            filtered.append(entry)
    return filtered


class WikiTextClsDataset:
    """An in-memory multi-label text classification dataset over Wikipedia articles.

    Iterating yields ``(entry, labels)`` pairs, where ``entry`` is the full record
    and ``labels`` is ``entry["labels"]``.
    """

    def __init__(self, json_file=None, json_dict=None, transform=None):
        if json_dict is not None:
            self.data = json_dict
        elif json_file is not None:
            with open(json_file, "r", encoding="utf-8") as file:
                self.data = json.load(file)
        else:
            raise ValueError("Provide either json_file or json_dict")
        self.transform = transform
        self.labels = [sample["labels"] for sample in self.data]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for index in range(len(self.data)):
            yield self[index]

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


# --- XML stream helpers -----------------------------------------------------


def _strip_templates(text: str) -> str:
    """Remove ``{{ ... }}`` templates, including nested ones.

    mwparserfromhell fails to parse some templates that contain raw HTML (e.g.
    ``{{DISPLAYTITLE:<span …>}}</span>``), so remove them textually first.
    """
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            if depth > 0:
                depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _strip_tables(text: str) -> str:
    """Remove wikitext tables (``{| ... |}``), which mwparserfromhell does not model."""
    out = []
    depth = 0
    for line in text.splitlines():
        opens = line.count("{|")
        closes = line.count("|}")
        if depth == 0:
            if opens:
                depth = opens - closes
                depth = max(depth, 0)
                continue
            out.append(line)
        else:
            depth += opens - closes
            depth = max(depth, 0)
    return "\n".join(out)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# Section headings whose entire section is boilerplate (references, links,
# "see also", etc.). Matched as substrings against the normalised heading so
# variants like "Imithombo yolwazi" and "Amakhasi amangaphandle" are covered.
BOILERPLATE_HEADING_KEYWORDS = (
    "imithombo",
    "izinkomba",
    "izikhombo",
    "izithenjwa",
    "amaphatho",
    "izixhumanisi",
    "amakhasi amangaphandle",
    "amakhasi ngaphandle",
    "izithombe",
    "bhekafuthi",
    "bheka futhi",
    "okuhlobene",
    "references",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "footnotes",
    "citations",
    "gallery",
    "notes",
)


def _normalise_heading(title: str) -> str:
    return re.sub(r"[\s_]+", " ", title.lower()).strip()


def is_boilerplate_heading(title: str) -> bool:
    normalised = _normalise_heading(title)
    return any(keyword in normalised for keyword in BOILERPLATE_HEADING_KEYWORDS)


# Exact heading strings safe to drop as standalone lines even if they were not
# parsed as a Heading node (e.g. after an unclosed tag). Exact matching avoids
# deleting ordinary sentences that merely contain the word "imithombo".
BOILERPLATE_HEADINGS_EXACT = frozenset(
    {
        "imithombo",
        "imithombo yolwazi",
        "izinkomba",
        "izikhombo",
        "izithenjwa",
        "amaphatho",
        "izixhumanisi zangaphandle",
        "amakhasi amangaphandle",
        "amakhasi ngaphandle",
        "izithombe",
        "bhekafuthi",
        "bheka futhi",
        "okuhlobene",
        "references",
        "external links",
        "see also",
        "further reading",
        "bibliography",
        "footnotes",
        "citations",
        "gallery",
        "notes",
    }
)


def _child(element, name):
    for child in element:
        if _localname(child.tag) == name:
            return child
    return None


def _child_text(element, name):
    child = _child(element, name)
    if child is None or child.text is None:
        return None
    return child.text


class WikiTextClsDatasetCreator:
    """Turn a Wikipedia dump + SQL table dumps into dataset records.

    Labels come from the ``categorylinks``/``linktarget`` tables (the same source
    Wikipedia itself uses), not from scraping wikitext, and are expanded along
    the category hierarchy. Each record keeps both the *direct* categories of the
    article and the fully expanded ``labels``.
    """

    def __init__(self, wiki_pages_articles_xml_path, categorylinks_df, page_df, redirect_df, linktarget_df):
        self.wiki_pages_articles_xml_path = wiki_pages_articles_xml_path
        self.categorylinks_df = categorylinks_df
        self.page_df = page_df
        self.redirect_df = redirect_df
        self.linktarget_df = linktarget_df

        self._linktarget_titles = self._build_linktarget_titles()
        self._title_to_canonical = self._build_redirect_aliases()
        self._page_id_to_title = dict(zip(self.page_df["page_id"], self.page_df["page_title"]))
        self.parent_map = self._build_parent_map()

    # -- category graph ------------------------------------------------------

    def _build_linktarget_titles(self) -> dict[int, str]:
        df = self.linktarget_df
        df = df[df["lt_namespace"] == WIKI_CATEGORY_NAMESPACE]
        return dict(zip(df["lt_id"], df["lt_title"]))

    def _build_redirect_aliases(self) -> dict[str, str]:
        """Map a category title to the title it redirects to (one hop; chains resolved lazily)."""
        category_pages = self.page_df[self.page_df["page_namespace"] == WIKI_CATEGORY_NAMESPACE]
        title_to_page_id = dict(zip(category_pages["page_title"], category_pages["page_id"]))
        redirect_targets = dict(zip(self.redirect_df["rd_from"], self.redirect_df["rd_title"]))
        aliases = {}
        for title, page_id in title_to_page_id.items():
            target = redirect_targets.get(page_id)
            if target is not None:
                aliases[title] = target
        return aliases

    def canonical_title(self, title: str) -> str:
        """Follow category redirects (with cycle protection) to a canonical title."""
        seen = set()
        while title in self._title_to_canonical and title not in seen:
            seen.add(title)
            title = self._title_to_canonical[title]
        return title

    def _build_parent_map(self) -> dict[str, set[str]]:
        """Map each category title to the set of its (canonical) parent category titles."""
        category_page_ids = set(
            self.page_df.loc[self.page_df["page_namespace"] == WIKI_CATEGORY_NAMESPACE, "page_id"]
        )
        subcat_rows = self.categorylinks_df[
            self.categorylinks_df["cl_from"].isin(category_page_ids)
        ]
        parent_map: dict[str, set[str]] = {}
        for child_id, target_id in zip(subcat_rows["cl_from"], subcat_rows["cl_target_id"]):
            child_title = self._page_id_to_title_for(child_id)
            parent_title = self._linktarget_titles.get(target_id)
            if not child_title or not parent_title:
                continue
            child_title = self.canonical_title(child_title)
            parent_title = self.canonical_title(parent_title)
            if check_if_is_junk_cat(child_title) or check_if_is_junk_cat(parent_title):
                continue
            if child_title == parent_title:
                continue
            parent_map.setdefault(child_title, set()).add(parent_title)
        return parent_map

    def _page_id_to_title_for(self, page_id: int):
        # page_id -> title is only meaningful for category pages here.
        return self._page_id_to_title.get(page_id)

    def get_hierarchy_edges(self) -> list[tuple[str, str]]:
        """Return ``(parent_title, child_title)`` pairs of the category hierarchy."""
        edges = set()
        for child, parents in self.parent_map.items():
            for parent in parents:
                edges.add((parent, child))
        return sorted(edges)

    # -- article processing --------------------------------------------------

    def _preprocess_raw(self, wikitext: str) -> str:
        """Remove constructs that mwparserfromhell handles poorly: comments, templates, tables."""
        wikitext = re.sub(r"<!--.*?-->", " ", wikitext, flags=re.DOTALL)
        wikitext = _strip_templates(wikitext)
        return _strip_tables(wikitext)

    def _remove_boilerplate_sections(self, code):
        """Drop boilerplate sections (references, external links, see-also) entirely.

        The heading alone is removed together with its body, because sections
        like "External links"/"See also" leave their link labels behind after
        cleaning and would otherwise become orphaned text.
        """
        nodes = list(code.nodes)
        headings = [(index, node) for index, node in enumerate(nodes) if isinstance(node, Heading)]
        drop = set()
        for position, (index, heading) in enumerate(headings):
            title = heading.title.strip_code().strip()
            if not is_boilerplate_heading(title):
                continue
            end = headings[position + 1][0] if position + 1 < len(headings) else len(nodes)
            drop.update(range(index, end))
        if not drop:
            return code
        return Wikicode([node for index, node in enumerate(nodes) if index not in drop])

    def clean_wikicode(self, wikicode) -> str:
        """Strip markup while preserving link display text (unlike deleting links outright).

        Templates, File/Image/Category links, and reference tags are removed
        entirely before ``strip_code`` renders the remaining links and
        formatting as plain text.
        """
        for template in wikicode.filter_templates():
            if wikicode.contains(template):
                wikicode.remove(template)
        for link in wikicode.filter_wikilinks():
            target = str(link.title).strip().lower()
            if target.startswith(("file:", "image:", "category:")) and wikicode.contains(link):
                wikicode.remove(link)
        for tag in wikicode.filter_tags(
            matches=lambda t: str(t.tag).strip().lower() in {"ref", "references"}
        ):
            if wikicode.contains(tag):
                wikicode.remove(tag)
        text = wikicode.strip_code().strip()
        # Remove stray HTML/XML tags (e.g. an unterminated <nowiki>) that survive
        # wikitext parsing. The pattern requires a letter after '<' so it does
        # not eat ordinary comparisons like "a < b".
        text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if _normalise_heading(line) not in BOILERPLATE_HEADINGS_EXACT]
        text = "\n".join(lines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def clean_article_text(self, article_text: str) -> str:
        code = mwparserfromhell.parse(self._preprocess_raw(article_text))
        code = self._remove_boilerplate_sections(code)
        return self.clean_wikicode(code)

    def extract_lead(self, article_text: str) -> str:
        """Return the intro section (text before the first heading)."""
        code = mwparserfromhell.parse(self._preprocess_raw(article_text))
        for index, node in enumerate(code.nodes):
            if isinstance(node, Heading):
                return self.clean_wikicode(Wikicode(list(code.nodes[:index])))
        return self.clean_wikicode(code)

    def read_and_preprocess_articles(self) -> dict[int, dict]:
        """Stream article pages out of the (optionally compressed) XML dump."""
        articles: dict[int, dict] = {}
        path = self.wiki_pages_articles_xml_path
        opener = bz2.open if str(path).endswith(".bz2") else open
        with opener(path, "rb") as f:
            context = ET.iterparse(f, events=("end",))
            for _, element in tqdm(context, desc="Reading and preprocessing articles"):
                if _localname(element.tag) != "page":
                    continue

                namespace = _child_text(element, "ns")
                is_redirect = _child(element, "redirect") is not None
                if namespace != "0" or is_redirect:
                    element.clear()
                    continue

                page_id_text = _child_text(element, "id")
                if page_id_text is None:
                    element.clear()
                    continue
                page_id = int(page_id_text)
                title = _child_text(element, "title")
                revision = _child(element, "revision")
                revision_id = None
                wikitext = ""
                if revision is not None:
                    revision_id_text = _child_text(revision, "id")
                    revision_id = int(revision_id_text) if revision_id_text else None
                    text_element = _child(revision, "text")
                    if text_element is not None and text_element.text:
                        wikitext = text_element.text

                text = self.clean_article_text(wikitext)
                if text:
                    articles[page_id] = {
                        "title": title,
                        "revision_id": revision_id,
                        "text": text,
                        "lead": self.extract_lead(wikitext),
                    }
                element.clear()

        return articles

    def _direct_categories_by_article(self) -> dict[int, set[str]]:
        article_page_ids = set(
            self.page_df.loc[self.page_df["page_namespace"] == 0, "page_id"]
        )
        rows = self.categorylinks_df[self.categorylinks_df["cl_from"].isin(article_page_ids)]
        direct: dict[int, set[str]] = {}
        for from_id, target_id in zip(rows["cl_from"], rows["cl_target_id"]):
            title = self._linktarget_titles.get(target_id)
            if not title:
                continue
            title = self.canonical_title(title)
            if check_if_is_junk_cat(title):
                continue
            direct.setdefault(from_id, set()).add(title)
        return direct

    def expand_to_ancestors(self, category_titles) -> list[str]:
        """Return the given categories plus all of their ancestors, sorted."""
        result: set[str] = set()
        stack = list(category_titles)
        while stack:
            title = stack.pop()
            if title in result:
                continue
            result.add(title)
            for parent in self.parent_map.get(title, ()):
                if parent not in result:
                    stack.append(parent)
        return sorted(result)

    # -- dataset assembly ----------------------------------------------------

    def create_dataset_list(self, min_words: int = 0) -> list[dict]:
        articles = self.read_and_preprocess_articles()
        direct_by_article = self._direct_categories_by_article()

        records = []
        for page_id in sorted(articles):
            direct = direct_by_article.get(page_id)
            if not direct:
                continue
            labels = self.expand_to_ancestors(direct)
            if not labels:
                continue
            info = articles[page_id]
            num_words = len(info["text"].split())
            if min_words and num_words < min_words:
                continue
            records.append(
                {
                    "id": f"{LANGUAGE_CODE}-{page_id}",
                    "language": LANGUAGE_CODE,
                    "page_id": page_id,
                    "revision_id": info["revision_id"],
                    "title": info["title"],
                    "url": f"{WIKI_DOMAIN}/?curid={page_id}",
                    "text": info["text"],
                    "lead": info["lead"],
                    "num_words": num_words,
                    "direct_categories": sorted(direct),
                    "labels": labels,
                }
            )
        return records

    def create_dataset_json(self, dataset_json_file_path, min_words: int = 0, indent=2) -> list[dict]:
        records = self.create_dataset_list(min_words=min_words)
        with open(dataset_json_file_path, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=indent, ensure_ascii=False)
        return records

    def create_hierarchy_list(self) -> list[dict]:
        """Return the category hierarchy as ``{"parent": ..., "child": ...}`` rows."""
        return [
            {"parent": parent, "child": child}
            for parent, child in self.get_hierarchy_edges()
        ]

    def save_artifacts(self, dataset_json_path, hierarchy_json_path, min_words: int = 0, indent=2) -> list[dict]:
        """Write both the dataset records and the category hierarchy JSON files."""
        records = self.create_dataset_list(min_words=min_words)
        with open(dataset_json_path, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=indent, ensure_ascii=False)
        with open(hierarchy_json_path, "w", encoding="utf-8") as file:
            json.dump(self.create_hierarchy_list(), file, indent=indent, ensure_ascii=False)
        return records


def build_dataset(dataset_json_path, hierarchy_json_path, min_words: int = 0) -> list[dict]:
    """Convenience entry point: load the dump tables, build the dataset, and write both JSON artifacts."""
    from lrl_wiki_cls.load_wiki_tables import load_table

    creator = WikiTextClsDatasetCreator(
        wiki_pages_articles_xml_path=dump_path("pages-articles.xml.bz2"),
        categorylinks_df=load_table("categorylinks"),
        page_df=load_table("page"),
        redirect_df=load_table("redirect"),
        linktarget_df=load_table("linktarget"),
    )
    return creator.save_artifacts(dataset_json_path, hierarchy_json_path, min_words=min_words)


if __name__ == "__main__":
    import argparse

    from lrl_wiki_cls.config import RESULTS_DIR

    parser = argparse.ArgumentParser(description="Build the dataset JSON artifacts from the dumps.")
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Drop articles with fewer than this many words (default: 0, keep all).",
    )
    parser.add_argument("--dataset-json", default=str(RESULTS_DIR / "zuwiki-dataset.json"))
    parser.add_argument("--hierarchy-json", default=str(RESULTS_DIR / "zuwiki-hierarchy.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = build_dataset(args.dataset_json, args.hierarchy_json, min_words=args.min_words)
    print(f"Wrote {len(records):,} records to {args.dataset_json}")
