# Attribution

This project produces two separately licensed things:

| Work | License | File |
| --- | --- | --- |
| The source code in this repository | GNU General Public License v3.0 or later (`GPL-3.0-or-later`) | [`LICENSE`](LICENSE) |
| The generated dataset (text labels, category hierarchy, and any published Parquet artifacts) | Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`) | [`LICENSE-CC-BY-SA-4.0.txt`](LICENSE-CC-BY-SA-4.0.txt) |

The dataset is a **derivative work** of the Zulu Wikipedia, which is itself
published under CC BY-SA 4.0. It therefore has to be distributed under the same
license (ShareAlike) and must carry attribution (BY). The code is a separate
work and is licensed independently under the GPL.

## Source

- **Project:** Zulu Wikipedia — <https://zu.wikipedia.org>
- **Underlying dump:** a Wikimedia `zuwiki` snapshot pinned in
  `src/lrl_wiki_cls/config.py` (`DUMP_DATE`, currently `20261101`), downloaded
  from <https://dumps.wikimedia.org/zuwiki/>.
- **Original snapshot:** the first version of this dataset was derived from
  dumps dated 2024-04-01 / 2024-04-02. Wikimedia no longer serves those files,
  so rebuilds use the currently pinned snapshot and may differ.
- **Copyright:** the text is © the contributors of the Zulu Wikipedia.
- **License of the source text:** CC BY-SA 4.0
  (<https://creativecommons.org/licenses/by-sa/4.0/>).

## Changes made

In accordance with the "indicate changes" requirement of CC BY-SA 4.0, the
dataset differs from the source text as follows:

- Wikitext markup was stripped from article bodies to produce plain text, while
  preserving the display text of links.
- Boilerplate sections are removed entirely: references/sources, "see also"/
  related, external links, and image galleries. Only article prose and content
  sections remain.
- Content section headings (e.g. "History"/"Umlando") are retained as plain
  text; boilerplate headings are not.
- Templates, tables, citations, and stray HTML/XML tags were removed.
- Articles were filtered to the main (article) namespace, with redirects and
  empty pages removed.
- Categories were extracted and expanded along the category hierarchy, filtered
  to remove maintenance/"junk" categories, and labels with fewer than a minimum
  number of associated articles may be rare.
- Articles whose cleaned text or category set was empty were discarded.

No attempt is made to alter the meaning of the source text; the labels are
derived computationally and may contain errors.

## How to attribute

If you use or redistribute the dataset, please credit it approximately as
follows:

> Dataset derived from the Zulu Wikipedia (https://zu.wikipedia.org),
> © the contributors of the Zulu Wikipedia, licensed under CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/). Changes were made to the
> source text (markup removed, categories derived); see the project's
> ATTRIBUTION file for details.

Because the license is ShareAlike, any redistributed or adapted version of the
dataset must also be released under CC BY-SA 4.0 (or a compatible license).
Each published record retains its source article `title`, `page_id`, and `url`
so that individual articles can be attributed and traced back to the origin.

## Questions

If you believe content in the dataset is misattributed or should not be
included, please open an issue in this repository.
