"""Project-wide configuration and paths.

Paths are derived from the location of this file so the package works
regardless of the current working directory.
"""

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
# Parsed SQL tables are cached here so repeated runs don't re-parse the dumps.
CACHE_DIR = DATA_DIR / "cache"

# ---------------------------------------------------------------------------
# Wikimedia dump source.
#
# The dump date is pinned (rather than using ``latest``) so a build is
# reproducible. Wikimedia only keeps recent dumps online; when the pinned date
# is purged, update ``DUMP_DATE`` to a date listed at
# https://dumps.wikimedia.org/zuwiki/ and rebuild.
# ---------------------------------------------------------------------------
WIKI = "zuwiki"
LANGUAGE_CODE = "zu"
WIKI_DOMAIN = "https://zu.wikipedia.org"
DUMP_DATE = "20261101"
DUMP_BASE_URL = f"https://dumps.wikimedia.org/{WIKI}/{DUMP_DATE}"

# Files needed to build the dataset, in download order. ``category`` and
# ``langlinks`` are optional but cheap and keep the table loader complete.
DUMP_FILES = (
    "pages-articles.xml.bz2",
    "categorylinks.sql.gz",
    "linktarget.sql.gz",
    "page.sql.gz",
    "redirect.sql.gz",
    "category.sql.gz",
    "langlinks.sql.gz",
)


def dump_filename(name: str) -> str:
    """Return the dated filename for a dump component, e.g. ``zuwiki-2026...-page.sql.gz``."""
    return f"{WIKI}-{DUMP_DATE}-{name}"


def dump_url(name: str) -> str:
    return f"{DUMP_BASE_URL}/{dump_filename(name)}"


def dump_path(name: str) -> Path:
    return DATA_DIR / dump_filename(name)


# Wikipedia categories live in namespace 14.
WIKI_CATEGORY_NAMESPACE = 14

# Gao et al. (2021) use 16 training examples per class for their 'few-shot'
# scenario. Assuming an 80/20 train/test split, we need at least 16/0.8 = 20
# examples per class. Published datasets are not filtered by default; this
# threshold is available as the documented recipe for a model-ready subset.
MIN_TRAINING_ENTRIES_PER_LABEL = 16
MIN_ENTRIES_PER_LABEL = math.ceil(MIN_TRAINING_ENTRIES_PER_LABEL / 0.8)

if __name__ == "__main__":
    print(f"Project Root:   {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Dir:    {RESULTS_DIR}")
    print(f"Cache Dir:      {CACHE_DIR}")
    print(f"Dump date:      {DUMP_DATE}")
    print(f"Dump base URL:  {DUMP_BASE_URL}")
    print(f"Min entries/label: {MIN_ENTRIES_PER_LABEL}")
