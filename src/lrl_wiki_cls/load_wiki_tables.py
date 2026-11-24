"""Load Wikimedia SQL table dumps into pandas DataFrames, without a database.

Wikimedia publishes each table as a gzipped MySQL dump containing ``CREATE
TABLE`` and extended ``INSERT`` statements. Rather than standing up a MySQL
server and importing the dumps, this module streams/parses the dumps directly
into DataFrames and caches the result as a pickle for fast repeated access.

Only the small subset of MySQL dump syntax that Wikimedia emits is supported:
single-quoted strings with backslash escapes, ``NULL``, integer and float
literals. That is sufficient for the Wikimedia table dumps used here.
"""

from __future__ import annotations

import gzip
import pickle
import re
from pathlib import Path

import pandas as pd

from lrl_wiki_cls.config import CACHE_DIR, DUMP_DATE, WIKI, dump_path

# Tables loaded by the pipeline. ``linktarget`` resolves ``categorylinks.cl_target_id``
# to a title in the modern (2025+) Wikimedia schema.
TABLES = ("category", "categorylinks", "langlinks", "page", "redirect", "linktarget")

_CREATE_TABLE_RE = re.compile(r"^CREATE TABLE `([^`]+)`")
_COLUMN_RE = re.compile(r"^\s{2}`([^`]+)`")

# MySQL backslash escapes used by mysqldump.
_ESCAPE_MAP = {
    "0": "\0",
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "\\": "\\",
    "'": "'",
    '"': '"',
}


def parse_create_tables(text: str) -> dict[str, list[str]]:
    """Return a mapping of table name -> ordered column names from CREATE TABLE blocks."""
    tables: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if current is None:
            match = _CREATE_TABLE_RE.match(line)
            if match:
                name = match.group(1)
                current = name
                tables[name] = []
            continue
        if line.startswith(")"):
            current = None
            continue
        match = _COLUMN_RE.match(line)
        if match:
            tables[current].append(match.group(1))
    return tables


def _iter_tuples(text: str, pos: int):
    """Yield value tuples from an INSERT ``VALUES`` clause starting at ``pos`` until ``;``."""
    n = len(text)

    def skip_ws(i: int) -> int:
        while i < n and text[i] in " \t\r\n":
            i += 1
        return i

    while True:
        pos = skip_ws(pos)
        if pos >= n or text[pos] == ";":
            return
        if text[pos] == ",":
            pos += 1
            continue
        if text[pos] != "(":
            raise ValueError(f"Malformed INSERT at offset {pos}: expected '(' or ';', got {text[pos]!r}")

        pos += 1
        row: list = []
        while True:
            pos = skip_ws(pos)
            if pos >= n:
                raise ValueError("Unexpected end of input inside INSERT tuple")
            if text[pos] == "'":
                pos += 1
                buf: list[str] = []
                while True:
                    char = text[pos]
                    if char == "\\":
                        escaped = text[pos + 1]
                        buf.append(_ESCAPE_MAP.get(escaped, escaped))
                        pos += 2
                    elif char == "'":
                        # '' is an escaped single quote, otherwise the string ends here.
                        if pos + 1 < n and text[pos + 1] == "'":
                            buf.append("'")
                            pos += 2
                        else:
                            pos += 1
                            break
                    else:
                        buf.append(char)
                        pos += 1
                row.append("".join(buf))
            else:
                start = pos
                while pos < n and text[pos] not in ",)":
                    pos += 1
                token = text[start:pos].strip()
                if token.upper() == "NULL":
                    row.append(None)
                elif re.fullmatch(r"-?\d+", token):
                    row.append(int(token))
                else:
                    try:
                        row.append(float(token))
                    except ValueError:
                        row.append(token)

            pos = skip_ws(pos)
            if pos >= n:
                raise ValueError("Unexpected end of input inside INSERT tuple")
            if text[pos] == ",":
                pos += 1
                continue
            if text[pos] == ")":
                pos += 1
                break
            raise ValueError(f"Malformed INSERT at offset {pos}: expected ',' or ')', got {text[pos]!r}")

        yield tuple(row)


def parse_insert_rows(text: str, table: str):
    """Yield all rows inserted into ``table`` by the dump's INSERT statements."""
    pattern = re.compile(rf"INSERT INTO `{re.escape(table)}` VALUES")
    for match in pattern.finditer(text):
        yield from _iter_tuples(text, match.end())


def parse_sql_dump(path: Path, table: str) -> pd.DataFrame:
    """Parse a single ``.sql.gz`` dump into a DataFrame for ``table``."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()

    columns = parse_create_tables(text).get(table)
    if not columns:
        raise ValueError(f"No CREATE TABLE statement for {table!r} in {path.name}")

    rows = list(parse_insert_rows(text, table))
    df = pd.DataFrame.from_records(rows, columns=columns)
    return df


def _cache_path(table: str) -> Path:
    return CACHE_DIR / f"{WIKI}-{DUMP_DATE}-{table}.pkl"


def load_table(table_name: str, use_cache: bool = True) -> pd.DataFrame:
    """Load a table, using (and populating) the on-disk pickle cache when possible."""
    dump_file = dump_path(f"{table_name}.sql.gz")
    if not dump_file.exists():
        raise FileNotFoundError(
            f"{dump_file} is missing. Run `python -m lrl_wiki_cls.download_dumps` first."
        )

    cache_file = _cache_path(table_name)
    if (
        use_cache
        and cache_file.exists()
        and cache_file.stat().st_mtime >= dump_file.stat().st_mtime
    ):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    df = parse_sql_dump(dump_file, table_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return df


def load_database_tables(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Load every table the pipeline needs."""
    return {table: load_table(table, use_cache=use_cache) for table in TABLES}


if __name__ == "__main__":
    tables = load_database_tables()
    for name, df in tables.items():
        print(f"{name:14} rows={len(df):>8,} columns={list(df.columns)}")
