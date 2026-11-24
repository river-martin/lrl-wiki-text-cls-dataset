import pytest

from lrl_wiki_cls.config import (
    DUMP_DATE,
    DUMP_FILES,
    dump_filename,
    dump_path,
    dump_url,
)


def test_dump_names_are_dated_and_consistent():
    for name in DUMP_FILES:
        filename = dump_filename(name)
        assert filename == f"zuwiki-{DUMP_DATE}-{name}"
        assert dump_url(name).endswith(filename)
        assert dump_path(name).name == filename


def test_required_dumps_are_listed():
    assert "pages-articles.xml.bz2" in DUMP_FILES
    assert "categorylinks.sql.gz" in DUMP_FILES
    assert "linktarget.sql.gz" in DUMP_FILES
    assert "page.sql.gz" in DUMP_FILES
    assert "redirect.sql.gz" in DUMP_FILES


def test_downloaded_dumps_if_present():
    missing = [name for name in DUMP_FILES if not dump_path(name).exists()]
    if missing:
        pytest.skip(f"raw dumps not downloaded: {missing}")
    for name in DUMP_FILES:
        assert dump_path(name).stat().st_size > 0
