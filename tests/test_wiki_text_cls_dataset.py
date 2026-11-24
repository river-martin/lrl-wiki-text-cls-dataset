import bz2
import textwrap

import pandas as pd
import pytest

from lrl_wiki_cls.config import DATA_DIR
from lrl_wiki_cls.wiki_text_cls_dataset import (
    WikiTextClsDataset,
    WikiTextClsDatasetCreator,
    count_labels,
    filter_records_by_label_count,
)

MEDIAWIKI_XML = textwrap.dedent(
    """\
    <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11">
      <page>
        <title>UZulu</title>
        <ns>0</ns>
        <id>1</id>
        <revision>
          <id>100</id>
          <text>UZulu [[ulimi|ulimi]] wamaZulu.
    {{DISPLAYTITLE:x}}
    {{#if:1|ignored}}

    == Isigaba ==
    Umzimba.</text>
        </revision>
      </page>
      <page>
        <title>IsiZulu</title>
        <ns>0</ns>
        <id>2</id>
        <revision>
          <id>101</id>
          <text>IsiZulu siyisilimi. &lt;ref&gt;citation&lt;/ref&gt;</text>
        </revision>
      </page>
      <page>
        <title>Izilimi</title>
        <ns>14</ns>
        <id>3</id>
        <revision><id>102</id><text>Izilimi</text></revision>
      </page>
      <page>
        <title>Isiqondiso</title>
        <ns>0</ns>
        <id>4</id>
        <redirect title="UZulu" />
        <revision><id>103</id><text>#REDIRECT [[UZulu]]</text></revision>
      </page>
    </mediawiki>
    """
)


@pytest.fixture()
def creator(tmp_path):
    xml_path = tmp_path / "dump.xml.bz2"
    xml_path.write_bytes(bz2.compress(MEDIAWIKI_XML.encode("utf-8")))

    page_df = pd.DataFrame(
        {
            "page_id": [1, 2, 3, 4, 5, 6],
            "page_namespace": [0, 0, 14, 0, 14, 14],
            "page_title": ["UZulu", "IsiZulu", "Izilimi", "Isiqondiso", "IAfrika", "Amazwe"],
            "page_is_redirect": [0, 0, 0, 1, 0, 0],
        }
    )
    linktarget_df = pd.DataFrame(
        {
            "lt_id": [10, 11, 12],
            "lt_namespace": [14, 14, 14],
            "lt_title": ["Izilimi", "IAfrika", "Amazwe"],
        }
    )
    categorylinks_df = pd.DataFrame(
        {
            "cl_from": [1, 2, 3, 5],
            "cl_type": ["page", "page", "subcat", "subcat"],
            "cl_target_id": [10, 10, 11, 12],
        }
    )
    redirect_df = pd.DataFrame({"rd_from": [4], "rd_title": ["UZulu"]})
    return WikiTextClsDatasetCreator(
        wiki_pages_articles_xml_path=xml_path,
        categorylinks_df=categorylinks_df,
        page_df=page_df,
        redirect_df=redirect_df,
        linktarget_df=linktarget_df,
    )


class TestDataset:
    def test_loads_json_and_iterates(self):
        dataset = WikiTextClsDataset(json_file=f"{DATA_DIR}/sample_data.json")
        assert len(dataset) == 2
        entry, labels = dataset[0]
        assert entry["text"]
        assert labels == entry["labels"]
        assert labels == ["CategoryName1", "CategoryName2"]
        assert [e["title"] for e, _ in dataset] == ["Article title0", "Article title1"]

    def test_count_labels_and_filter(self):
        records = [
            {"labels": ["a", "b"]},
            {"labels": ["a"]},
            {"labels": ["a"]},
        ]
        assert count_labels(records) == {"a": 3, "b": 1}
        filtered = filter_records_by_label_count(records, min_entries_per_label=2)
        assert filtered == [{"labels": ["a"]}, {"labels": ["a"]}, {"labels": ["a"]}]


class TestCreator:
    def test_cleaning_preserves_link_text_and_drops_markup(self, creator):
        cleaned = creator.clean_article_text(
            "Siyabonga [[ix|kakhulu]] <ref>x</ref> [[File:A.png|thumb]] yes."
        )
        assert "kakhulu" in cleaned
        assert "ix" not in cleaned
        assert "File:" not in cleaned
        assert "ref" not in cleaned
        assert "{{" not in cleaned

    def test_boilerplate_sections_removed_but_content_headings_kept(self, creator):
        text = (
            "Intro sentence.\n\n"
            "== Imithombo ==\n* [[Isihloko esihlobene]]\n\n"
            "== Izixhumanisi zangaphandle ==\n* [https://example.com iwebhusayithi]\n\n"
            "== Umlando ==\nUmlando omuhle kakhulu."
        )
        cleaned = creator.clean_article_text(text)
        assert "Imithombo" not in cleaned
        assert "Izixhumanisi" not in cleaned
        assert "Isihloko esihlobene" not in cleaned  # whole see-also section dropped
        assert "iwebhusayithi" not in cleaned  # external-link label dropped
        assert "Umlando" in cleaned  # content heading kept
        assert "Umlando omuhle kakhulu." in cleaned
        assert cleaned.startswith("Intro sentence.")

    def test_stray_tags_removed(self, creator):
        assert creator.clean_article_text("<nowiki>") == ""
        assert creator.clean_article_text("Text <nowiki> keeps") == "Text keeps"

    def test_lead_is_text_before_first_heading(self, creator):
        lead = creator.extract_lead("Intro [[a|shown]].\n\n== Heading ==\nBody")
        assert lead == "Intro shown."

    def test_parent_map_and_ancestor_expansion(self, creator):
        assert creator.parent_map["Izilimi"] == {"IAfrika"}
        assert creator.parent_map["IAfrika"] == {"Amazwe"}
        assert creator.expand_to_ancestors({"Izilimi"}) == ["Amazwe", "IAfrika", "Izilimi"]
        assert ("Amazwe", "IAfrika") in creator.get_hierarchy_edges()

    def test_read_and_preprocess_skips_non_articles_and_redirects(self, creator):
        articles = creator.read_and_preprocess_articles()
        assert set(articles) == {1, 2}
        assert articles[1]["revision_id"] == 100
        assert articles[2]["text"] == "IsiZulu siyisilimi."

    def test_dataset_records(self, creator):
        records = creator.create_dataset_list()
        by_id = {record["page_id"]: record for record in records}
        assert set(by_id) == {1, 2}

        first = by_id[1]
        assert first["id"] == "zu-1"
        assert first["language"] == "zu"
        assert first["url"] == "https://zu.wikipedia.org/?curid=1"
        assert first["direct_categories"] == ["Izilimi"]
        assert first["labels"] == ["Amazwe", "IAfrika", "Izilimi"]
        assert first["num_words"] > 0
        assert "ulimi" in first["text"]
        assert "DISPLAYTITLE" not in first["text"]
        assert first["lead"].startswith("UZulu ulimi")

    def test_min_words_filter(self, creator):
        unfiltered = creator.create_dataset_list()
        assert {record["page_id"] for record in unfiltered} == {1, 2}
        # Article 2 is only two words; article 1 has five.
        filtered = creator.create_dataset_list(min_words=3)
        assert {record["page_id"] for record in filtered} == {1}
        assert creator.create_dataset_list(min_words=99) == []

    def test_save_artifacts(self, creator, tmp_path):
        import json

        dataset_path = tmp_path / "dataset.json"
        hierarchy_path = tmp_path / "hierarchy.json"
        records = creator.save_artifacts(dataset_path, hierarchy_path)
        assert len(records) == 2
        assert json.loads(dataset_path.read_text())[0]["id"] == "zu-1"
        edges = json.loads(hierarchy_path.read_text())
        assert {"parent": "Amazwe", "child": "IAfrika"} in edges
