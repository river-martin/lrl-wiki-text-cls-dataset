import json

from lrl_wiki_cls.structure_category_metadata import (
    CategoryMetadata,
    get_initial_hierarchy_metadata_and_save_to_json_files,
)
from lrl_wiki_cls.wiki_text_cls_dataset import WikiTextClsDataset


def _record(page_id, labels):
    return {
        "id": f"zu-{page_id}",
        "language": "zu",
        "page_id": page_id,
        "revision_id": page_id * 10,
        "title": f"T{page_id}",
        "url": f"https://zu.wikipedia.org/?curid={page_id}",
        "text": "some text",
        "lead": "some text",
        "num_words": 2,
        "direct_categories": labels[:1],
        "labels": labels,
    }


def test_metadata_ids_links_and_roundtrip(tmp_path):
    dataset = WikiTextClsDataset(json_dict=[_record(1, ["Amazwe", "IAfrika", "Izilimi"])])
    edges = [
        {"parent": "Amazwe", "child": "IAfrika"},
        {"parent": "IAfrika", "child": "Izilimi"},
    ]

    title_to_id = tmp_path / "title_to_id.json"
    id_to_title = tmp_path / "id_to_title.json"
    subcat_links = tmp_path / "subcat_links.json"
    get_initial_hierarchy_metadata_and_save_to_json_files(
        dataset, edges, title_to_id, id_to_title, subcat_links
    )

    metadata = CategoryMetadata(title_to_id, id_to_title, subcat_links)
    assert set(metadata.get_category_titles()) == {"Amazwe", "IAfrika", "Izilimi"}
    for title in metadata.get_category_titles():
        assert metadata.get_title(metadata.get_id(title)) == title
        assert isinstance(metadata.get_id(title), int)

    assert metadata.get_subcats_of("IAfrika") == ["Izilimi"]
    assert metadata.get_supcats_of("Izilimi") == ["IAfrika"]
    assert metadata.get_subcats_of("Amazwe") == ["IAfrika"]

    saved_links = json.loads(subcat_links.read_text())
    assert all(isinstance(link, list) and len(link) == 2 for link in saved_links)


def test_metadata_skips_junk_and_labels_outside_dataset(tmp_path):
    dataset = WikiTextClsDataset(json_dict=[_record(1, ["Amazwe", "IAfrika"])])
    edges = [
        {"parent": "Amazwe", "child": "IAfrika"},
        {"parent": "Amazwe", "child": "Stubs"},
        {"parent": "Amazwe", "child": "NotInDataset"},
    ]
    title_to_id = tmp_path / "t.json"
    id_to_title = tmp_path / "i.json"
    subcat_links = tmp_path / "s.json"
    get_initial_hierarchy_metadata_and_save_to_json_files(
        dataset, edges, title_to_id, id_to_title, subcat_links
    )
    metadata = CategoryMetadata(title_to_id, id_to_title, subcat_links)
    assert "Stubs" not in metadata.get_category_titles()
    assert "NotInDataset" not in metadata.get_category_titles()
    assert metadata.get_subcats_of("Amazwe") == ["IAfrika"]
