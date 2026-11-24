from lrl_wiki_cls.publish_hf_dataset import (
    build_dataset_card,
    build_hierarchy_rows,
    iterative_stratified_split,
    merge_readme,
    split_front_matter,
)


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


def _records(n=20):
    records = []
    for i in range(n):
        labels = ["common"]
        if i < 10:
            labels.append("half")
        if i < 4:
            labels.append("rare")
        records.append(_record(i, labels))
    return records


def test_split_partitions_all_records_and_respects_ratios():
    records = _records(20)
    splits = iterative_stratified_split(records, ratios=(0.8, 0.1, 0.1), seed=1)
    assert set(splits) == {"train", "validation", "test"}
    all_ids = [r["page_id"] for rows in splits.values() for r in rows]
    assert sorted(all_ids) == list(range(20))  # every record exactly once
    assert len(splits["train"]) == 16
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2


def test_split_keeps_common_label_in_train():
    records = _records(20)
    splits = iterative_stratified_split(records, ratios=(0.8, 0.1, 0.1), seed=1)
    # The ubiquitous label should still be well represented in train.
    train_common = sum("common" in r["labels"] for r in splits["train"])
    assert train_common >= 12


def test_split_is_deterministic_for_a_seed():
    records = _records(20)
    a = iterative_stratified_split(records, seed=7)
    b = iterative_stratified_split(records, seed=7)
    assert [r["page_id"] for r in a["train"]] == [r["page_id"] for r in b["train"]]


def test_build_hierarchy_rows_depth_parents_and_counts():
    records = [
        _record(1, ["Amazwe", "IAfrika"]),
        _record(2, ["Amazwe", "IAfrika"]),
        _record(3, ["Amazwe"]),
    ]
    edges = [
        {"parent": "Amazwe", "child": "IAfrika"},
        {"parent": "IAfrika", "child": "Izilimi"},  # child not in dataset labels
    ]
    rows = {row["title"]: row for row in build_hierarchy_rows(records, edges)}
    assert set(rows) == {"Amazwe", "IAfrika"}
    assert rows["Amazwe"]["depth"] == 0
    assert rows["Amazwe"]["parent_titles"] == []
    assert rows["IAfrika"]["depth"] == 1
    assert rows["IAfrika"]["parent_titles"] == ["Amazwe"]
    assert rows["Amazwe"]["num_documents"] == 3
    assert rows["IAfrika"]["num_documents"] == 2


def test_dataset_card_contains_license_and_stats():
    records = _records(4)
    splits = iterative_stratified_split(records, seed=1)
    card = build_dataset_card(records, splits, repo_id="someone/somewhere")
    assert "license: cc-by-sa-4.0" in card
    assert "language:" in card and "- zu" in card
    assert "CC BY-SA 4.0" in card
    assert "Documents: **4**" in card
    assert "Changes made" in card
    assert "external-links" in card
    assert "removed entirely" in card


def test_split_front_matter():
    yaml_text, body = split_front_matter("---\nlicense: mit\n---\n\nBody")
    assert yaml_text is not None and "license: mit" in yaml_text
    assert body.strip() == "Body"
    assert split_front_matter("No front matter") == (None, "No front matter")


def test_merge_readme_preserves_dataset_info_and_updates_metadata():
    existing = (
        "---\nlicense: mit\ndataset_info:\n  configs:\n    default: {}\n---\n\n"
    )
    merged = merge_readme(
        existing, "BODY", {"license": "cc-by-sa-4.0", "language": ["zu"]}
    )
    assert "dataset_info" in merged  # Hub-generated config info kept
    assert "cc-by-sa-4.0" in merged  # our metadata overrides
    assert "license: mit" not in merged
    assert merged.rstrip().endswith("BODY")
