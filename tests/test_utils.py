import pytest

from lrl_wiki_cls.utils import check_if_is_junk_cat

MAINTENANCE_CATEGORIES = [
    "Stubs",
    "Webarchive_template_wayback_links",
    "Articles_containing_Afrikaans-language_text",
    "Articles_needing_additional_references_from_July_2019",
    "CS1_maint:_archived_copy_as_title",
    "All_article_disambiguation_pages",
    "Infobox_templates",
    "Candidates_for_speedy_deletion",
    "Pages_with_broken_file_links",
    "IPA_templates",
    "User_pages",
    "Wikipedia_articles_needing_cleanup_after_translation",
    "Coordinates_on_Wikidata",
    "Ill-formatted_IPAc-en_transclusions",
    "Short_description_is_different_from_Wikidata",
    "Use_dmy_dates_from_November_2015",
    "Commons_category_with_local_link_different_from_Wikidata",
    "Harv_and_Sfn_no-target_errors",
    "EngvarB_from_March_2017",
    "Izigaba_ze-Wikipedia_eziqanjwe_ngamazwe",
]

TOPICAL_CATEGORIES = [
    "Amazwe",
    "Umlando",
    "Izilimi_zaseNingizimu_Afrika",
    "Amazwe_e-Afrika",
    "Amakhosi_akwaZulu",
    "Wikipedia",  # a topic category on the Zulu wiki's main page
]


@pytest.mark.parametrize("title", MAINTENANCE_CATEGORIES)
def test_maintenance_categories_are_junk(title):
    assert check_if_is_junk_cat(title)


@pytest.mark.parametrize("title", TOPICAL_CATEGORIES)
def test_topical_categories_are_not_junk(title):
    assert not check_if_is_junk_cat(title)
