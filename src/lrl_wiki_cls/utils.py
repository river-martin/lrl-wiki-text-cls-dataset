def check_if_is_junk_cat(cat_title):
    """Return True for maintenance/tracking categories that are not topics.

    MediaWiki and templates add these automatically (e.g. "Articles containing
    X-language text", "Coordinates on Wikidata", "Use dmy dates"), so they must
    not be used as classification labels.
    """
    assert isinstance(cat_title, str)
    junk_prefixes = [
        "Stub",
        "Webarchive",
        "Articles_",
        "CS1",
        "Template",
        "All_",
        "Infobox",
        "Candidates",
        "Pages_",
        "IPA",
        "User_",
        "Wikipedia_",
        "Coordinates_on_Wikidata",
        "Ill-formatted",
        "Short_description",
        "Use_",
        "Commons_category",
        "Harv",
        "Engvar",
        "Izigaba_ze-Wikipedia",
    ]
    junk_suffixes = ["_templates", "_template", "_stubs"]
    for prefix in junk_prefixes:
        if cat_title.startswith(prefix):
            return True
    for suffix in junk_suffixes:
        if cat_title.endswith(suffix):
            return True
    return False
