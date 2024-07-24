def check_if_is_junk_cat(cat_title):
    assert isinstance(cat_title, str)
    junk_prefixes = [
        "Stub",
        "Webarchive",
        "Articles_with",
        "CS1",
        "Template",
        "All_articles",
        "Infobox",
        "Candidates",
        "Pages_",
        "IPA_templates",
        'User_'
    ]
    junk_suffixes = ['_templates', '_template', '_stubs']
    for prefix in junk_prefixes:
        if cat_title.startswith(prefix):
            return True
    for suffix in junk_suffixes:
        if cat_title.endswith(suffix):
            return True
    return False
