from lrl_wiki_cls.config import NAMESPACES, RESULTS_DIR
import xml.etree.ElementTree as ET
import pytest
from lrl_wiki_cls.randomly_sample_xml_file import randomly_sample_xml_file


def test_randomly_sample_xml_file():
    """Tests that the function samples the XML file correctly. 
    Specifically ensuring that the tags used in the output file are the same as the tags used in the input file."""
    xml_file = f"{RESULTS_DIR}/zuwiki-latest-pages-articles.xml"
    sampled_xml_file = f"{RESULTS_DIR}/zuwiki-latest-pages-articles-sampled.xml"
    num_pages_to_keep = 500
    tag_to_sample = "mw:page"
    randomly_sample_xml_file(xml_file, sampled_xml_file, tag_to_sample, num_pages_to_keep, seed=42)
    tree = ET.parse(sampled_xml_file)
    root = tree.getroot()
    pages = root.findall(tag_to_sample, namespaces=NAMESPACES)
    assert len(pages) == num_pages_to_keep
    for page in pages:
        # Check that the page tags use the correct namespace
        assert page.tag == f"{{{NAMESPACES['mw']}}}page"
        assert page.find("mw:id", NAMESPACES) is not None
        assert page.find("mw:title", NAMESPACES) is not None
        assert page.find("mw:revision/mw:text", NAMESPACES) is not None
        assert page.find("mw:revision/mw:text", NAMESPACES).text is not None
        assert len(page.find("mw:revision/mw:text", NAMESPACES).text) > 0
        break
