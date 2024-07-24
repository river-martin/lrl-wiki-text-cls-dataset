from lrl_wiki_cls.config import NAMESPACES
import xml.etree.ElementTree as ET
import numpy as np

def randomly_sample_xml_file(xml_file, sampled_xml_file, tag_to_sample, num_samples, seed=None):
    """
    Randomly sample an XML file and save the sampled XML file (used to make a smaller dataset for faster testing).

    Returns:
    None
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if seed:
        np.random.seed(seed)
    all_pages = root.findall(tag_to_sample, namespaces=NAMESPACES)
    sampled_page_indices = np.random.choice(
        len(all_pages), num_samples, replace=False
    )
    for i, page in enumerate(all_pages):
        if i not in sampled_page_indices:
            root.remove(page)
    ET.register_namespace('', NAMESPACES['mw'])
    tree.write(sampled_xml_file, encoding="utf-8")
    