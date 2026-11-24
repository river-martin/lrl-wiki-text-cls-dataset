import unittest

import lrl_wiki_cls


class TestProjectPackageImport(unittest.TestCase):
    def test_project_package_import(self):
        """Test that the project package can be imported."""
        assert lrl_wiki_cls.__name__ == "lrl_wiki_cls"
