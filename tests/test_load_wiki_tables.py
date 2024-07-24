import unittest
from sqlalchemy import create_engine, text
from lrl_wiki_cls.config import CONNECTION_STRING
import lrl_wiki_cls.load_wiki_tables
import pandas as pd


class TestDatabaseLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(CONNECTION_STRING)

    def test_load_database_tables(self):
        tables = lrl_wiki_cls.load_wiki_tables.load_database_tables(self.engine)
        assert tables is not None and tables.keys() == {
            "category",
            "categorylinks",
            "langlinks",
            "page",
            "redirect",
        }
        del tables
        tables = lrl_wiki_cls.load_wiki_tables.load_database_tables()
        assert tables is not None and tables.keys() == {
            "category",
            "categorylinks",
            "langlinks",
            "page",
            "redirect",
        }

    def test_db_connection(self):
        with self.engine.connect() as connection:
            connection.execute(text("show tables;"))

    def test_load_table(self):
        df = lrl_wiki_cls.load_wiki_tables.load_table("categorylinks", self.engine)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0

    def test_category_df_loading(self):
        df = pd.read_sql_table("category", self.engine)
        expected_columns = ["cat_id", "cat_title", "cat_pages", "cat_subcats"]
        self.assertTrue(
            all(column in df.columns for column in expected_columns),
            "category DataFrame does not have the expected columns.",
        )

    def test_categorylinks_df_loading(self):
        df = pd.read_sql_table("categorylinks", self.engine)
        expected_columns = ["cl_from", "cl_to", "cl_type"]
        self.assertTrue(
            all(column in df.columns for column in expected_columns),
            "categorylinks DataFrame does not have the expected columns.",
        )

    def test_redirect_df_loading(self):
        df = pd.read_sql_table("redirect", self.engine)
        expected_columns = ["rd_from", "rd_title"]
        self.assertTrue(
            all(column in df.columns for column in expected_columns),
            "redirect DataFrame does not have the expected columns.",
        )

    def test_langlinks_df_loading(self):
        df = pd.read_sql_table("langlinks", self.engine)
        expected_columns = ["ll_from", "ll_lang", "ll_title"]
        self.assertTrue(
            all(column in df.columns for column in expected_columns),
            "langlinks DataFrame does not have the expected columns.",
        )

    def test_page_df_loading(self):
        df = pd.read_sql_table("page", self.engine)
        expected_columns = ["page_id", "page_title", "page_lang", "page_len"]
        self.assertTrue(
            all(column in df.columns for column in expected_columns),
            "page DataFrame does not have the expected columns.",
        )

    def test_query_with_cat_title(self):
        df = pd.read_sql_table("category", self.engine)
        example_title = "Amazwe".encode("utf-8")
        res = df.query("cat_title==@example_title")
        self.assertTrue(len(res) >= 1)
