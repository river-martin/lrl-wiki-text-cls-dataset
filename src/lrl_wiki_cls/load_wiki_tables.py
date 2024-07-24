import pandas as pd
from sqlalchemy import create_engine
from lrl_wiki_cls.config import CONNECTION_STRING


def load_database_tables(engine=None):
    """
    Warning: the columns that contain strings actually contain raw bytes. Ensure that queries are formatted appropriately, e.g. cat_df.query('cat_title==@some_str.encode("utf-8")')
    """
    if engine is None:
        engine = create_engine(CONNECTION_STRING)
    tables = ["category", "categorylinks", "langlinks", "page", "redirect"]
    dfs = {}
    for table in tables:
        dfs[table] = load_table(table, engine)
    return dfs


def load_table(table_name, engine=None):
    """
    Warning: the columns that contain strings actually contain raw bytes. Ensure that queries are formatted appropriately, e.g. cat_df.query('cat_title==@some_str.encode("utf-8")')
    """
    if engine is None:
        engine = create_engine(CONNECTION_STRING)
    return pd.read_sql_table(table_name, engine)
