from lrl_wiki_cls.load_wiki_tables import parse_create_tables, parse_insert_rows

SQL = """
CREATE TABLE `page` (
  `page_id` int(8) unsigned NOT NULL AUTO_INCREMENT,
  `page_namespace` int(11) NOT NULL DEFAULT 0,
  `page_title` varbinary(255) NOT NULL DEFAULT '',
  `page_is_redirect` tinyint(1) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`page_id`)
) ENGINE=InnoDB DEFAULT CHARSET=binary;

CREATE TABLE `redirect` (
  `rd_from` int(8) unsigned NOT NULL DEFAULT 0,
  `rd_title` varbinary(255) NOT NULL DEFAULT '',
  PRIMARY KEY (`rd_from`)
) ENGINE=InnoDB DEFAULT CHARSET=binary;

INSERT INTO `page` VALUES
(1,0,'UZulu',0),
(2,14,'Izilimi',1),
(3,0,'I\\'quote',NULL);

INSERT INTO `redirect` VALUES
(2,'Izilimi_zeAfrika');
"""


def test_parse_create_tables_returns_columns():
    tables = parse_create_tables(SQL)
    assert tables["page"] == ["page_id", "page_namespace", "page_title", "page_is_redirect"]
    assert tables["redirect"] == ["rd_from", "rd_title"]


def test_parse_insert_rows_handles_escapes_int_null_and_unicode():
    rows = list(parse_insert_rows(SQL, "page"))
    assert rows == [(1, 0, "UZulu", 0), (2, 14, "Izilimi", 1), (3, 0, "I'quote", None)]


def test_parse_insert_rows_ignores_other_tables():
    rows = list(parse_insert_rows(SQL, "redirect"))
    assert rows == [(2, "Izilimi_zeAfrika")]


def test_parse_insert_rows_handles_embedded_newlines():
    sql = "INSERT INTO `t` VALUES\n(1,'a\\nb'),\n(2,'c');"
    rows = list(parse_insert_rows(sql, "t"))
    assert rows == [(1, "a\nb"), (2, "c")]


def test_parse_insert_rows_handles_doubled_quotes():
    sql = "INSERT INTO `t` VALUES (1,'it''s');"
    assert list(parse_insert_rows(sql, "t")) == [(1, "it's")]
