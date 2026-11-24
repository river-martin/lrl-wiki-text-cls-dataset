# LRL Wiki Text Classification Dataset

## Exploratory Data Analysis

See `"scripts/An Exploratory Analysis of Zulu Wikipedia for Multi-label Classification.ipynb"` for an overview of the structured dataset.

## Installation (Ubuntu)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

    # Run these commands from the project's root directory.
    uv sync

`uv sync` creates a `.venv`, installs this project in editable mode (so code edits take effect immediately), and installs the exact versions recorded in `uv.lock` (including the `dev` dependency group).

`pygraphviz` needs the system Graphviz libraries before it can be built:

    sudo apt install graphviz graphviz-dev

Run anything inside the environment with `uv run`, e.g. `uv run pytest tests -s`.

To add or change a dependency, use `uv add <package>` (or `uv add --dev <package>` for development tools). This updates both `pyproject.toml` and `uv.lock` together.

## Using the processed dataset

After installing the project (by following the steps in the previous section), you should be able to run the the notebook in the `scripts/` directory. `ipykernel` is part of the `dev` group, so register a notebook kernel for the uv environment with:

    uv run python -m ipykernel install --user --name lrl-wiki-cls

You can then select the `lrl-wiki-cls` kernel in JupyterLab or VS Code, and run the notebook.

Here are some general instructions for using the dataset:

    # The code below loads the dataset consisting of the texts and the lists of categories associated with them.
    from lrl_wiki_cls.wiki_text_cls_dataset import WikiTextClsDataset
    from lrl_wiki_cls.config import RESULTS_DIR
    dataset = WikiTextClsDataset(f"{RESULTS_DIR}/zuwiki-dataset.json")
    for entry, categories in dataset:
        print(entry['text'])
        print(entry['title'])
        # categories and entry['categories'] should be the same.
        print(entry['categories'])
        print(categories)

### Getting the hierarchy graph and metadata

    from lrl_wiki_cls.structure_category_metadata import (
        load_dataset_and_cat_metadata,
        get_and_plot_hierarchy_graph,
        print_dataset_overview
    )
    dataset, cat_metadata = load_dataset_and_cat_metadata()
    # The text length is the number of characters in the text
    print_dataset_overview(dataset)
    networkx_digraph = get_and_plot_hierarchy_graph(dataset, cat_metadata)

## Execution and testing (for the processing of the raw data)

The instructions in this section are only necessary to recreate the dataset from the raw data.

    # Start the database server
    sudo service mysql start

    # If you are going to run the project from scratch, decompress the dump files with the command below.
    uv run python -m lrl_wiki_cls.decompress_dumps

### Database setup from MySQL dump files

    # Create a database and load the mysql dump files
    mysql -u yourusername -p

    mysql> create database wiki_categories;

    mysql> source results/zuwiki-latest-category.sql; 
    mysql> source results/zuwiki-latest-categorylinks.sql;
    mysql> source results/zuwiki-latest-langlinks.sql;
    mysql> source results/zuwiki-latest-page.sql;

    # Create a database user with permission to access the local db from Python (the user's password is password)
    mysql> create user 'pythonuser'@'localhost' identified by 'password';
    mysql> grant all privileges on wiki_categories.* to 'pythonuser'@'localhost';
    mysql> flush privileges;

    # Change the type of the cl_type column so that it can be read with pd.read_sql_table()
    mysql> ALTER TABLE categorylinks ADD COLUMN casted_cl_type char(6);
    mysql> UPDATE categorylinks SET temp_age = CAST(cl_type as char(6));
    mysql> ALTER TABLE categorylinks DROP COLUMN cl_type;
    mysql> ALTER TABLE categorylinks CHANGE COLUMN casted_cl_type cl_type char(6);
    mysql> exit;

### Producing the structured dataset

    # If you only want to generate the final dataset:
    uv run pytest tests tests/test_wiki_text_cls_dataset.py::TestWikiTextClsDatasetCreator::test_create_dataset_json -s

### Running the tests

    # Run the tests (Note that the test for the TextWikiClsDatasetCreator class takes quite a while)
    # The flag `-s` is used so that the progress bars (made with tqdm in the functions being tested) are printed as a test runs.
    uv run pytest tests -s

## Notes

The Zulu part of the dataset is from the First and the Second of April, 2024
