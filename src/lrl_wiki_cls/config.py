from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = f"{PROJECT_ROOT}/data"
RESULTS_DIR = f"{PROJECT_ROOT}/results"
# Gao et al. (2021) use 16 training examples per class for their 'few-shot' scenario. Assuming an 80/20 train/test split, we need at least 16=.8*20 examples per class. 
# We will use the same number of examples for our 'few-shot' scenario.
MIN_TRAINING_ENTRIES_PER_LABEL = 16
MIN_ENTRIES_PER_LABEL = int(math.ceil(MIN_TRAINING_ENTRIES_PER_LABEL / 0.8))
username = "pythonuser"
password = "password"
host = "localhost"
port = 3306
database = "wiki_categories"
WIKI_CATEGORY_NAMESPACE = 14
# The XML namespace of each element needs to be specified in each XML ElementTree query.
NAMESPACES = {
    "mw": "http://www.mediawiki.org/xml/export-0.10/",
}
CONNECTION_STRING = (
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
)
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Connection string: {CONNECTION_STRING}")
    print(f"Wiki Category Namespace: {WIKI_CATEGORY_NAMESPACE}")
    print(f"Min Training Entries Per Label: {MIN_TRAINING_ENTRIES_PER_LABEL}")
    print(f"Min Entries Per Label: {MIN_ENTRIES_PER_LABEL}")