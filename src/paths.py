import pathlib

file_path = pathlib.Path(__file__)
ROOT = file_path.parent
DATA_DIR = ROOT / "data"
IAM_DIR = DATA_DIR / "IAM_HW"
IAM_WORDS_DIR = IAM_DIR / 'words'
IAM_XML_DIR = IAM_DIR / "xml"
IAM_RULE_DIR = IAM_DIR / "rules"
RUNS = ROOT / 'runs'