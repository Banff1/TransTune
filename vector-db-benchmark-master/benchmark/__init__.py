import os
from pathlib import Path

# Base directory point to the main directory of the project, so all the data
# loaded from files can refer to it as a root directory

BASE_DIRECTORY = Path(__file__).parent.parent
# Override with VDB_DATASETS_DIR if needed; default uses repo datasets/ on store1.
_DATASETS_DIR = os.environ.get("VDB_DATASETS_DIR")
DATASETS_DIR = Path(_DATASETS_DIR) if _DATASETS_DIR else BASE_DIRECTORY / "datasets"
CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = Path(os.path.dirname(CODE_DIR))
