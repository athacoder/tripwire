"""The public demo: the dashboard over the benchmark's results, read-only.

Streamlit Community Cloud runs this file. A checkout has no results database (it is not
in git), so the packed copy made by scripts/make_demo_db.py is unpacked in its place. On
a machine that has its own tripwire.db, that one is used and nothing is overwritten.
"""

import gzip
import os
import runpy
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATABASE = ROOT / "tripwire.db"

if not DATABASE.exists():
    with gzip.open(ROOT / "demo" / "tripwire-demo.db.gz") as packed, DATABASE.open("wb") as out:
        shutil.copyfileobj(packed, out)

os.environ["TRIPWIRE_CONFIG"] = str(ROOT / "experiments" / "zoo.toml")
os.environ["TRIPWIRE_READ_ONLY"] = "1"
runpy.run_path(str(ROOT / "src" / "tripwire" / "dashboard.py"), run_name="__main__")
