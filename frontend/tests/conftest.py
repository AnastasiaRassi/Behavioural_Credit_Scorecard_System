"""Lets the frontend's flat imports (`import api`, `import review`) work under pytest.

Run from the backend folder, like the rest of the suite:
    cd backend && python -m pytest ../frontend/tests
"""
import os
import sys
from pathlib import Path

# Before `api` is imported: its file log then goes to logs/testing/.
os.environ["APP_ENV"] = "test"

FRONTEND = Path(__file__).resolve().parents[1]
if str(FRONTEND) not in sys.path:
    sys.path.append(str(FRONTEND))
