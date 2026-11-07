"""Lets the frontend's flat imports (`import api`, `import review`) work under pytest.

Run from the backend folder, like the rest of the suite:
    cd backend && python -m pytest ../frontend/tests
"""
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1]
if str(FRONTEND) not in sys.path:
    sys.path.append(str(FRONTEND))
