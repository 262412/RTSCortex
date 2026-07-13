from __future__ import annotations

import sys
from pathlib import Path

BRIDGE_SRC = Path(__file__).parents[1] / "integrations" / "llm_pysc2" / "src"
sys.path.insert(0, str(BRIDGE_SRC))
