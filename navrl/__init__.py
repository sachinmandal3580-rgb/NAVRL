"""NavRL: PPO-based navigation among obstacles for a differential-drive robot in MuJoCo."""
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ARENA_XML = ASSET_DIR / "arena.xml"
