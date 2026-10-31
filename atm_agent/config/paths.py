from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
DATA = PROJECT / "data"
PROMPTS = PROJECT / "prompts"
