"""Regenerate every README asset in docs/assets/."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [
    "generate_hero.py",
    "generate_pipeline.py",
    "generate_causal.py",
    "generate_stability.py",
    "generate_component.py",
    "generate_ladder.py",
    "generate_demo_gif.py",
]

for script in SCRIPTS:
    print(f"── {script}")
    subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
print("done.")
