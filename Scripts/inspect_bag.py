"""
This script loads a bag file, then prints all available topics contained in it.
"""

import os

from pathlib import Path
from bagpy import bagreader

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Build path relative to the root
BAG_PATH = PROJECT_ROOT / "Data" / "BAG Files" / 'E3RDAA23250004_00227_149_1flight_obcbag.bag'

if not os.path.exists(BAG_PATH):
    raise FileNotFoundError(f'BAG File Not Found:{BAG_PATH}')

# Read bag file
bag = bagreader(str(BAG_PATH))

# All topics
print('\n ---------- Topics ----------\n')
print(bag.topic_table)