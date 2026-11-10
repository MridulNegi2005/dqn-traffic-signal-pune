from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "processed" / "pune_5min.csv"
SUMO_DIR = ROOT / "sumo"
NET_FILE = SUMO_DIR / "cross.net.xml"
VTYPE_FILE = SUMO_DIR / "vtypes.add.xml"

CLASSES = ["car", "motorbike", "bus", "truck"]

# IRC:106-1990 passenger car units for urban roads
PCU = {"car": 1.0, "motorbike": 0.5, "bus": 2.2, "truck": 2.2}

# Jehangir Chowk cameras mapped to arms of the simulated crossing
ARM_CAMERA = {"N": "j1", "E": "j2", "S": "j3"}
MINOR_ARM = "W"          # no camera: modelled as a minor road
MINOR_ARM_FACTOR = 0.4   # share of the mean of the three observed arms
INBOUND_SHARE = 0.5      # each camera sees traffic in both directions
TURN_SPLIT = {"straight": 0.7, "left": 0.15, "right": 0.15}

TRAIN_DATES = ["2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14", "2023-01-15"]
TEST_DATES = ["2023-01-16", "2023-01-17", "2023-01-18"]
DAY_START = "09:00"
DAY_SECONDS = 3 * 3600

DELTA_T = 5        # seconds between decisions
YELLOW = 3
MIN_GREEN = 10
MAX_GREEN = 90
