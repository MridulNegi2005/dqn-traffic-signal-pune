"""Generate notebooks/pune_signal_rl.ipynb (run once; the .ipynb is committed)."""
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/MridulNegi2005/pune-signal-rl.git"

cells = [
    ("md", "# DQN traffic-signal control on real Pune traffic counts\n"
           "This notebook reproduces the project end to end on Kaggle or Google Colab: "
           "it downloads the Pune dataset, builds 5-minute demand, trains a DQN agent in SUMO "
           "and compares it with fixed-time and actuated signal control."),
    ("code", "import os, subprocess, sys\n"
             "REPO = os.environ.get('REPO_URL', '" + REPO + "')\n"
             "if not os.path.exists('pune-signal-rl'):\n"
             "    subprocess.run(['git', 'clone', '--depth', '1', REPO], check=True)\n"
             "os.chdir('pune-signal-rl')\n"
             "!pip -q install eclipse-sumo==1.23.1 libsumo==1.23.1 traci==1.23.1 stable-baselines3 gymnasium"),
    ("md", "## 1. Download the dataset (Mendeley Data, CC BY 4.0)"),
    ("code", "!python src/download_data.py"),
    ("md", "## 2. Turn per-frame running totals into 5-minute counts"),
    ("code", "!python src/preprocess.py\n"
             "import pandas as pd\n"
             "pd.read_csv('data/processed/pune_5min.csv').head()"),
    ("md", "## 3. Train the DQN agent\nSUMO runs on the CPU in parallel copies; "
           "the Q-network trains on the GPU when one is present."),
    ("code", "import torch; print('GPU:', torch.cuda.is_available())\n"
             "!cd src && python train.py --steps 150000 --envs 4 --seed 1 --name dqn_seed1"),
    ("md", "## 4. Test on unseen days, heavier demand, an unseen junction and free left turns"),
    ("code", "!cd src && python evaluate.py --models ../results/dqn_seed1.zip\n"
             "!cd src && python plots.py"),
    ("code", "from IPython.display import Image, display\n"
             "for f in ['demand_profile.png', 'learning_curve.png', 'comparison.png', 'wait_by_class.png']:\n"
             "    display(Image('results/' + f))"),
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("pune_signal_rl.ipynb")
nbf.write(nb, out)
print("wrote", out)
