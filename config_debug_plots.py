# debug_plot.py
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

# Detect headless mode (cluster or no DISPLAY) or SLURM environment
HEADLESS = not bool(os.environ.get("DISPLAY")) or os.environ.get("SLURM_JOB_ID") is not None
if HEADLESS:
    matplotlib.use("Agg")  # headless backend

# Global counter for numbering plots
_plot_counter = 0

# Folder to save debug plots
DEBUG_FOLDER = "Temporal-DebugPlots"

if os.path.exists(DEBUG_FOLDER):
    shutil.rmtree(DEBUG_FOLDER) #delete entire folder
os.makedirs(DEBUG_FOLDER, exist_ok=True)  # create folder if it doesn't exist

def debug_plot(name="debug", overwrite=True):
    """
    Saves or shows the current figure.
    - On cluster/headless: saves figure to Temporal-DebugPlots folder.
        - If overwrite=True → debug.png
        - If overwrite=False → debug_001.png, debug_002.png, etc.
    - Locally: opens interactive window.
    """
    global _plot_counter
    _plot_counter += 1

    if HEADLESS:
        if overwrite:
            fname = os.path.join(DEBUG_FOLDER, f"{name}.pdf")
        else:
            fname = os.path.join(DEBUG_FOLDER, f"{name}_{_plot_counter:03d}.pdf")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[debug] plot saved → {fname}")
    else:
        plt.show()
