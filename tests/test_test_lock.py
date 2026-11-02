"""Hard rule 1: scripts that touch the FiQA test split refuse to run without --final."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_final_generate_refuses_without_flag():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "final_generate.py")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "locked" in r.stderr


def test_run_experiment_refuses_final_config_without_flag():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_experiment.py"),
                        str(REPO / "experiments" / "configs" / "final_a.yaml")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "locked" in r.stderr
