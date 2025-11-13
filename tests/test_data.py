# mlops_sport_analytics/tests/test_data.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.process_data import process_multiple_matches, load_config
import pandas as pd


def test_multi_process():
    config = load_config()
    paths = process_multiple_matches(
        config["data_dir"], selected_match="2648661"
    )  # Test filtre
    assert len(paths) >= 1
    df = pd.read_csv(paths[0])
    assert "player_impact" in df.columns
    print("Test multi-fichiers passé !")


if __name__ == "__main__":
    test_multi_process()
