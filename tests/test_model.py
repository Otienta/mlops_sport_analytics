# mlops_sport_analytics/tests/test_model.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.train_model import train_and_log_model, prepare_dataset
import tempfile
import pandas as pd
import pytest


def test_prepare_and_train():
    # Mock petit dataset pour test rapide
    mock_df = pd.DataFrame(
        {
            "name": ["Player1", "Player2"],
            "sPoints": [10, 20],
            "sReboundsTotal": [5, 8],
            "player_impact": [15, 25],
            "off_efficiency": [1.2, 1.5],
        }
    )
    mock_path = tempfile.mktemp(suffix=".csv")
    mock_df.to_csv(mock_path, index=False)

    try:
        X_train, X_test, y_train, y_test, scaler, features = prepare_dataset(mock_path)
        assert len(features) > 0
        assert len(X_train) > 0

        # Full train (MLFlow skip pour test ; mock run)
        results = train_and_log_model(run_name="mock_test")
        assert results["test_r2"] >= 0
        print("Test modèle passé !")
    finally:
        os.remove(mock_path)


if __name__ == "__main__":
    test_prepare_and_train()
