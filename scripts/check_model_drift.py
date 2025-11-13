# mlops_sport_analytics/scripts/check_model_drift.py
#!/usr/bin/env python3
"""
Script de surveillance de la dérive des modèles
"""

import mlflow
import sys
from datetime import datetime, timedelta


def check_model_drift():
    """Vérifie si le modèle actuel subit une dérive"""

    # Connect to MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")  # À adapter en prod

    try:
        # Récupère les 10 derniers runs
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=10)

        if len(runs) < 2:
            print("❌ Not enough runs for drift detection")
            return False

        latest_r2 = runs.iloc[0]["metrics.test_r2"]
        previous_r2 = runs.iloc[1]["metrics.test_r2"]

        drift_threshold = 0.1  # 10% de baisse

        if latest_r2 < previous_r2 * (1 - drift_threshold):
            print(f"🚨 Model drift detected!")
            print(f"Previous R²: {previous_r2:.3f}, Current R²: {latest_r2:.3f}")
            return True
        else:
            print(f"✅ No drift detected - R²: {latest_r2:.3f}")
            return False

    except Exception as e:
        print(f"❌ Error checking model drift: {e}")
        return True


if __name__ == "__main__":
    drift_detected = check_model_drift()
    sys.exit(1 if drift_detected else 0)
