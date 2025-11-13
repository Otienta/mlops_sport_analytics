# mlops_sport_analytics/src/models/train_model.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml
import os
from typing import Dict, Tuple


def load_config(config_path: str = None) -> Dict:
    """Charge le config.yaml avec un chemin absolu basé sur la racine du projet."""
    if config_path is None:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )  # -> mlops_sport_analytics/
        config_path = os.path.join(base_dir, "config", "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    csv_path: str = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """Prépare X/y : features vs player_impact, filtre players."""

    if csv_path is None:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )  # -> mlops_sport_analytics/
        csv_path = os.path.join(base_dir, "data", "processed", "all_matches_merged.csv")

    print(f"Chargement dataset : {csv_path}")
    df = pd.read_csv(csv_path)

    # Filtre players only (exclut rows équipe avec 'tot_s' dans name ou NaN impact)
    players_df = df[
        df["player_impact"].notna() & ~df["name"].str.contains("tot_s", na=False)
    ].copy()
    print(f"Dataset players : {len(players_df)} rows sur {len(df)} total.")

    # Features : Sélection auto (colonnes numériques s*/tot_s* pertinentes)
    num_cols = players_df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in num_cols
        if col.startswith(("s", "tot_s")) and col not in ["player_impact", "match_id"]
    ]
    feature_cols += ["off_efficiency", "possessions"]  # Nos engineered
    available_features = [col for col in feature_cols if col in players_df.columns]
    X = players_df[available_features].fillna(0)  # Remplit NaN
    y = players_df["player_impact"].fillna(0)

    # Split 80/20 (stratifié si besoin, mais random pour simplicité)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"Features utilisées : {available_features[:10]}... ({len(available_features)} total)"
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features


def train_and_log_model(
    experiment_name: str = "sport_impact_v1", run_name: str = "rf_v1_large_dataset"
) -> Dict:
    """Entraîne, logue MLFlow, check threshold."""
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Hyperparams (ex. : tune-les plus tard via MLFlow UI)
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)

        # Prep data
        X_train, X_test, y_train, y_test, scaler, features = prepare_dataset()

        # Train
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)

        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mae", mae)

        # Log artefacts
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.sklearn.log_model(scaler, "scaler")
        mlflow.log_text(str(features), "features_list.txt")

        # Feature importance (utile pour agent LLM : "quelles stats comptent ?")
        importance = dict(zip(features, model.feature_importances_))
        mlflow.log_dict(importance, "feature_importance.json")

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Test R²: {test_r2:.4f} | MAE: {mae:.4f} (sur {len(y_test)} samples)")

        # Threshold pour gouvernance (PDF : "examen et gouvernance")
        if test_r2 >= config["min_r2_threshold"]:
            mlflow.set_tag("status", "production_ready")
            print("✅ Modèle performant – prêt pour déploiement !")
        else:
            mlflow.set_tag("status", "needs_tuning")
            print(
                f"⚠️ R² {test_r2:.4f} < {config['min_r2_threshold']} – tuning suggéré."
            )

        return {
            "test_r2": test_r2,
            "mae": mae,
            "run_id": mlflow.active_run().info.run_id,
            "features": features,
        }


if __name__ == "__main__":
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )  # -> mlops_sport_analytics/
    csv_path = os.path.join(base_dir, "data", "processed", "all_matches_merged.csv")
    if not os.path.exists(csv_path):

        raise FileNotFoundError("Run Étape 1 pour générer le merged CSV !")
    results = train_and_log_model()
    print("Entraînement terminé :", results)
