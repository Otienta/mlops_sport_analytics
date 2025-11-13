# mlops_sport_analytics/src/data/process_data.py
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Charge la config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_match_data(json_path: str) -> Dict:
    """Charge un JSON de match."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_team_stats(
    data: Dict, team_key: str = "1", match_id: str = ""
) -> pd.DataFrame:
    """Extrait stats équipe/joueurs, avec ID match pour traçabilité."""
    tm = data["tm"][team_key]
    # Fix: Collect tot_s keys directly (flat structure)
    team_stats_row = {
        k: v
        for k, v in tm.items()
        if k.startswith("tot_s") and isinstance(v, (int, float))
    }
    team_stats = pd.DataFrame([team_stats_row])
    team_stats["team"] = tm["name"]
    team_stats["match_id"] = match_id

    # Fix: s keys direct for players
    players = []
    for p_id, p_data in tm["pl"].items():
        p_row = {
            k: v
            for k, v in p_data.items()
            if k.startswith("s") and isinstance(v, (int, float))
        }
        p_row.update(
            {
                "name": p_data["name"],
                "shirt": p_data.get("shirtNumber", "N/A"),
                "match_id": match_id,
            }
        )
        players.append(p_row)
    players_df = pd.DataFrame(players)

    return pd.concat([team_stats, players_df], ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingénierie features : utilise s* pour players, tot_s* pour team."""
    # Possessions (adapté par row)
    df["possessions"] = np.where(
        df["tot_sFieldGoalsAttempted"].notna(),
        df["tot_sFieldGoalsAttempted"]
        + 0.44 * df["tot_sFreeThrowsAttempted"]
        - df["tot_sReboundsOffensive"]
        + df["tot_sTurnovers"],
        df["sFieldGoalsAttempted"]
        + 0.44 * df["sFreeThrowsAttempted"]
        - df["sReboundsOffensive"]
        + df["sTurnovers"],
    )
    df["off_efficiency"] = df["tot_sPoints"].fillna(df["sPoints"]) / df[
        "possessions"
    ].replace(0, np.nan)

    # Impact (players only, using s*)
    df["player_impact"] = np.where(
        df["sPoints"].notna(),
        df["sPlusMinusPoints"].fillna(0)
        + 0.5 * df["sPoints"]
        + df["sReboundsTotal"].fillna(0)
        + df["sAssists"].fillna(0)
        - df["sTurnovers"].fillna(0),
        np.nan,  # NaN for team row
    )

    # EDA : Top 5 par impact (players only)
    players_only = df[df["name"].str.contains("tot_s", na=False) == False].copy()
    if not players_only.empty:
        top_impact = players_only.nlargest(5, "player_impact")[
            ["name", "player_impact", "off_efficiency"]
        ]
        print(
            f"EDA - Top 5 Joueurs par Impact (Match {df['match_id'].iloc[0] if 'match_id' in df else 'unknown'}):\n{top_impact.to_string()}"
        )

    return df


def process_single_match(json_path: str, output_dir: str = "data/processed/") -> str:
    """Traite un seul match."""
    match_id = (
        Path(json_path).stem.split("_")[-1]
        if "_" in Path(json_path).stem
        else Path(json_path).stem
    )
    data = load_match_data(json_path)
    df_team1 = extract_team_stats(data, "1", match_id)
    df_enhanced = engineer_features(df_team1)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"match_{match_id}.csv")
    df_enhanced.to_csv(output_path, index=False)
    print(f"Match {match_id} traité et sauvegardé : {output_path}")
    return output_path


def process_multiple_matches(
    data_dir: str,
    selected_match: Optional[str] = None,
    output_dir: str = "data/processed/",
) -> List[str]:
    """Traite multiples JSON : auto-scan, filtre optionnel."""
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"Aucun JSON trouvé dans {data_dir}")

    processed_paths = []
    for json_file in json_files:
        if selected_match and selected_match not in json_file:
            continue
        json_path = os.path.join(data_dir, json_file)
        try:
            path = process_single_match(json_path, output_dir)
            processed_paths.append(path)
        except Exception as e:
            print(f"Erreur sur {json_file}: {e}")

    if not processed_paths:
        print(f"Aucun match traité (filtre: {selected_match}).")
    else:
        print(f"{len(processed_paths)} matchs traités.")

    # Merge si multi
    if len(processed_paths) > 1:
        merged_df = pd.concat(
            [pd.read_csv(p) for p in processed_paths], ignore_index=True
        )
        merged_path = os.path.join(output_dir, "all_matches_merged.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"Dataset merged sauvegardé : {merged_path}")

    return processed_paths


# Main pour test
if __name__ == "__main__":
    config = load_config()
    process_multiple_matches(config["data_dir"], config["selected_match"])
