# mlops_sport_analytics/scripts/run_pipeline.py
#!/usr/bin/env python3
"""
Script d'exécution complet du pipeline MLOps
Utilisable en local et en CI/CD
"""

import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(season="2021", run_name=None):
    """Exécute le pipeline MLOps complet"""
    try:
        # 1. Data Processing
        logger.info("🚀 Starting MLOps Pipeline...")

        from src.data.process_data import process_multiple_matches, load_config

        config = load_config()
        paths = process_multiple_matches(config["data_dir"], config["selected_match"])
        logger.info(f"✅ Data processed: {len(paths)} matches")

        # 2. Model Training
        from src.models.train_model import train_and_log_model

        if not run_name:
            run_name = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results = train_and_log_model(run_name=run_name)
        r2_score = results["test_r2"]
        run_id = results["run_id"]

        logger.info(f"✅ Model trained - R²: {r2_score:.3f}, Run ID: {run_id}")

        # 3. AI Report Generation
        from src.agents.coaching_agent import CoachingAgent

        agent = CoachingAgent(run_id=run_id)
        report = asyncio.run(agent.generate_report(season))

        # 4. Save report
        os.makedirs("/tmp/pipeline_reports", exist_ok=True)
        report_path = f"/tmp/pipeline_reports/report_{run_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"✅ Report generated: {report_path}")
        logger.info(f"📊 Report preview: {report[:200]}...")

        # 5. Quality check
        if r2_score < 0.7:
            logger.warning("⚠️  Model quality below threshold (R² < 0.7)")
            return False, report_path
        else:
            logger.info("🎉 Pipeline completed successfully!")
            return True, report_path

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description="MLOps Sport Analytics Pipeline")
    parser.add_argument("--season", default="2021", help="Season to analyze")
    parser.add_argument("--run-name", help="MLFlow run name")
    parser.add_argument("--ci", action="store_true", help="CI mode (no interactive)")

    args = parser.parse_args()

    success, report_path = run_complete_pipeline(args.season, args.run_name)

    if args.ci:
        # En mode CI, on retourne un code de sortie
        sys.exit(0 if success else 1)
    else:
        # En mode local, on affiche le rapport
        if report_path and os.path.exists(report_path):
            print("\n" + "=" * 50)
            print("FINAL REPORT")
            print("=" * 50)
            with open(report_path, "r") as f:
                print(f.read())


if __name__ == "__main__":
    main()
