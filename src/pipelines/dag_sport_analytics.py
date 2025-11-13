from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 12),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_data_processing(**context):
    """Tâche 1: Process JSON data"""
    try:
        # Import inside function to avoid circular imports
        from src.data.process_data import process_multiple_matches, load_config

        config = load_config()
        paths = process_multiple_matches(config["data_dir"], config["selected_match"])
        logger.info(f"✅ Processed {len(paths)} matches")
        return len(paths)
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        raise


def run_model_training(**context):
    """Tâche 2: Train model with MLFlow"""
    try:
        from src.models.train_model import train_and_log_model

        results = train_and_log_model(
            run_name=f"airflow_run_{datetime.now().strftime('%Y%m%d')}"
        )
        r2 = results["test_r2"]
        logger.info(f"✅ Model trained - R²: {r2}, Run ID: {results['run_id']}")

        # Push individual values to XCom for easier access
        context["ti"].xcom_push(key="test_r2", value=r2)
        context["ti"].xcom_push(key="run_id", value=results["run_id"])

        return results
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise


def run_agent_report(**context):
    """Tâche 3: Generate coaching report - SYNC version"""
    try:
        from src.agents.coaching_agent import CoachingAgent

        run_id = context["ti"].xcom_pull(task_ids="run_model_training", key="run_id")
        agent = CoachingAgent(run_id=run_id)

        # Use sync wrapper instead of async
        report = agent.generate_report_sync("2021")

        logger.info(f"✅ Report generated: {report[:100]}...")

        # Save report
        os.makedirs("/tmp/airflow_reports", exist_ok=True)
        report_path = f"/tmp/airflow_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"📄 Report saved to: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"❌ Agent report failed: {e}")
        raise


def check_drift(**context):
    """Tâche 4: Check for model drift"""
    try:
        current_r2 = context["ti"].xcom_pull(
            task_ids="run_model_training", key="test_r2"
        )

        # Simple drift detection - compare with minimum threshold
        drift_threshold = 0.7
        warning_threshold = 0.75

        if current_r2 < drift_threshold:
            logger.warning(f"🚨 DRIFT DETECTED: R² {current_r2} < {drift_threshold}")
            return "send_alert"
        elif current_r2 < warning_threshold:
            logger.warning(f"⚠️  WARNING: R² {current_r2} < {warning_threshold}")
            return "end_pipeline"
        else:
            logger.info(f"✅ No drift detected: R² {current_r2} >= {warning_threshold}")
            return "end_pipeline"
    except Exception as e:
        logger.error(f"❌ Drift check failed: {e}")
        return "end_pipeline"  # Fail safe


def send_alert(**context):
    """Tâche 5: Send alert for drift"""
    current_r2 = context["ti"].xcom_pull(task_ids="run_model_training", key="test_r2")
    logger.error(f"🚨 ALERT: Model drift detected! R² = {current_r2}")
    # In production, you would add email/Slack notification here
    return "Alert sent - check logs"


def end_pipeline(**context):
    """Tâche 6: Successful pipeline completion"""
    current_r2 = context["ti"].xcom_pull(task_ids="run_model_training", key="test_r2")
    logger.info(f"🎉 Pipeline completed successfully! Final R²: {current_r2}")
    return "Pipeline completed"


with DAG(
    dag_id="sport_analytics_mlop_pipeline",  # ✅ Matchs ton nom de test
    default_args=default_args,
    description="DAG MLOps: Process → Train → Agent Rapport",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["mlops", "basketball"],
) as dag:

    # Define tasks
    process_task = PythonOperator(
        task_id="process_data", python_callable=run_data_processing
    )

    train_task = PythonOperator(
        task_id="train_model", python_callable=run_model_training
    )

    agent_task = PythonOperator(
        task_id="generate_report", python_callable=run_agent_report
    )

    drift_check_task = BranchPythonOperator(
        task_id="check_drift", python_callable=check_drift
    )

    alert_task = PythonOperator(task_id="send_alert", python_callable=send_alert)

    end_task = PythonOperator(task_id="end_pipeline", python_callable=end_pipeline)

    # Define workflow
    process_task >> train_task >> [agent_task, drift_check_task]
    drift_check_task >> [alert_task, end_task]
    alert_task >> end_task  # Ensure pipeline ends after alert
