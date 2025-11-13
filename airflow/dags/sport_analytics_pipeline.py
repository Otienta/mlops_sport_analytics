from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Configuration du chemin ABSOLU
PROJECT_ROOT = "/home/students/Adapter/sk/mlops_sport_analytics"
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 12),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def step1_process_data(**context):
    """Étape 1: Traitement des données"""
    try:
        from src.data.process_data import process_multiple_matches, load_config
        
        logger.info("📊 Starting data processing...")
        config = load_config()
        paths = process_multiple_matches(config['data_dir'], config['selected_match'])
        
        logger.info(f"✅ Processed {len(paths)} matches")
        return len(paths)
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        raise

def step2_train_model(**context):
    """Étape 2: Entraînement du modèle"""
    try:
        from src.models.train_model import train_and_log_model
        
        logger.info("🤖 Starting model training...")
        results = train_and_log_model(run_name=f"airflow_run_{datetime.now().strftime('%Y%m%d')}")
        
        r2 = results['test_r2']
        logger.info(f"✅ Model trained - R²: {r2}")
        
        # Stocke les résultats pour les étapes suivantes
        context['ti'].xcom_push(key='r2_score', value=r2)
        context['ti'].xcom_push(key='run_id', value=results['run_id'])
        
        return f"Training completed with R²: {r2}"
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise

def step3_generate_report(**context):
    """Étape 3: Génération du rapport"""
    try:
        from src.agents.coaching_agent import CoachingAgent
        import asyncio
        
        logger.info("📝 Starting report generation...")
        
        # Récupère le run_id de l'étape précédente
        run_id = context['ti'].xcom_pull(task_ids='step2_train_model', key='run_id')
        agent = CoachingAgent(run_id=run_id)
        
        # Génère le rapport
        report = asyncio.run(agent.generate_report("2021"))
        
        # Sauvegarde le rapport
        os.makedirs('/tmp/airflow_reports', exist_ok=True)
        report_path = f"/tmp/airflow_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report generated and saved to: {report_path}")
        logger.info(f"📄 Report preview: {report[:200]}...")
        
        return f"Report saved to {report_path}"
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise

def step4_check_quality(**context):
    """Étape 4: Vérification de la qualité"""
    try:
        r2_score = context['ti'].xcom_pull(task_ids='step2_train_model', key='r2_score')
        
        logger.info(f"🔍 Checking model quality - R²: {r2_score}")
        
        if r2_score < 0.7:
            logger.error(f"🚨 MODEL QUALITY ALERT: R² {r2_score} is below threshold 0.7")
            return "QUALITY_FAIL"
        elif r2_score < 0.8:
            logger.warning(f"⚠️  MODEL WARNING: R² {r2_score} is below 0.8")
            return "QUALITY_WARNING"
        else:
            logger.info(f"🎉 MODEL EXCELLENT: R² {r2_score} is above 0.8")
            return "QUALITY_EXCELLENT"
    except Exception as e:
        logger.error(f"❌ Quality check failed: {e}")
        raise

with DAG(
    dag_id='sport_analytics_pipeline',
    default_args=default_args,
    description='Pipeline MLOps complet: Data → Train → Report → Quality Check',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'basketball', 'coaching']
) as dag:

    # Définition des tâches
    process_task = PythonOperator(
        task_id='step1_process_data',
        python_callable=step1_process_data
    )

    train_task = PythonOperator(
        task_id='step2_train_model',
        python_callable=step2_train_model
    )

    report_task = PythonOperator(
        task_id='step3_generate_report',
        python_callable=step3_generate_report
    )

    quality_task = PythonOperator(
        task_id='step4_check_quality',
        python_callable=step4_check_quality
    )

    # Workflow linéaire simple
    process_task >> train_task >> report_task >> quality_task