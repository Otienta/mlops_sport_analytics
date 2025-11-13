from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

def test_imports():
    print("=== TESTING IMPORTS ===")
    
    # Test 1: Import de base
    try:
        import pandas as pd
        print("✅ pandas import OK")
    except Exception as e:
        print(f"❌ pandas failed: {e}")
    
    # Test 2: Import de ton projet
    try:
        # Ajoute le chemin source
        project_root = "/home/students/Adapter/sk/mlops_sport_analytics"
        sys.path.insert(0, project_root)
        
        from src.data.process_data import load_config
        print("✅ load_config import OK")
        
        config = load_config()
        print(f"✅ Config loaded: {config}")
    except Exception as e:
        print(f"❌ Project imports failed: {e}")
    
    return "Import test completed"

with DAG(
    dag_id='test_imports_dag',
    start_date=datetime(2025, 11, 12),
    schedule=None,
    catchup=False,
    tags=['test']
) as dag:

    import_task = PythonOperator(
        task_id='test_imports',
        python_callable=test_imports
    )