from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello_world():
    print("🎉 Hello from Airflow! Basic test successful!")
    return "Success"

with DAG(
    dag_id='test_basic_dag',
    start_date=datetime(2025, 11, 12),
    schedule=None,
    catchup=False,
    tags=['test']
) as dag:

    test_task = PythonOperator(
        task_id='hello_task',
        python_callable=hello_world
    )