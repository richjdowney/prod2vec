from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from config.load_prod2vec_config import load_yaml
from config import constants
from utils.logging_framework import log

# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

with DAG(**config["dag"]) as dag:

    # Pre-process the data to get the target/context pairs required by the prod2vec model
    pre_process_data = BashOperator(
        task_id="pre_process_data",
        bash_command="python /home/ubuntu/prod2vec/runners/data_preprocessing_runner.py",
        run_as_user="airflow",
    )

    # Pre-process the data to get the target/context pairs required by the prod2vec model
    train_prod2vec = BashOperator(
        task_id="train_prod2vec",
        bash_command="python /home/ubuntu/prod2vec/runners/model_train_runner.py",
        run_as_user="airflow",
    )


pre_process_data >> train_prod2vec
