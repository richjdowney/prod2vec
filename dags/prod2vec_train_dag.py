# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/prod2vec")

# Local modules
from config.load_prod2vec_config import load_yaml, Config
from config import constants
from utils.logging_framework import log
from pipeline.data_preprocessing import data_prep
from pipeline.tuning_analysis import tuning_analysis
from pipeline.post_processing import post_process

# Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.contrib.hooks.aws_hook import AwsHook
from airflow.contrib.operators.sagemaker_training_operator import (
    SageMakerTrainingOperator,
)
from airflow.contrib.operators.sagemaker_tuning_operator import SageMakerTuningOperator


# AWS
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.airflow import training_config, tuning_config
from sagemaker.tuner import IntegerParameter, HyperparameterTuner, ContinuousParameter


# =============================================================================
# load configuration file
# =============================================================================

config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)


# =============================================================================
# functions
# =============================================================================


def get_sagemaker_role_arn(role_name, region_name):
    """Function to get SageMaker role"""
    iam = boto3.client("iam", region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]


def is_hpo_enabled():
    """check if hyper-parameter optimization is enabled in the config
    """

    hpo = False
    run_hpo_config = config["estimator_config"]["static_params"][
        "run_hyperparameter_opt"
    ]
    if run_hpo_config.lower() == "yes":
        hpo = True

    return hpo


# =============================================================================
# setting up training and tuning configuration
# =============================================================================

# Should hyperparameter optimization be run?
hpo_enabled = is_hpo_enabled()

# set configuration for tasks
hook = AwsHook(aws_conn_id="airflow-sagemaker")
region = "us-west-2"
sess = hook.get_session(region_name=region)
role = get_sagemaker_role_arn("sagemaker-role", sess.region_name)

# =====================================
# ===== Create training estimator =====
# =====================================

# create train estimator
tf_train_estimator = TensorFlow(
    entry_point=config["estimator_config"]["static_params"]["entry_point"],
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="2.0.0",
    py_version="py3",
    metric_definitions=config["estimator_config"]["static_params"][
        "metric_definitions"
    ],
    hyperparameters={
        "epochs": config["estimator_config"]["static_params"]["epochs"],
        "num_prods": config["preprocess_constants"]["num_prods"],
        "valid_size": config["estimator_config"]["static_params"]["valid_size"],
        "valid_window": config["estimator_config"]["static_params"]["valid_window"],
        "num_folds": config["estimator_config"]["static_params"]["num_folds"],
        "cross_validate": config["estimator_config"]["static_params"]["cross_validate"],
        "num_embeddings": config["estimator_config"]["train_hyperparameters"][
            "embeddings"
        ],
        "learning_rate": config["estimator_config"]["train_hyperparameters"][
            "learning_rate"
        ],
    },
)

# train_config specifies SageMaker training configuration
train_config = training_config(
    estimator=tf_train_estimator,
    inputs=config["estimator_config"]["inputs"],
    job_name="hyperparameter-tuner-prod2vec",
)

# =====================================
# ====== Create tuning estimator ======
# =====================================

# Create tuning estimator
tf_tune_estimator = TensorFlow(
    entry_point=config["estimator_config"]["static_params"]["entry_point"],
    role=role,
    sagemaker_session=sagemaker.session.Session(sess),
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="2.0.0",
    py_version="py3",
    metric_definitions=config["estimator_config"]["static_params"][
        "metric_definitions"
    ],
    hyperparameters={
        "epochs": config["estimator_config"]["static_params"]["epochs"],
        "num_prods": config["preprocess_constants"]["num_prods"],
        "valid_size": config["estimator_config"]["static_params"]["valid_size"],
        "valid_window": config["estimator_config"]["static_params"]["valid_window"],
        "num_folds": config["estimator_config"]["static_params"]["num_folds"],
        "cross_validate": config["estimator_config"]["static_params"]["cross_validate"],
    },
)

# create hyper-parameter tuning object
prod2vec_tuner = HyperparameterTuner(
    estimator=tf_tune_estimator,
    metric_definitions=config["estimator_config"]["static_params"][
        "metric_definitions"
    ],
    objective_metric_name=config["estimator_config"]["static_params"][
        "objective_metric_name"
    ],
    objective_type=config["estimator_config"]["static_params"]["objective_type"],
    max_jobs=config["estimator_config"]["tune_hyperparameters"]["max_jobs"],
    max_parallel_jobs=config["estimator_config"]["tune_hyperparameters"][
        "max_parallel_jobs"
    ],
    hyperparameter_ranges={
        "num_embeddings": IntegerParameter(
            config["estimator_config"]["tune_hyperparameters"]["min_embeddings"],
            config["estimator_config"]["tune_hyperparameters"]["max_embeddings"],
        ),
        "learning_rate": ContinuousParameter(
            config["estimator_config"]["tune_hyperparameters"]["min_learning_rate"],
            config["estimator_config"]["tune_hyperparameters"]["max_learning_rate"],
        ),
    },
)

# tune_config specifies SageMaker tuning configuration
tune_config = tuning_config(
    tuner=prod2vec_tuner,
    inputs=config["estimator_config"]["inputs"],
    job_name="hyperparameter-tuner-prod2vec",
)

# =============================================================================
# define airflow DAG and tasks
# =============================================================================

with DAG(**config["dag"]) as dag:

    # Initialization task
    init = DummyOperator(task_id="start", dag=dag)

    # Pre-process the data to get the target/context pairs required by the prod2vec model
    pre_process_data = PythonOperator(
        task_id="pre_process_data",
        dag=dag,
        provide_context=False,
        python_callable=data_prep.run_data_preprocessing,
    )

    # Determine if hyper-parameter tuning should be run
    branching = BranchPythonOperator(
        task_id="branching",
        dag=dag,
        python_callable=lambda: "model_tuning" if hpo_enabled else "model_training",
    )

    # launch SageMaker training job and wait until it completes
    train_prod2vec = SageMakerTrainingOperator(
        task_id="model_training",
        dag=dag,
        config=train_config,
        aws_conn_id="airflow-sagemaker",
        wait_for_completion=True,
        check_interval=30,
    )

    # launch SageMaker tuning job and wait until it completes
    tune_prod2vec = SageMakerTuningOperator(
        task_id="model_tuning",
        dag=dag,
        config=tune_config,
        aws_conn_id="airflow-sagemaker",
        wait_for_completion=True,
        check_interval=30,
    )

    # Conduct tuning analysis (if hyper-parameter tuning is run)
    tuning_analysis = PythonOperator(
        task_id="tuning_analysis",
        dag=dag,
        provide_context=False,
        python_callable=tuning_analysis.run_tuning_analysis,
        op_kwargs={"job_name": "hyperparameter-tuner-prod2vec"},
    )

    # Post processing to save the embeddings
    post_processing = PythonOperator(
        task_id="post_processing",
        dag=dag,
        provide_context=False,
        python_callable=post_process.save_embeddings,
        op_kwargs={"hpo_enabled": hpo_enabled,
                   "job_name": "hyperparameter-tuner-prod2vec",
                   "bucket": config["s3"]["bucket"],
                   "products_key": config["s3"]["products_key"]},
    )

init.set_downstream(pre_process_data)
pre_process_data.set_downstream(branching)
branching.set_downstream(train_prod2vec)
branching.set_downstream(tune_prod2vec)
tune_prod2vec.set_downstream(tuning_analysis)
train_prod2vec.set_downstream(post_processing)
tuning_analysis.set_downstream(post_processing)
