from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
from sagemaker.tuner import IntegerParameter, HyperparameterTuner, ContinuousParameter
from keras.models import load_model
from urllib.parse import urlparse
import tarfile
import os
import boto3
from utils.logging_framework import log
from config.load_prod2vec_config import load_yaml
from config import constants
from src.hyperparameter_plots.plot_helpers import *

# load config
log.info("Loading config")
config = load_yaml(constants.config_path)

role = get_execution_role()

# Run the hyper-parameter tuning with Sagemaker and Keras
log.info("Run hyper-parameter tuning")
tf_estimator = TensorFlow(
    entry_point=config["model_train_constants"]["Entry_point"],
    role=role,
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="2.0.0",
    py_version="py3",
    metric_definitions=[{"Name": "loss", "Regex": "loss: ([0-9\\.]+)"}],
    hyperparameters={
        "epochs": config["hyperparameter"]["Epochs"],
        "num_prods": config["preprocess_constants"]["num_prods"],
        "valid_size": config["model_train_constants"]["Valid_size"],
        "valid_window": config["model_train_constants"]["Valid_window"],
        "num_folds": config["model_train_constants"]["Num_folds"],
        "cross_validate": False,
    },
)

# Set up the hyper-parameter tuning job
tf_hyperparameter_tuner = HyperparameterTuner(
    estimator=tf_estimator,
    objective_metric_name="loss",
    objective_type="Minimize",
    metric_definitions=[{"Name": "loss", "Regex": "loss: ([0-9\\.]+)"}],
    max_jobs=config["hyperparameter"]["Max_jobs"],
    max_parallel_jobs=config["hyperparameter"]["Max_parallel_jobs"],
    hyperparameter_ranges={
        "num_embeddings": IntegerParameter(
            config["hyperparameter"]["Min_embeddings"],
            config["hyperparameter"]["Max_embeddings"],
        ),
        "learning_rate": ContinuousParameter(
            config["hyperparameter"]["Min_learning_rate"],
            config["hyperparameter"]["Max_learning_rate"],
        ),
    },
)

tf_hyperparameter_tuner.fit({"train": config["s3"]["Bucket"]})

# Get the best training job (lowest loss)
log.info('best training job')
tf_hyperparameter_tuner.best_training_job()

# Get analysis on the tuning job including best hyper-parameters
tuning_analysis = (
    tf_hyperparameter_tuner.analytics()
    .dataframe()
    .sort_values(by=["TrainingStartTime"], ascending=False, axis=0)
)
tuning_analysis["iteration"] = tuning_analysis.index
tuning_analysis.sort_values(
    by=["FinalObjectiveValue"], ascending=True, axis=0, inplace=True
)

log.info('hyper-parameter tuning analysis')
tuning_analysis.head()

# Plot the value of hyper-parameters over the tuning window
params_over_search = plot_hyperparams_over_search(
    df=tuning_analysis, hyperparams=["num_embeddings", "learning_rate"]
)
params_over_search.savefig('params_over_search.png')

# Create a kde plot of hyper-parameters used over the tuning window
search_kde = plot_search_dist(
    df=tuning_analysis, hyperparams=["num_embeddings", "learning_rate"]
)
search_kde.savefig('search_kde.png')

log.info('Re-fitting best model')
# Get the best hyper-parameters and refit the model over the training data with the best params
tuning_analysis.sort_values(
    by=["FinalObjectiveValue"], ascending=False, axis=0, inplace=True
)
learning_rate = tuning_analysis["learning_rate"].reset_index(drop=True)[0]
num_embeddings = int(tuning_analysis["num_embeddings"].reset_index(drop=True)[0])

# Retrain the best model using all data - no cross-validation
tf_estimator = TensorFlow(
    entry_point=config["model_train_constants"]["Entry_point"],
    role=role,
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="2.0.0",
    py_version="py3",
    metric_definitions=[{"Name": "loss", "Regex": "loss: ([0-9\\.]+)"}],
    hyperparameters={
        "epochs": config["model_train_constants"]["Epochs"],
        "num_prods": config["preprocess_constants"]["num_prods"],
        "valid_size": config["model_train_constants"]["Valid_size"],
        "valid_window": config["model_train_constants"]["Valid_window"],
        "learning_rate": learning_rate,
        "num_embeddings": num_embeddings,
        "cross_validate": False,
    },
)

tf_estimator.fit({"train": config["s3"]["Bucket"]})

# Get path to the model assets
training_job_description = tf_estimator.jobs[-1].describe()
model_data_s3_uri = "{}{}/{}".format(
    training_job_description["OutputDataConfig"]["S3OutputPath"],
    training_job_description["TrainingJobName"],
    "output/model.tar.gz",
)

# Download and extract the model assets
parsed_url = urlparse(model_data_s3_uri)
bucket = parsed_url.netloc
key = os.path.join(parsed_url.path[1:])
s3 = boto3.resource("s3")
s3.Bucket(bucket).download_file(key, "./model.tar.gz")

with tarfile.open("./model.tar.gz") as tar:
    tar.extractall(path="./")

# Load the model
model = load_model("prod2vec_model")

# Get the embedding weights
embed_weights = get_model_weights(model, 2)
embed_weights.head()

# Map embedding to product_id and description
embed_weights_prods = map_weights_to_prods(embed_weights, products, reversed_dictionary)
embed_weights_prods.head()
