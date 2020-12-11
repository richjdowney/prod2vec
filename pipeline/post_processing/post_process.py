import sys

sys.path.insert(1, "/home/ubuntu/prod2vec")

import tensorflow as tf
import pickle
import tarfile
import os
from urllib.parse import urlparse
from utils.util_functions import *
from utils.logging_framework import log


def save_embeddings(
    hpo_enabled: bool, job_name: str, bucket: str, run_id: str, model_save_loc: str
):
    """ Function to save the embedding weights to s3

        Parameters
        ----------
        hpo_enabled : bool
            Flag to identify if hyper-parameter tuning was run
        job_name : str
            Name of the hyper-parameter tuning job or training job
        bucket: str
            Name of the s3 bucket with the product descriptions and dictionaries
        run_id : str
            Identifier for the run
        model_save_loc : str
            Location on ec2 to save the model artifacts

    """

    log.info("Running post-processing for tuning job {}".format(job_name))

    # ======== Download product descriptions and dictionaries required =========

    log.info("Downloading product descriptions and dictionaries")

    # Loading the integer to product_id dictionary
    s3 = boto3.resource("s3")
    with open("reversed_dictionary.pkl", "wb") as data:
        s3.Bucket(bucket).download_fileobj("reversed_dictionary.pkl", data)

    with open("reversed_dictionary.pkl", "rb") as data:
        reversed_dictionary = pickle.load(data)

    # Loading the product ID to product description dictionary
    with open("products_dict.pkl", "wb") as data:
        s3.Bucket(bucket).download_fileobj("products_dict.pkl", data)

    with open("products_dict.pkl", "rb") as data:
        products_dict = pickle.load(data)

    # ======== Download and extract the model assets =========

    log.info("Downloading and extracting model artifacts")

    # If tuning job has run get the best job name
    if hpo_enabled:

        # Get the name of the best job from the tuning analysis
        tuning_analysis = read_csv_from_s3(bucket, "{}.csv".format(job_name))
        tuning_analysis.sort_values("FinalObjectiveValue", inplace=True)
        best_job = tuning_analysis["TrainingJobName"][0]

    else:
        best_job = job_name

    # Get path to the model assets
    client = boto3.client("sagemaker")
    model_data_s3_uri = client.describe_training_job(TrainingJobName=best_job)[
        "ModelArtifacts"
    ]["S3ModelArtifacts"]
    parsed_url = urlparse(model_data_s3_uri)
    model_bucket = parsed_url.netloc
    key = os.path.join(parsed_url.path[1:])
    s3 = boto3.resource("s3")
    s3.Bucket(model_bucket).download_file(key, "{}/model.tar.gz".format(model_save_loc))

    # ======== Extract and save the embeddings =========

    log.info("Extracting and saving the embeddings")

    with tarfile.open("./model.tar.gz") as tar:
        tar.extractall(path="./")

    # Load the model
    model = tf.keras.models.load_model("{}/prod2vec_model".format(model_save_loc))

    # Get the embedding weights
    embed_weights = pd.DataFrame(model.get_weights()[0])

    # Map embedding to product_id and description
    embed_weights.loc[:, "index"] = embed_weights.index
    embed_weights.loc[:, "product_id"] = embed_weights["index"].map(reversed_dictionary)
    embed_weights.loc[:, "product_name"] = embed_weights["product_id"].map(
        products_dict
    )

    # Upload embeddings to s3 as csv
    embed_file_name = "prod2vec_embed_weights_run_{}".format(run_id)
    embed_weights.to_csv("s3://{}/{}.csv".format(bucket, embed_file_name), index=False)
