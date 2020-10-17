import sys

sys.path.insert(1, "/home/ubuntu/prod2vec")

import keras
from keras.models import load_model
import pickle
import tarfile
import os
from urllib.parse import urlparse
from utils.util_functions import *
from utils.logging_framework import log


def map_weights_to_prods(
    df: pd.DataFrame, prod_lookup: pd.DataFrame, index_to_id_dict: dict
) -> pd.DataFrame:
    """ Function to append product details to the embeddings

        Parameters
        ----------
        df : pandas.DataFrame
            Pandas DataFrame containing the product embeddings
        prod_lookup : pandas.DataFrame
            Pandas DataFrame containing product details
        index_to_id_dict : dict
            Dictionary containing the index used to train the model and the product ID

        Returns
        -------
        df : pandas.DataFrame
            Embeddings with the product details appended

    """

    df["prod_index"] = df.index
    df["product_id"] = df["prod_index"].map(index_to_id_dict)
    df = df.merge(prod_lookup, on=["product_id"], how="left")

    return df


def get_model_weights(model: keras.engine.training.Model, layer: int) -> pd.DataFrame:
    """ Function to get model weights from trained model and return them as a Pandas DataFrame

        Parameters
        ----------
        model : keras.engine.training.Model
            trained model
        layer : int
            layer to extract weights from

        Returns
        -------
        embed_weights_pd : pandas.DataFrame
            DataFrame containing the weights from the specified layer

    """

    embed_weights = model.layers[layer].get_weights()
    embed_weights_pd = pd.DataFrame(embed_weights[0])

    return embed_weights_pd


def save_embeddings(
    hpo_enabled: bool,
    job_name: str,
    bucket: str,
    products_key: str,
    run_id: str,
    model_save_loc: str,
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
        products_key : str
            Name of the csv file with the product ID to product description lookup
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

    # Read the product ID to product description data
    products = read_csv_from_s3(bucket=bucket, key=products_key)

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

    with tarfile.open("{}/model.tar.gz".format(model_save_loc)) as tar:
        tar.extractall(path="{}/".format(model_save_loc))

    # Load the model
    model = load_model("{}/prod2vec_model".format(model_save_loc))

    # Get the embedding weights
    embed_weights = get_model_weights(model, 2)

    # Map embedding to product_id and description
    embed_weights_prods = map_weights_to_prods(
        embed_weights, products, reversed_dictionary
    )

    # Upload embeddings to s3 as csv
    embed_file_name = "prod2vec_embed_weights_run_{}".format(run_id)
    embed_weights_prods.to_csv(
        "s3://{}/{}.csv".format(bucket, embed_file_name), index=False
    )
