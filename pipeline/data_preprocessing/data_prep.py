import sys

sys.path.insert(1, "/home/ubuntu/prod2vec")

import random
import collections
import numpy as np
import tensorflow as tf
import pickle
import pandas
from utils.logging_framework import log
from utils.util_functions import *
from config.load_prod2vec_config import load_yaml
from config import constants


def create_prod_lists(df: pandas.DataFrame, bask_id: str, prod_id: str) -> tuple:
    """ Function to create list of all products in all baskets and an array of lists containing all
        products purchased in EACH basket

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing unique combinations of baskets and products
        bask_id : str
            name of the column containing the basket ID
        prod_id : str
            name of the column containing the product ID

        Returns
        -------
        product_id_list : list
            list of unique product_id purchased in every basket as a single list
        product_id_group_list : list
            list containing the unique product_id purchased in every basket

        """

    product_id_list = df[prod_id].tolist()
    product_id_group_list = (
        df.groupby(bask_id)[prod_id].apply(pd.Series.tolist).tolist()
    )

    return product_id_list, product_id_group_list


def create_data(prod_list: list, prod_group_list: list, num_prods: int) -> tuple:
    """ Function to create counts of products, a dictionary mapping between product_id and an index,
        a reversed dictionary that maps back index to product_id and the basket data with product_id
        mapped to the index

        Parameters
        ----------
        prod_list : list
            list of unique product_id purchased in every basket as a single list
        prod_group_list : list
            list containing the unique product_id purchased in every basket
        num_prods : int
            the number of products on which to train the embeddings e.g. top X products
            (all others are tagged as "UNK" (unknown))

        Returns
        -------
        all_basket_data : array
           array of lists containing the index of the product_id purchased in every basket
        count : array
            array of tuples containing the product_id and the count of the number of baskets
            the product is found in
        dictionary : dict
            dictionary containing the mapping of product_id to index
        reversed_dictionary : dict
            dictionary containing the reverse mapping of index to product_id


    """

    # Create counts of products
    count = [["UNK", -1]]  # Placeholder for unknown
    count.extend(collections.Counter(prod_list).most_common(num_prods - 1))

    # Create a dictionary mapping of product to index
    dictionary = dict()
    for prod, _ in count:
        dictionary[prod] = len(dictionary)

    # Create a reversed mapping of index to product
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # Get counts for unknown products and map the product index from the dictionary
    # to each product in the basket
    unk_count = 0
    all_basket_data = list()
    for i in range(0, len(prod_group_list)):
        basket_list = list()
        for prod in prod_group_list[i]:
            if prod in dictionary:
                index = dictionary[prod]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            basket_list.append(index)
        all_basket_data.append(basket_list)
    count[0][1] = unk_count

    return all_basket_data, count, dictionary, reversed_dictionary


def generate_training_data(
    sequences: list,
    window_size: int,
    num_ns: int,
    num_prods: int,
    seed: int,
    max_basket_length: int,
) -> tuple:
    """ Function to create training data containing lists of target products, context products
        and labels

        Parameters
        ----------
        sequences : list
            Lists of baskets and unique items purchased in each
        window_size : int
            Size of the window for selecting the context products - for this use case the window
            size should be the length of the basket but to speed up training a smaller window
            is recommended
        num_ns : int
            The number of negative samples to select per positive sample
        num_prods : int
            The number of unique products in the product dictionary (equivalent to the vocab size in NLP)
        seed : int
            The seed for the negative sampling
        max_basket_length : int
            The maximum number of items to consider in each basket, used to speed up training

        Returns
        -------
        targets : list
            List of target products
        contexts : list
            List of context products (including negative samples)
        labels : list
            Binary 0/1 indicator if the target / context pair were a positive sample (1) or negative sample (0)

        """

    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for num_prods tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(num_prods)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in sequences:

        # shuffle the items in the basket
        random.shuffle(sequence)

        # cap the maximum basket length (number of items in basket)
        sequence = sequence[0:max_basket_length]

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=num_prods,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_prod, context_prod in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_prod], dtype="int64"), 1
            )

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=num_prods,
                seed=seed,
                name="negative_sampling",
            )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1
            )

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_prod)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def training_data_to_s3(obj: any, bucket: str, key: str):
    """ Function to upload the training data to s3 - required for Sagemaker to access the files for training

        Parameters
        ----------
        obj : list, np.ndarray, dict
            object to upload, either a list, numpy array or dict
        bucket : str
            name of the s3 bucket
        key : str
            name of the file to upload

        """

    bucket = bucket
    key = key
    s3c = boto3.client("s3")

    if isinstance(obj, list):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, dict):
        with open(key, "wb") as obj_pickle:
            pickle.dump(obj, obj_pickle)
        s3c.upload_file(key, bucket, key)

    if isinstance(obj, np.ndarray):
        np.savetxt(key, obj, delimiter=",")
        s3c.upload_file(key, bucket, key)


def run_data_preprocessing():
    """Runs all data pre-processing steps"""

    log.info("Reading config")
    config = load_yaml(constants.config_path)

    # Read the orders and products data from s3
    log.info("Reading orders and products data from s3")
    orders_data = read_csv_from_s3(
        bucket=config["s3"]["bucket"], key=config["s3"]["orders_key"]
    )

    products = read_csv_from_s3(
        bucket=config["s3"]["bucket"], key=config["s3"]["products_key"]
    )

    log.info(
        "DataFrame has {} unique baskets and {} unique products".format(
            orders_data[["order_id"]].drop_duplicates().shape[0],
            orders_data[["product_id"]].drop_duplicates().shape[0],
        )
    )

    # Create dictionary of product_id to product_name
    log.info("Creating dictionary of product_id to product_name")
    products_dict = create_dict_from_df(products, "product_id", "product_name")

    # Create 2 lists:
    # 1.) list of all products in all baskets
    # 2.) array of lists containing all products in each basket
    log.info("Creating product_id_list and product_id_group_list")
    product_id_list, product_id_group_list = create_prod_lists(
        df=orders_data, bask_id="order_id", prod_id="product_id"
    )

    # create a list of the basket data with the product ID replaced by the index, the counts of number
    # of times a product appeared in a basket, the mappings of product ID to index and index to product ID
    log.info("Creating dictionaries of index to product_id and product_id to index")
    all_basket_data, count, dictionary, reversed_dictionary = create_data(
        product_id_list,
        product_id_group_list,
        num_prods=config["preprocess_constants"]["num_prods"],
    )

    # Create the lists required for model training, target and context pairs and labels
    log.info("Creating target and context pairs using Keras Skipgrams")
    targets, contexts, labels = generate_training_data(
        sequences=all_basket_data,
        window_size=config["preprocess_constants"]["train_window_size"],
        num_ns=config["preprocess_constants"]["num_ns"],
        num_prods=config["preprocess_constants"]["num_prods"],
        seed=1,
        max_basket_length=config["preprocess_constants"]["max_basket_length"],
    )

    log.info("Uploading training data and dictionary mappings to s3")
    # Upload the training data and the dictionary mappings to S3
    training_data_to_s3(obj=targets, bucket=config["s3"]["bucket"], key="targets.txt")

    # Upload context to s3
    training_data_to_s3(
        obj=contexts, bucket=config["s3"]["bucket"], key="contexts.txt"
    )

    # Upload labels to s3
    training_data_to_s3(obj=labels, bucket=config["s3"]["bucket"], key="labels.txt")

    # Upload the product index to product ID dictionary to s3
    training_data_to_s3(
        obj=reversed_dictionary,
        bucket=config["s3"]["bucket"],
        key="reversed_dictionary.pkl",
    )

    # Upload the product ID to product description dictionary to s3
    training_data_to_s3(
        obj=products_dict, bucket=config["s3"]["bucket"], key="products_dict.pkl"
    )
