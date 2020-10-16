import sys
sys.path.insert(1, '/home/ubuntu/prod2vec')

import pandas as pd
import random
import boto3
import collections
import numpy as np
from keras.preprocessing import sequence
import pickle
import pandas
from utils.logging_framework import log
from config.load_prod2vec_config import load_yaml
from config import constants


def read_csv_from_s3(bucket: str, key: str) -> pandas.DataFrame:
    """ Function to read a csv from s3 and return a pandas DataFrame

        Parameters
        ----------
        bucket : str
            s3 bucket to read from
        key : str
            key within the s3 bucket to read from

        Returns
        -------
        df : pandas.DataFrame
           Pandas DataFrame containing the data read from the s3 bucket

        """

    s3c = boto3.client("s3")

    obj = s3c.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"])

    return df


def create_dict_from_df(df: pandas.DataFrame, key: str, value: str) -> dict:
    """ Function to create dictionary from Pandas DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the key and value pairs
        key : str
            name of the column containing the dictionary key
        value : str
            name of the column containing the dictionary value

        Returns
        -------
        out_dict : dict
            Dictionary containing specified columns in the DataFrame as a dictionary

        """

    out_dict = df[[key, value]]
    out_dict = out_dict.set_index(key)[value].to_dict()

    return out_dict


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


def create_target_context(
    basks: list,
    num_prods: int,
    train_window_size: int,
    max_bask_length: int,
    num_samples=0,
) -> tuple:
    """ Function to create tuples of target / context pairs with associated labels.  This function
        utilizes the keras skipgrams function which generates positive and negative target / context pairs
        utilizing negative sampling to obtain example negative targets

        Parameters
        ----------
        basks : list
            list of baskets with each list containing the index of the product_id contained in the basket
        num_samples : int
            the number of baskets to sample (if samping is desired)
        train_window_size : int
            window size to select context. To speed up processing it may be beneficial to choose a smaller
            context within the (randomly shuffled) items in the basket
        max_bask_length : int
            the maximum number of items to train from each basket - speeds up processing by capping
            very large baskets that cause the skipgram function to be very slow
        num_prods : int
            the number of products on which to train the embeddings e.g. top X products

        Returns
        -------
        prod_target : array
            array containing the product indices used for targets
        prod_context : array
            array containing the product indices used for context
        couples_list : list
            list containing the target / context pairs
        labels_list : list
            list containing the associated labels for the target / context pairs

        """

    sampling_table = sequence.make_sampling_table(num_prods)

    if num_samples > 0:
        # Check if the number of samples requested is < the number of baskets
        assert num_samples < len(
            basks
        ), "Number of samples {} requested is > the number of available baskets {}".format(
            num_samples, len(basks)
        )

        basks = random.sample(basks, num_samples)

    couples_list = []
    labels_list = []

    # need to split the baskets into smaller chunks to run through the skipgrams method
    # runs out of memory if trying to process all baskets together
    chunks_of_basks = zip(*(iter(basks),) * 1000)  # maximum of 1000 baskets per group

    for chunk in chunks_of_basks:

        couples_list_chunk = []
        labels_list_chunk = []

        for basket in chunk:
            # shuffle the items in the basket
            random.shuffle(basket)

            # cap the maximum basket length (number of items in basket)
            basket = basket[0:max_bask_length]

            couples, labels = sequence.skipgrams(
                basket,
                num_prods,
                window_size=train_window_size,
                sampling_table=sampling_table,
                shuffle=True,
            )
            couples_list_chunk = [*couples_list_chunk, *couples]
            labels_list_chunk = [*labels_list_chunk, *labels]

    couples_list = couples_list + couples_list_chunk
    labels_list = labels_list + labels_list_chunk

    # Shuffle the final output so that consecutive pairs wont be from the same basket
    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(couples_list)
    random.seed(seed)
    random.shuffle(labels_list)

    prod_target, prod_context = zip(*couples_list)
    prod_target = np.array(prod_target, dtype="int32")
    prod_context = np.array(prod_context, dtype="int32")

    return prod_target, prod_context, couples_list, labels_list


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


def create_train_validation(input_list: list, train_ratio: float) -> tuple:
    """ Function to create training and validation sets from the target/context pairs lists.  NOTE:
        lists should be shuffled before being input into the function to avoid order bias

        Parameters
        ----------
        input_list : list
            list of targets, context, target/context pairs or labels
        train_ratio : float
            percentage of observations to use for training

        Returns
        -------
        train_list : list
            the list to use for training
        valid_list : list
            the list to use for validation

        """

    train_size = int(len(input_list) * train_ratio)
    train_list = input_list[0:train_size]
    valid_list = input_list[train_size + 1 :]

    return train_list, valid_list


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
    prod_target, prod_context, couples_list, labels_list = create_target_context(
        basks=all_basket_data,
        num_prods=config["preprocess_constants"]["num_prods"],
        train_window_size=config["preprocess_constants"]["train_window_size"],
        max_bask_length=50,
        num_samples=0,
    )

    log.info(couples_list[:10], labels_list[:10])

    # Split data into training and validation sets
    log.info("Splitting data into training and validation sets")
    train_ratio = config["preprocess_constants"]["train_ratio"]
    prod_target_train, prod_target_valid = create_train_validation(prod_target, train_ratio)
    prod_context_train, prod_context_valid = create_train_validation(
        prod_context, train_ratio
    )
    couples_list_train, couples_list_valid = create_train_validation(
        couples_list, train_ratio
    )
    labels_list_train, labels_list_valid = create_train_validation(labels_list, train_ratio)

    log.info("Uploading training and validation data and dictionary mappings to s3")
    # Upload the training and validation data and the dictionary mappings to S3
    training_data_to_s3(
        obj=couples_list_train,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["couples_list_train_key"],
    )

    training_data_to_s3(
        obj=couples_list_valid,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["couples_list_valid_key"],
    )

    training_data_to_s3(
        obj=labels_list_train,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["labels_list_train_key"],
    )

    training_data_to_s3(
        obj=labels_list_valid,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["labels_list_valid_key"],
    )

    training_data_to_s3(
        obj=prod_target_train,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["prod_target_train_key"],
    )

    training_data_to_s3(
        obj=prod_target_valid,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["prod_target_valid_key"],
    )

    training_data_to_s3(
        obj=prod_context_train,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["prod_context_train_key"],
    )

    training_data_to_s3(
        obj=prod_context_valid,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["prod_context_valid_key"],
    )

    training_data_to_s3(
        obj=reversed_dictionary,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["reversed_dict_key"],
    )

    training_data_to_s3(
        obj=products_dict,
        bucket=config["s3"]["bucket"],
        key=config["s3"]["products_dict_key"],
    )
