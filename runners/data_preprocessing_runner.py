import sys
sys.path.insert(1, '/home/ubuntu/prod2vec')


from src.data_preprocessing.data_prep_helpers import *
from utils.logging_framework import log
from config.load_prod2vec_config import load_yaml
from config import constants

config = load_yaml(constants.config_path)

# Read the orders and products data from s3
log.info("Reading orders and products data from s3")
orders_data = read_csv_from_s3(
    bucket=config["s3"]["Bucket"], key=config["s3"]["Orders_key"]
)

products = read_csv_from_s3(
    bucket=config["s3"]["Bucket"], key=config["s3"]["Products_key"]
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
    key=config["s3"]["Couples_list_train_key"],
)

training_data_to_s3(
    obj=couples_list_valid,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Couples_list_valid_key"],
)

training_data_to_s3(
    obj=labels_list_train,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Labels_list_train_key"],
)

training_data_to_s3(
    obj=labels_list_valid,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Labels_list_valid_key"],
)

training_data_to_s3(
    obj=prod_target_train,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Prod_target_train_key"],
)

training_data_to_s3(
    obj=prod_target_valid,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Prod_target_valid_key"],
)

training_data_to_s3(
    obj=prod_context_train,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Prod_context_train_key"],
)

training_data_to_s3(
    obj=prod_context_valid,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Prod_context_valid_key"],
)

training_data_to_s3(
    obj=reversed_dictionary,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Reversed_dict_key"],
)

training_data_to_s3(
    obj=products_dict,
    bucket=config["s3"]["bucket"],
    key=config["s3"]["Products_dict_key"],
)
