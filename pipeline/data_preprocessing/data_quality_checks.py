import sys
sys.path.insert(1, '/home/ubuntu/prod2vec')

from pipeline.data_preprocessing.data_prep import *


def check_miss(df: pd.DataFrame, col: str):
    """ Function to check a DataFrame / column combination for missing values

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to run the quality check
        col : str
            column to check for missing values

        Raises
        -------
        ValueError

    """

    if np.sum(df[col].isnull()) != 0:
        raise ValueError(
            "Data quality check failed, column {} has missing values".format(col, df)
        )


def check_product_counts(orders_data: pd.DataFrame, products: pd.DataFrame):
    """ Function to check product counts in orders and product data

        Parameters
        ----------
        orders_data : pandas.DataFrame
            DataFrame for orders
        products : pandas.DataFrame
            DataFrame for products

        Raises
        -------
        ValueError

    """

    num_prods_orders = orders_data[["product_id"]].drop_duplicates().shape[0]
    num_prods_products = products[["product_id"]].drop_duplicates().shape[0]

    # Check range of products are +/- 10% of current observed values in each DataFrame
    if (num_prods_orders <= 35000) | (num_prods_orders >= 44000):
        raise ValueError(
            "Data quality check failed, number of products in orders data outside of expected range"
        )

    if (num_prods_products <= 45000) | (num_prods_products >= 55000):
        raise ValueError(
            "Data quality check failed, number of products in products data outside of expected range"
        )


def check_column_counts(df: pd.DataFrame, expected_col_count: int):
    """ Function to check number of columns in a DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to check
        expected_col_count : int
            Expected number of columns in the DataFrame

        Raises
        -------
        ValueError

    """

    col_count = df.shape[1]

    if col_count != expected_col_count:
        raise ValueError(
            "Data quality check failed, number of columns != expected value {}".format(
                df, col_count, expected_col_count
            )
        )


def check_dtypes(df: pd.DataFrame, expected_dtypes: dict):
    """ Function to check data types from a DataFrame match expected values

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to check
        expected_dtypes : dict
            Dictionary of expected data types

        Raises
        -------
        ValueError

    """

    df_dtypes = df.dtypes.apply(lambda x: x.name).to_dict()

    if df_dtypes != expected_dtypes:
        raise ValueError(
            "Data quality check failed, data types do not match expected values".format(
                df
            )
        )


def check_records_in_products_dict(dict_to_check: dict, products_df: pd.DataFrame):
    """ Function to check number of records in products dictionary matches number of products in
        products DataFrame

        Parameters
        ----------
        dict_to_check : dict
            Products dictionary
        products_df : pandas.DataFrame
            DataFrame containing the product ID and product descriptions for all products

        Raises
        -------
        ValueError
    """

    if len(dict_to_check) != products_df.shape[0]:
        raise ValueError(
            "Data quality check failed, number of records in dictionary does not match number of products".format(
                dict_to_check
            )
        )


def run_data_quality_checks():

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

    log.info("Checking orders and products data for missing values on key columns")
    # Check key columns in orders_data for missing values
    check_miss(orders_data, "order_id")
    check_miss(orders_data, "product_id")

    # Check key columns in products for missing values
    check_miss(products, "product_id")

    log.info("Checking counts of unique products in orders and products data")
    check_product_counts(orders_data, products)

    log.info("Checking orders and products data has the expected number of columns")
    check_column_counts(products, 4)
    check_column_counts(orders_data, 4)

    log.info("Checking data types of orders and products DataFrames")
    # Products
    expected_dtypes_products = {
        "product_id": "int64",
        "product_name": "object",
        "aisle_id": "int64",
        "department_id": "int64",
    }
    check_dtypes(products, expected_dtypes_products)

    # Orders
    expected_dtypes_orders = {
        "order_id": "int64",
        "product_id": "int64",
        "add_to_cart_order": "int64",
        "reordered": "int64",
    }
    check_dtypes(orders_data, expected_dtypes_orders)

    log.info(
        "Checking the number of products in products dictionary matches products DataFrame"
    )
    products_dict = create_dict_from_df(products, "product_id", "product_name")
    check_records_in_products_dict(products_dict, products)
