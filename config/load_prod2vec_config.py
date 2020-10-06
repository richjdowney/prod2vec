from utils.logging_framework import log
import yaml
import pydantic


class ConfigDefaultArgs(pydantic.BaseModel):
    """Configuration for the default args when setting up the DAG"""

    owner: str
    start_date: str
    end_date: str
    depends_on_past: bool
    retries: int
    catchup: bool
    email: str
    email_on_failure: bool
    email_on_retry: bool


class ConfigDag(pydantic.BaseModel):
    """Configuration for the DAG runs"""

    # Name for the DAG run
    dag_id: str

    # Default args for DAG run e.g. owner, start_date, end_date
    default_args: ConfigDefaultArgs

    # DAG schedule interval
    schedule_interval: str


class ConfigS3(pydantic.BaseModel):
    """Configuration for the s3 buckets and keys"""

    Bucket: str
    Orders_key: str
    Products_key: str
    Couples_list_train_key: str
    Couples_list_valid_key: str
    Labels_list_train_key: str
    Labels_list_valid_key: str
    Prod_target_train_key: str
    Prod_target_valid_key: str
    Prod_context_train_key: str
    Prod_context_valid_key: str
    Reversed_dict_key: str
    Products_dict_key: str


class ConfigPreprocess(pydantic.BaseModel):
    """Configuration for the data pre-processing constants"""

    num_prods: int
    train_ratio: float


class ConfigModelTrainConstants(pydantic.BaseModel):
    """Configuration for the model training constants"""

    Entry_point: str
    Epochs: int
    Valid_size: int
    Valid_window: int
    Num_folds: int


class ConfigHyperparameter(pydantic.BaseModel):
    """Configuration for hyper-parameter tuning"""

    Epochs: int
    Min_embeddings: int
    Max_embeddings: int
    Min_learning_rate: float
    Max_learning_rate: float


class Config(pydantic.BaseModel):
    """Main configuration"""

    s3: ConfigS3
    preprocess_constants: ConfigPreprocess
    model_train_constants: ConfigModelTrainConstants
    hyperparameter: ConfigHyperparameter
    dag: ConfigDag


class ConfigException(Exception):
    pass


def load_yaml(config_path):

    """Function to load yaml file from path

    Parameters
    ----------
    config_path : str
        string containing path to yaml

    Returns
    ----------
    config : dict
        dictionary containing config

    """

    log.info("Importing config file from {}".format(config_path))

    if config_path is not None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        log.info("Successfully imported the config file from {}".format(config_path))

    if config_path is None:
        raise ConfigException("Must supply path to the config file")

    return config
