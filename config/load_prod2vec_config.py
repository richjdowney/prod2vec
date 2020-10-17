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

    bucket: str
    orders_key: str
    products_key: str
    couples_list_train_key: str
    couples_list_valid_key: str
    labels_list_train_key: str
    labels_list_valid_key: str
    prod_target_train_key: str
    prod_target_valid_key: str
    prod_context_train_key: str
    prod_context_valid_key: str
    reversed_dict_key: str
    products_dict_key: str


class ConfigPreprocess(pydantic.BaseModel):
    """Configuration for the data pre-processing constants"""

    num_prods: int
    train_ratio: float


class ConfigStaticParams(pydantic.BaseModel):
    """Configuration for the estimator parameters shared across tuning and training"""
    run_id: str
    run_hyperparameter_opt: str
    epochs: int
    num_folds: int
    valid_size: int
    valid_window: int
    cross_validate: bool
    metric_definitions: list
    objective_metric_name: str
    objective_type: str


class ConfigTuningHyperparamaters(pydantic.BaseModel):
    """Configuration for the tuning hyper-parameters"""

    max_jobs: int
    max_parallel_jobs: int
    min_embeddings: int
    max_embeddings: int
    min_learning_rate: float
    max_learning_rate: float


class ConfigTrainHyperparamaters(pydantic.BaseModel):
    embeddings: int
    learning_rate: float


class ConfigTrainInput(pydantic.BaseModel):
    """Configuration for the tuning hyperparameters"""

    train: str


class ConfigEstimator(pydantic.BaseModel):
    """Configuration for the model training constants"""

    static_params: ConfigStaticParams
    train_hyperparameters: ConfigTrainHyperparamaters
    tune_hyperparameters: ConfigTuningHyperparamaters
    inputs: ConfigTrainInput


class ConfigPostProcess(pydantic.BaseModel):
    """Configuration for post-processing"""
    model_artifact: str

class Config(pydantic.BaseModel):
    """Main configuration"""

    s3: ConfigS3
    preprocess_constants: ConfigPreprocess
    estimator_config: ConfigEstimator
    dag: ConfigDag
    post_process_config: ConfigPostProcess


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
