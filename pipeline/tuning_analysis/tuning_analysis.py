import sys

sys.path.insert(1, "/home/ubuntu/prod2vec")

from sagemaker.analytics import HyperparameterTuningJobAnalytics
from pipeline.tuning_analysis.plot_helpers import *
from utils.logging_framework import log
from config.load_prod2vec_config import load_yaml
from config import constants

log.info("Reading config")
config = load_yaml(constants.config_path)


def get_top_tuning_jobs(tuning_job_name: str):
    """ Function to save the hyperparameter tuning analysis to s3

        Parameters
        ----------
        tuning_job_name : str
            Name of the hyper-parameter tuning job to analyze

        Returns
        -------
        tuning_analysis : pandas.DataFrame
            Pandas DataFrame containing the tuning analysis

    """

    # get tuning analytics
    tuner = HyperparameterTuningJobAnalytics(tuning_job_name)
    tuning_analysis = tuner.dataframe()
    tuning_analysis.sort_values(by=["TrainingStartTime"], ascending=False, axis=0)
    tuning_analysis["iteration"] = tuning_analysis.index
    tuning_analysis.sort_values(
        by=["FinalObjectiveValue"], ascending=True, axis=0, inplace=True
    )

    log.info("writing tuning analysis to s3")
    tuning_analysis.to_csv(
        "s3://{}/{}.csv".format(config["s3"]["bucket"], tuning_job_name), index=False
    )

    return tuning_analysis


def run_tuning_analysis(tuning_job_name: str):
    """run the tuning analysis"""

    tuning_analysis = get_top_tuning_jobs(tuning_job_name)

    # Plot the value of hyper-parameters over the tuning window
    plot_hyperparams_over_search(
        df=tuning_analysis,
        hyperparams=["num_embeddings", "learning_rate"],
        bucket=config["s3"]["bucket"],
        key="params_over_search.png",
    )

    # Create a kde plot of hyper-parameters used over the tuning window
    plot_search_dist(
        df=tuning_analysis,
        hyperparams=["num_embeddings", "learning_rate"],
        bucket=config["s3"]["bucket"],
        key="search_kde.png",
    )
