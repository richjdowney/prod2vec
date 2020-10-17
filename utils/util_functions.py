import boto3
import pandas as pd


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
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


def create_dict_from_df(df: pd.DataFrame, key: str, value: str) -> dict:
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
