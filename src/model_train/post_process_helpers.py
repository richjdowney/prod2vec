import pandas
import keras


def map_weights_to_prods(
    df: pandas.DataFrame, prod_lookup: pandas.DataFrame, index_to_id_dict: dict
) -> pandas.DataFrame:
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


def get_model_weights(model: keras.engine.training.Model, layer: int) -> pandas.DataFrame:
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
