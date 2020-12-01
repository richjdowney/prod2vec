import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dot, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import pickle
import numpy as np
import pandas as pd


class SimilarityCallback:
    """Creates top k nearest products for each validation target product"""

    def run_sim(self):
        valid_examples = np.random.choice(
            args.valid_window, args.valid_size, replace=False
        )
        embed_weights = prod2vec.get_layer("p2v_embedding").get_weights()[0]

        for i in range(args.valid_size):
            # Get the product to validate and obtain its embedding
            valid_prod = reversed_dictionary[valid_examples[i]]
            valid_embed = embed_weights[valid_examples[i]].reshape(1, -1)

            # Calculate cosine similarity between the validation product and all other prods
            cosine_sim = cosine_similarity(
                embed_weights, Y=valid_embed, dense_output=True
            )
            cosine_sim_pd = pd.DataFrame(cosine_sim)
            cosine_sim_pd.columns = ["cosine_sim"]

            # Get the products that are most similar to the validation product
            cosine_sim_pd.loc[:, "index"] = cosine_sim_pd.index
            cosine_sim_pd.sort_values("cosine_sim", ascending=False, inplace=True)
            cosine_sim_pd = cosine_sim_pd[cosine_sim_pd["index"] != valid_examples[i]]
            cosine_sim_pd.loc[:, "valid_prod"] = cosine_sim_pd["index"].map(
                reversed_dictionary
            )
            cosine_sim_pd.loc[:, "prod_desc"] = cosine_sim_pd["valid_prod"].map(
                products_dict
            )

            # Write out the product descriptions
            valid_desc = products_dict.get(valid_prod)
            nearest_desc = cosine_sim_pd["prod_desc"][:20].str.cat(sep="; ")
            print("Nearest to {}: {}".format(valid_desc, nearest_desc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_prods", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--embedding_dim", type=int)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid_size", type=int)
    parser.add_argument("--valid_window", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_sampled", type=int)

    args = parser.parse_args()
    print("Received arguments {}".format(args))

    # Loading the list pickles
    with open(os.path.join(args.train, "targets.txt"), "rb") as fp:  # Unpickling
        targets = pickle.load(fp)

    with open(os.path.join(args.train, "contexts.txt"), "rb") as fp:  # Unpickling
        contexts = pickle.load(fp)

    with open(os.path.join(args.train, "labels.txt"), "rb") as fp:  # Unpickling
        labels = pickle.load(fp)

    # Loading the integer to product_id dictionary
    with open(
        os.path.join(args.train, "reversed_dictionary.pkl"), "rb"
    ) as fp:  # Unpickling
        reversed_dictionary = pickle.load(fp)

    # Loading the product_id to product_description dictionary
    with open(os.path.join(args.train, "products_dict.pkl"), "rb") as fp:  # Unpickling
        products_dict = pickle.load(fp)

    BATCH_SIZE = args.batch_size
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    class Prod2Vec(Model):
        """Create the prod2vec model """

        def __init__(self, num_prods, embedding_dim):
            super(Prod2Vec, self).__init__()
            self.target_embedding = Embedding(
                num_prods, embedding_dim, input_length=1, name="p2v_embedding"
            )

            self.context_embedding = Embedding(
                num_prods, embedding_dim, input_length=args.num_sampled + 1
            )
            self.dots = Dot(axes=(3, 2))
            self.flatten = Flatten()

        def call(self, pair):
            target, context = pair
            we = self.target_embedding(target)
            ce = self.context_embedding(context)
            dots = self.dots([ce, we])
            return self.flatten(dots)

    prod2vec = Prod2Vec(args.num_prods, args.embedding_dim)
    otptim_rmsprop = RMSprop(learning_rate=args.learning_rate)
    prod2vec.compile(
        optimizer=otptim_rmsprop,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    )

    prod2vec.fit(dataset, epochs=args.epochs)

    sim_cb = SimilarityCallback()
    sim_cb.run_sim()

    prod2vec.save(os.path.join("/opt/ml/model/", "prod2vec_model"))
