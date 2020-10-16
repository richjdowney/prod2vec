# Keras
from keras.layers import Embedding, Input, Reshape, Dot, Dense
from keras import Model
from keras.optimizers import Adam

# Other packages
import numpy as np
import argparse
import os
import pickle

# Sklearn
from sklearn import metrics


class SimilarityCallback:
    """Creates top k nearest products for each validation target product"""

    def run_sim(self):
        valid_examples = np.random.choice(
            args.valid_window, args.valid_size, replace=False
        )
        for prod in range(args.valid_size):
            valid_prod = reversed_dictionary[valid_examples[prod]]
            valid_prod = products_dict.get(valid_prod)
            top_k = 10  # number of nearest neighbors
            sim = self._get_sim(valid_examples[prod])
            nearest = (-sim).argsort()[1: top_k + 1]
            log_str = "Nearest to %s:" % valid_prod
            for k in range(top_k):
                close_prod = reversed_dictionary[nearest[k]]
                close_prod = products_dict.get(close_prod)
                log_str = "%s %s," % (log_str, close_prod)
            print(log_str)

    @staticmethod
    def _get_sim(valid_prod_idx):
        sim = np.zeros((args.num_prods,))
        in_arr1 = np.zeros((1, ))
        in_arr2 = np.zeros((1, ))
        for prod in range(args.num_prods):
            in_arr1[0, ] = valid_prod_idx
            in_arr2[0, ] = prod
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[prod] = out
        return sim


sim_cb = SimilarityCallback()


def create_cv_folds(input_array: np.ndarray, num_folds: iter) -> iter:
    """ Takes input lists and creates an iter object with the folds

        Parameters
        ----------
        input_array : ndarray
            list of targets, context, target/context pairs or labels
        num_folds : int
            the number of desired folds for cross validation

        Returns
        -------
        folds_list : iter
            an iterator containing the folds for each input

        """

    num_per_fold = int(np.floor(len(input_array) / num_folds))
    folds_list = zip(*(iter(input_array),) * num_per_fold)

    return folds_list


def prod2vec_model(num_prods: int, embed_size: int) -> tuple:
    """ Creates the prod2vec model by creating target and context embeddings, calculating the cosine similarity of
        the embeddings and passing to a sigmoid layer to predict the label (correct target and context pair)

        Parameters
        ----------
        num_prods : int
            the number of products on which to train the embeddings e.g. top X products
        embed_size : int
            the size of the target and context embeddings

        Returns
        -------
        model : keras.Model
            prod2vec model
        validation_model : keras.Model
            Model utilized for validation - outputs the cosine similarity between the target and context
            embeddings

    """

    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(num_prods, embed_size, input_length=1, name="embedding")

    target = embedding(input_target)
    target = Reshape((embed_size, 1))(target)
    context = embedding(input_context)
    context = Reshape((embed_size, 1))(context)

    # cosine similarity between the target and context embeddings for validation
    similarity = Dot(axes=1, normalize=True)([target, context])

    # cosine similarity between the target and context embeddings to pass to the Sigmoid layer
    dot_product = Dot(axes=1, normalize=True)([target, context])
    dot_product = Reshape((1,))(dot_product)

    # sigmoid output layer
    output = Dense(1, activation="sigmoid")(dot_product)

    # create the primary training model
    train_model = Model(inputs=[input_target, input_context], outputs=output)

    # validation model to run similarity checks during training
    valid_model = Model(inputs=[input_target, input_context], outputs=similarity)

    return train_model, valid_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_prods", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_embeddings", type=int)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid_size", type=int)
    parser.add_argument("--valid_window", type=int)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument(
        "--cross_validate", type=str, default="True"
    )  # Sagemaker is passing the boolean as a string

    args = parser.parse_args()
    print("Received arguments {}".format(args))

    # Loading the list pickles
    with open(
        os.path.join(args.train, "couples_list_train.txt"), "rb"
    ) as fp:  # Unpickling
        couples_list_train = pickle.load(fp)

    with open(
        os.path.join(args.train, "couples_list_valid.txt"), "rb"
    ) as fp:  # Unpickling
        couples_list_valid = pickle.load(fp)

    with open(
        os.path.join(args.train, "labels_list_train.txt"), "rb"
    ) as fp:  # Unpickling
        labels_list_train = pickle.load(fp)

    with open(
        os.path.join(args.train, "labels_list_valid.txt"), "rb"
    ) as fp:  # Unpickling
        labels_list_valid = pickle.load(fp)

    # Loading the integer to product_id dictionary
    with open(
        os.path.join(args.train, "reversed_dictionary.pkl"), "rb"
    ) as fp:  # Unpickling
        reversed_dictionary = pickle.load(fp)

    # Loading the product_id to product_description dictionary
    with open(os.path.join(args.train, "products_dict.pkl"), "rb") as fp:  # Unpickling
        products_dict = pickle.load(fp)

    # Loading the arrays
    prod_target_train = np.loadtxt(
        os.path.join(args.train, "prod_target_train.csv"), delimiter=","
    )
    prod_target_valid = np.loadtxt(
        os.path.join(args.train, "prod_target_valid.csv"), delimiter=","
    )

    prod_context_train = np.loadtxt(
        os.path.join(args.train, "prod_context_train.csv"), delimiter=","
    )
    prod_context_valid = np.loadtxt(
        os.path.join(args.train, "prod_context_valid.csv"), delimiter=","
    )

    model, validation_model = prod2vec_model(
        num_prods=args.num_prods, embed_size=args.num_embeddings
    )

    arr_1 = np.zeros((1,))
    arr_2 = np.zeros((1,))
    arr_3 = np.zeros((1,))

    # Use Adam optimizer
    optim_adam = Adam(learning_rate=args.learning_rate)

    if args.cross_validate == "True":  # Sagemaker passes bool as string

        # Create the cross-validation folds from the training data
        prod_target_folds = create_cv_folds(prod_target_train, args.num_folds)
        prod_context_folds = create_cv_folds(prod_context_train, args.num_folds)
        couples_list_folds = create_cv_folds(couples_list_train, args.num_folds)
        labels_list_folds = create_cv_folds(labels_list_train, args.num_folds)

        loss_per_fold = []  # array to hold loss for each fold

        for fold in range(0, args.num_folds):
            model.compile(loss="binary_crossentropy", optimizer=optim_adam)
            labels_list = next(labels_list_folds)
            prod_target = next(prod_target_folds)
            prod_context = next(prod_context_folds)

            for cnt in range(args.epochs):
                idx = np.random.randint(0, len(labels_list) - 1)
                arr_1[0, ] = prod_target[idx]
                arr_2[0, ] = prod_context[idx]
                arr_3[0, ] = labels_list[idx]
                train_loss = model.train_on_batch([arr_1, arr_2], arr_3)

                if cnt % 100 == 0:  # Print loss and validation loss every 100 epochs

                    pred_prob = []

                    for i in range(0, len(prod_target_valid)):
                        arr_1[0, ] = prod_target_valid[i]
                        arr_2[0, ] = prod_context_valid[i]
                        pred = model.predict([arr_1, arr_2])
                        pred_prob.append(float(pred[0]))

                    valid_loss = metrics.log_loss(list(labels_list_valid), pred_prob)
                    print(
                        "Fold {}, iteration {}, train_loss={}".format(
                            fold + 1, cnt, train_loss
                        )
                    )
                    print(
                        "Fold {}, iteration {}, valid_loss={}".format(
                            fold + 1, cnt, valid_loss
                        )
                    )

                    loss_per_fold.append(valid_loss)

        sim_cb.run_sim()
        avg_valid_loss = np.mean(loss_per_fold)
        print("loss: {}".format(avg_valid_loss))

    else:

        model.compile(loss="binary_crossentropy", optimizer=optim_adam)
        labels_list = labels_list_train
        prod_target = prod_target_train
        prod_context = prod_context_train

        for cnt in range(args.epochs):
            idx = np.random.randint(0, len(labels_list) - 1)
            arr_1[0, ] = prod_target[idx]
            arr_2[0, ] = prod_context[idx]
            arr_3[0, ] = labels_list[idx]
            train_loss = model.train_on_batch([arr_1, arr_2], arr_3)

            if cnt % 100 == 0:  # Print loss and validation loss every 100 epochs

                pred_prob = []

                for i in range(0, len(prod_target_valid)):
                    arr_1[0, ] = prod_target_valid[i]
                    arr_2[0, ] = prod_context_valid[i]
                    pred = model.predict([arr_1, arr_2])
                    pred_prob.append(float(pred[0]))

                valid_loss = metrics.log_loss(list(labels_list_valid), pred_prob)
                print("iteration {}, valid_loss={}".format(cnt, valid_loss))

        sim_cb.run_sim()

        # Get the final validation loss
        pred_prob = []

        for i in range(0, len(prod_target_valid)):
            arr_1[0, ] = prod_target_valid[i]
            arr_2[0, ] = prod_context_valid[i]
            pred = model.predict([arr_1, arr_2])
            pred_prob.append(float(pred[0]))

        valid_loss = metrics.log_loss(list(labels_list_valid), pred_prob)
        print("loss: {}".format(valid_loss))

        model.save(os.path.join("/opt/ml/model/", "prod2vec_model"))
