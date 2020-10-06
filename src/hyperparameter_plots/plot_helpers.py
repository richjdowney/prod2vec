import matplotlib.pyplot as plt
import seaborn as sns
import pandas


def plot_hyperparams_over_search(
    df: pandas.DataFrame, hyperparams: list
) -> sns.regplot:
    """Function to plot the value of hyper-parameters over the tuning window

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the hyper-parameter search results
    hyperparams : list
        list of hyper-parameters used for tuning
    Returns
    -------
        sns.regplot object
    """

    hyperparams = hyperparams
    fig, axs = plt.subplots(1, len(hyperparams), figsize=(24, 6))

    if len(hyperparams) > 1:
        for i, hyper in enumerate(hyperparams):
            sns.regplot("iteration", hyper, data=df, ax=axs[i])
            axs[i].set(
                xlabel="Iteration",
                ylabel="{}".format(hyper),
                title="{} over Search".format(hyper),
            )

    else:
        sns.regplot("iteration", hyperparams[0], data=df)
        axs.set(
            xlabel="Iteration",
            ylabel="{}".format(hyperparams[0]),
            title="{} over Search".format(hyperparams[0]),
        )

    return plt.tight_layout()


def plot_search_dist(df: pandas.DataFrame, hyperparams: list) -> sns.kdeplot:
    """Function to create a kde plot of hyper-parameters used over the tuning window

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the hyper-parameter search results
    hyperparams : list
        list of hyper-parameters used for tuning
    Returns
    -------
        sns.kdeplot object
    """

    hyperparams = hyperparams
    fig, axs = plt.subplots(1, len(hyperparams), figsize=(24, 6))
    i = 0

    if len(hyperparams) > 1:
        for i, hyper in enumerate(hyperparams):
            sns.kdeplot(df[hyperparams[i]], linewidth=2, ax=axs[i])
            axs[i].set(
                xlabel=hyperparams[i],
                ylabel="Density",
                title="{} Search Distribution".format(hyperparams[i]),
            )

    else:
        sns.kdeplot(df[hyperparams[0]], linewidth=2)
        axs.set(
            xlabel=hyperparams[i],
            ylabel="Density",
            title="{} Search Distribution".format(hyperparams[i]),
        )

    return plt.tight_layout()
