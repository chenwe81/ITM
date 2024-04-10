import matplotlib.pyplot as plt
import seaborn as sns

def PlotSignificance(score, perm_scores, pvalue):
    """
    Plot the significance of a score by comparing it to a distribution of permuted scores.

    Parameters:
    score (float): The score on the original data.
    perm_scores (list): A list of permuted scores.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    sns.histplot(perm_scores, kde=True, ax=ax, bins=100, stat="density")
    ax.axvline(score, ls="--", color="r")
    score_label = f"Score on original data: {score:.2f}\n(p-value: {pvalue:.4f})"
    ax.set_title(score_label)
    ax.set_xlabel("Prediction score")
    _ = ax.set_ylabel("Probability density")
    plt.show()
