import numpy as np
import matplotlib.pyplot as plt

def plot_joint(data):
    """
    Plots a scatterplot of a 2-dimensional array along with histogram projections
    on the x and y axes.

    Parameters:
        data (numpy.ndarray): The 2-dimensional array of data.

    Returns:
        None
    """
    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Scatterplot
    ax[1, 0].scatter(data[:, 0], data[:, 1], s=1)
    ax[1, 0].set_title('Scatterplot')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')

    # Histogram along x-axis
    ax[0, 0].hist(data[:, 0], bins=100, orientation='vertical')
    ax[0, 0].set_title('Histogram along X-axis')
    ax[0, 0].set_xlabel('Frequency')
    ax[0, 0].set_ylabel('X')

    # Histogram along y-axis (moved to the right side)
    ax[1, 1].hist(data[:, 1], bins=100, orientation='horizontal')
    ax[1, 1].set_title('Histogram along Y-axis')
    ax[1, 1].set_xlabel('Y')
    ax[1, 1].set_ylabel('Frequency')

    # Remove the empty subplot in the bottom left corner
    fig.delaxes(ax[0, 1])

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

