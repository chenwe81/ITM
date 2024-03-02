import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
import plot_joint
 
 
# Load 'dist2d.csv' into a variable X
X = pd.read_csv('/mnt/home/chenwe81/ITM/assignment 2/dist2d.csv').values
 
# Use the plot_joint function to scatter plot X and the corresponding projected distributions of the data
plot_joint.plot_joint(X)
 %
# Center X and plot again
X_centered = X - np.mean(X, axis=0)
plot_joint.plot_joint(X_centered)
 
# Use PCA from sklearn.decomposition to do a 2-dimensional principal component decomposition of the centered X and plot the result
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)
plot_joint.plot_joint(X_pca)
 
# Use FastICA from sklearn.decomposition to do a 2-dimensional independent component decomposition of the centered X and plot again
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X_centered)
plot_joint.plot_joint(X_ica)