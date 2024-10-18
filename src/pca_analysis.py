# Library imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

#  Project imports
from data_preprocessing import load_data, feature_label_split

def number_of_correlated_columns(correlation_matrix, threshold = 0.9):
    correlated_pairs = []
    count = 0
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                count += 1
                correlated_pairs.append((i, j))
                
    return count, correlated_pairs


def cancer_dataframe_PCA(features: pd.DataFrame,
                         n_components = 3,
                         scaler = StandardScaler()):
    pca = PCA(n_components = n_components)

    # Standardize the features before PCA
    features_standardized = scaler.fit_transform(features)
    
    # Fit PCA to the standardized features
    features_PCA_reduced = pca.fit_transform(features_standardized)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance ratio: {explained_variance}')
    print(f'Total variance explained by {len(explained_variance)} components: {sum(explained_variance):.2f}')

    # Components dataframe
    principal_components = []
    for i in range(len(explained_variance)):
        principal_components.append("PC"+str(i+1))
    components = pca.components_.T
    # loadings = components * np.sqrt(pca.explained_variance_[:, np.newaxis]) 
    components_df = pd.DataFrame(components, columns = principal_components)

    return features_PCA_reduced, explained_variance, components_df



def visualize_dataset_3d(points_df_3D: pd.DataFrame,
                         labels_df: pd.DataFrame,
                         reduction_name: str,
                         custom_index_function = (lambda i: i)):
    """
    Visualize points dataset after reducing to 3D.

    Args:
        points_df__3D: Dimension-reduced dataset
        labels_df: The column with which to label the points
        reduction_name: Name of the reduction technique, used for plot title and axes labeling
        custom_index_function: Pass custom_index for labeling all three AJCC stages with the same color
    Returns:
        fig, ax.
    """
    levels, categories = pd.factorize(labels_df)
    colors = [plt.cm.tab10(custom_index_function(i)) for i in levels] # using the "tab10" colormap
    handles = [Patch(color=plt.cm.tab10(custom_index_function(i)), label=c) for i, c in enumerate(categories)]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points in 3D space, color by the cancer detection label (y)
    ax.scatter(points_df_3D[:, 0], points_df_3D[:, 1], points_df_3D[:, 2],
                        c=colors,
                        edgecolor='k',
                        s=30)

    # Label the axes
    
    plt.gca().set(xlabel = reduction_name+' 1',
                  ylabel = reduction_name+' 2',
                  zlabel = reduction_name+' 3',
                  title = reduction_name+f' of {categories[0]} Samples')
    plt.legend(handles=handles,  title=labels_df.name)
    return fig, ax



#  Debug code. PCA of Ovary samples
if __name__ == "__main__":
    categories, dfs = load_data()
    category_index = 6
    features, labels = feature_label_split(dfs[category_index])
    
    features_PCA_reduced, explained_variance, components_df = cancer_dataframe_PCA(features, n_components = 3)
    
    visualize_dataset_3d(points_df_3D = features_PCA_reduced, 
                         labels_df = labels,
                         reduction_name = 'PCA')
    plt.show()