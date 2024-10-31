# Library imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from data_preprocessing import load_data, feature_label_split

def descriptive_statistics(categories, dfs, biomarker_index):   
    """
    Do descriptive statistics on the biomarkers for each cancer type.

    Parameters
    ----------
    categories : list
        The list of cancer types.
    dfs : list
        The list of dataframes corresponding to each cancer type.
    biomarker_index : int
        The index of the biomarker to do the descriptive statistics on.

    Returns
    -------
    None
    """
    
    # Find the list of biomarkers and pick the biomarker with the given biomarker_index
    biomarker_levels, _ = feature_label_split(dfs[0])
    biomarkers = biomarker_levels.columns
    biomarker = biomarkers[biomarker_index]

    # Create a figure with 3 rows and 3 columns
    fig, axs = plt.subplots(3,3, figsize=(15, 15), constrained_layout=True)
    axs = axs.flatten()

    for i, cancer_type in enumerate(categories):
        df = dfs[i]

        # Do the descriptive statistics
        desc_stats = df[biomarker].describe()
        mean = desc_stats['mean']
        std = desc_stats['std']
        cv = std / mean
        quantiles = desc_stats['25%'], desc_stats['50%'], desc_stats['75%']

        # Plot the histogram
        ax = axs[i]
        ax = sns.histplot(df[biomarker].to_numpy(), ax=axs[i], kde = True)

        # Add text box with descriptive statistics: Mean, Std, CV, Q1, Q2, Q3
        ax.text(0.95,
                0.95,
                f"Mean: {mean:.2f}\n Std: {std:.2f}\n CV: {cv:.2f}\n Q1: {quantiles[0]: .2f}\n Q2: {quantiles[1]: .2f}\n Q3: {quantiles[2]: .2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5)
        )

        # Add vertical lines corresponding to the quantiles Q1, Q2, and Q3
        colors = ['green', 'red', 'blue']
        for i in range(3):
            ax.axvline(quantiles[i], color=colors[i])

        # Set the title for each plot based on the cancer type
        ax.set_title(f"Cancer type: {cancer_type}")

    # Set the overall title for the figure based on the biomarker
    fig.suptitle(f"Biomarker {biomarker_index}: {biomarker} ", size = 15)
    
    # Display the plot
    plt.show()
    

def identify_outliers_mad(values, threshold_factor=100):
    """
    Identify outlier categories in the given list of Q2 values using the
    Median Absolute Deviation (MAD) approach.

    Parameters
    ----------
    Q2_values : list
        List of Q2 values from the descriptive statistics calculation
    threshold_factor : int, optional
        Factor by which the MAD is multiplied to define the threshold for
        outlier detection (default is 100)

    Returns
    -------
    list
        List of tuples, where each tuple contains the index of the outlier
        category in the input list and the corresponding Q2 value
    """

    # Calculate overall median value
    overall_median = np.median(values)

    # Calculate MAD (Median Absolute Deviation)
    mad = np.median([abs(x - overall_median) for x in values])

    # Define threshold for outlier detection
    threshold = overall_median + threshold_factor * mad

    # Identify and return indices of outlier categories
    outlier_with_indices = [(i,x) for i, x in enumerate(values) if x > threshold]
    return outlier_with_indices


def quantiles_across_categories(categories, dfs, biomarker_index, quantile_cut=0.75):
    """
    Calculate the quantiles for the given biomarker across all cancer types.

    Parameters
    ----------
    categories : list
        List of cancer types
    dfs : list
        List of DataFrames, each containing the features and labels for a particular cancer type
    biomarker_index : int
        Index of the biomarker of interest in the feature DataFrames
    quantile_cut : float, optional
        Quantile to calculate (default is 0.75)

    Returns
    -------
    list
        List of quantile values, one for each cancer type
    """
    Q_levels = []
    for df in dfs:
        features_df = feature_label_split(df)[0]
        biomarker = features_df.columns[biomarker_index]
        # Calculate Q2 (median) for the biomarker of interest in the current cancer type
        Q_value = df[biomarker].quantile(quantile_cut)
        Q_levels.append(float(Q_value))  # Append the median to the list
    return Q_levels

def demo_outlier_identification_mad(categories, dfs, biomarker_index):
    """
    Demonstrate outlier identification using the Median Absolute Deviation (MAD)
    approach on Q2 and Q3 values across different cancer types.

    Parameters
    ----------
    categories : list
        List of cancer types
    dfs : list
        List of DataFrames, each containing the features and labels for a particular cancer type
    biomarker_index : int
        Index of the biomarker of interest in the feature DataFrames

    Returns
    -------
    None
    """
    biomarker = feature_label_split(dfs[0])[0].columns[biomarker_index]
    print(f"Biomarker {biomarker_index}: {biomarker}")
    
    Q2_levels = quantiles_across_categories(categories, dfs, biomarker_index, quantile_cut=0.5)
    Q2_levels_with_categories = [(categories[index], Q2_value) for index, Q2_value in enumerate(Q2_levels)]
    print(f"Q2 values across cancer types: {Q2_levels_with_categories}")
    Q2_outliers_with_indices = identify_outliers_mad(Q2_levels)
    Q2_outliers_with_categories = [(categories[index], Q2_value) for index, Q2_value in Q2_outliers_with_indices]
    print(f"Outlier categories with uniquely high Q2 levels: {Q2_outliers_with_categories}")

    Q3_levels = quantiles_across_categories(categories, dfs, biomarker_index, quantile_cut=0.75)
    Q3_levels_with_categories = [(categories[index], Q3_value) for index, Q3_value in enumerate(Q3_levels)]
    print(f"Q3 values across cancer types: {Q3_levels_with_categories}")
    Q3_outliers_with_indices = identify_outliers_mad(Q3_levels)
    Q3_outliers_with_categories = [(categories[index], Q3_value) for index, Q3_value in Q3_outliers_with_indices]
    print(f"Outlier categories with uniquely high Q3 levels: {Q3_outliers_with_categories}")

# Debug code
if __name__ == "__main__":
    categories, dfs = load_data()
    # Biomarkers: AFP: 0, CA-125: 3, CA19-9: 5, Prolactin: 29, sHER2/sEGFR2/sErbB2: 33

    # descriptive_statistics(categories, dfs, biomarker_index = 0)
    for i in [0, 3, 5, 29, 33]:
        demo_outlier_identification_mad(categories, dfs, biomarker_index = i)
        print("\n")