# Library imports
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
    




# Debug code
if __name__ == "__main__":
    categories, dfs = load_data()
    biomarker_index = 0
    descriptive_statistics(categories, dfs, biomarker_index)