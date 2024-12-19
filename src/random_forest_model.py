# Library imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Permutation importance calculation
from sklearn.inspection import permutation_importance

# Project imports
from data_preprocessing import load_data, feature_label_split

def rf_normal_cancers(categories, 
                      dfs, 
                      cancer1_category_index, 
                      cancer2_category_index=None, 
                      cancer3_category_index=None, 
                      selected_biomarkers = np.arange(39),
                      test_size = 0.2,
                      iterations = 100, 
                      threshold = 0.05,
                      debug = True):
    # Initialize variables for resampling
    feature_importance_list = []  # To store feature importance scores
    accuracies = []  # To store accuracies

    # Cancer1 samples (resampled in each iteration)
    cancer_1_df = dfs[cancer1_category_index]  # Cancer1 dataset

    # Cancer2 samples (resampled in each iteration)
    if cancer2_category_index is not None:
        cancer_2_df = dfs[cancer2_category_index]  # Cancer2 dataset
        
    # Cancer3 samples (resampled in each iteration)
    if cancer3_category_index is not None:
        cancer_3_df = dfs[cancer3_category_index]  # Cancer3 dataset

    # Normal samples (resampled in each iteration)
    normal_df = dfs[5]  # Normal dataset

    # Minimum sample size for normal and each of the cancer datasets
    if cancer2_category_index is not None:
        if cancer3_category_index is not None:
            sample_size = np.min([len(cancer_1_df), len(cancer_2_df), len(cancer_3_df)])
        else:
            sample_size = np.min([len(cancer_1_df), len(cancer_2_df)])
    else:
        sample_size = len(cancer_1_df)

    # Loop through the specified number of iterations
    for i in range(iterations):
        # Step 1: Randomly sample from Normal dataset to match the minimum sample size
        normal_subsampled_df = normal_df.sample(n=sample_size, random_state=i)
        normal_biomarkers, normal_labels = feature_label_split(normal_subsampled_df, selected_biomarkers = selected_biomarkers)

        # Step 2: Randomly sample from Cancer1 dataset to match the minimum sample size
        cancer_1_subsampled_df = cancer_1_df.sample(n=sample_size, random_state=i)
        cancer_1_biomarkers, cancer_1_labels = feature_label_split(cancer_1_subsampled_df, selected_biomarkers = selected_biomarkers)

        # Step 3: Randomly sample from Cancer2 dataset (if present) to match the minimum sample size
        if cancer2_category_index is not None:
            cancer_2_subsampled_df = cancer_2_df.sample(n=sample_size, random_state=i)
            cancer_2_biomarkers, cancer_2_labels = feature_label_split(cancer_2_subsampled_df, selected_biomarkers = selected_biomarkers)
            
        # Step 4: Randomly sample from Cancer3 dataset (if present) to match the minimum sample size
        if cancer3_category_index is not None:
            cancer_3_subsampled_df = cancer_3_df.sample(n=sample_size, random_state=i)
            cancer_3_biomarkers, cancer_3_labels = feature_label_split(cancer_3_subsampled_df, selected_biomarkers = selected_biomarkers)

        # Step 4: Combine Normal, Cancer1, Cancer2 (if present) and  Cancer3 (if present) samples
        if cancer2_category_index is not None:
            if cancer3_category_index is not None:
                X = pd.concat([normal_biomarkers, cancer_1_biomarkers, cancer_2_biomarkers, cancer_3_biomarkers], ignore_index=True)
                y = pd.concat([normal_labels, cancer_1_labels, cancer_2_labels, cancer_3_labels], ignore_index=True)
            else:
                X = pd.concat([normal_biomarkers, cancer_1_biomarkers, cancer_2_biomarkers], ignore_index=True)
                y = pd.concat([normal_labels, cancer_1_labels, cancer_2_labels], ignore_index=True)
        else:
            X = pd.concat([normal_biomarkers, cancer_1_biomarkers], ignore_index=True)
            y = pd.concat([normal_labels, cancer_1_labels], ignore_index=True)

        # Step 5: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = i)

        # Step 6: Train RandomForestClassifier
        rf_normal_ovary_pancreas = RandomForestClassifier(random_state=i)
        rf_normal_ovary_pancreas.fit(X_train, y_train)

        # Step 7: Make predictions on the test set
        y_pred = rf_normal_ovary_pancreas.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)  # Store accuracy for this iteration

        # Step 8: Get feature importance scores and store them
        # Permutation importance
        # result = permutation_importance(rf_normal_ovary_pancreas, X_test, y_test, n_repeats=10, random_state=i, n_jobs=-1)
        # importance = result.importances_mean
        # MDI importance
        importance = rf_normal_ovary_pancreas.feature_importances_
        feature_importance_list.append(importance)
        
    # Step 9: Average feature importance scores across all iterations
    average_importance = np.mean(feature_importance_list, axis=0)

    # Step 10: Rank biomarkers by average importance
    feature_importance_df = pd.DataFrame({'Biomarker': X.columns, 'Importance': average_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Step 11: Filter biomarkers with average importance >= 0.05
    important_biomarkers = feature_importance_df[feature_importance_df['Importance'] >= threshold]

    # Step 12: Print results
    if debug:
        if cancer2_category_index is not None:
            if cancer3_category_index is not None:
                print(f"Random forest classification: {categories[5]} + {categories[cancer1_category_index]} + {categories[cancer2_category_index]} + {categories[cancer3_category_index]}")
            else:
                print(f"Random forest classification: {categories[5]} + {categories[cancer1_category_index]} + {categories[cancer2_category_index]}")
        else:
            print(f"Random forest classification: {categories[5]} + {categories[cancer1_category_index]}")
        print(f"\nAverage Accuracy over {iterations} iterations: {np.mean(accuracies):.4f}")
        print(f"\nBiomarkers with Importance >= {threshold}:")
        print(important_biomarkers)

    return important_biomarkers




def plot_important_biomarkers(important_biomarkers, 
                              datasets = ['Normal', 'Ovary', 'Pancreas'], 
                              ax = None, 
                              color = 'goldenrod',
                              threshold = 0.05):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Biomarker', data=important_biomarkers, color = color, ax=ax)
    if len(datasets) == 3:
        titletext = f"{datasets[0]} + {datasets[1]} + {datasets[2]} \n Biomarkers with Average Importance >= {threshold}"
    elif len(datasets) == 2:
        titletext = f"{datasets[0]} + {datasets[1]} \nBiomarkers with Average Importance >= {threshold}"
    ax.set_title(titletext)
    return ax


# if __name__ == "__main__":
#     fig = plt.figure(figsize=(10, 8))
#     grid = plt.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

#     ax1 = plt.subplot(grid[:, 0])
#     ax2 = plt.subplot(grid[0, 1])
#     ax3 = plt.subplot(grid[1, 1])

#     plot_important_biomarkers(important_biomarkers = important_biomarkers_normal_ovary_pancreas,
#                               datasets = ['Normal', 'Ovary', 'Pancreas'],
#                               threshold = 0.05,
#                               color = 'goldenrod',
#                               ax = ax1)

#     plot_important_biomarkers(important_biomarkers = important_biomarkers_normal_ovary,
#                               datasets = ['Normal', 'Ovary'],
#                               threshold = 0.05,
#                               color = 'mediumturquoise',
#                               ax = ax2)

#     plot_important_biomarkers(important_biomarkers = important_biomarkers_normal_pancreas,
#                               datasets = ['Normal', 'Pancreas'],
#                               threshold = 0.05,
#                               color = 'mediumslateblue',
#                               ax = ax3)

#     fig.suptitle("Important Biomarkers in Random Forest Classification", fontsize=14)
#     fig.tight_layout(pad=2.0)
#     plt.show()