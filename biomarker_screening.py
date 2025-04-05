# %% [markdown]
# # Tutorial for Cancer-Biomarkers-from-Clinical-Data

# %% [markdown]
# # 1. Data loading and preprocessing

# %% [markdown]
# ## 1.1. Import the necessary modules

# %%
# Library imports
import numpy as np
import pandas as pd
import warnings
# Add the path to the src folder
import sys
sys.path.append('src')

# Import the functions necessary for extracting the blood test data from the table by Cohen et al.
from extract_blood_test_table import extract_blood_test_table
from append_sheets_by_tumor_type import append_sheets_by_tumor_type

# Import project functions
from data_preprocessing import load_data, feature_label_split
from random_forest_model import rf_normal_cancers, plot_important_biomarkers
from desc_stats import descriptive_statistics, cancer_biomarkers_uniquely_high, cancer_biomarkers_higher_side_filtering
from stats_tests import find_shared_nature_of_biomarkers

# Import visualization libraries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# %% [markdown]
# ## 1.2. Load the data

def extract_and_clean_data(file_path = "data/aar3247_cohen_sm_tables-s1-s11.xlsx"):
    extract_blood_test_table(file_path)
    append_sheets_by_tumor_type()


def biomarker_discovery_ROC_AUC_and_other_visualizations(file_path = "data/clinical_cancer_data.xlsx"):
    # %%
    categories, dfs = load_data('data/clinical_cancer_data.xlsx')

    # %%
    biomarkers = feature_label_split(dfs[0])[0].columns

    # %% [markdown]
    # # 2. Analysis of `Ovary`, `Pancreas` and `Liver` samples, taken together with random subsamples of `Normal` samples
    # 
    # In this section, we will see three cancer types for which the list of biomarkers given by random forest classifier contain biomarkers with uniquely high level in the particular cancer type, along with other biomarkers whose Q3 values are in the top 2 among all cancer types. Essentially, in each of these cancer types, we obtain practically viable biomarkers. 

    # %% [markdown]
    # ## 2.1. Analysis of `Normal + Ovary` samples

    # %% [markdown]
    # ### 2.1.1. Random forest classification for `Normal + Ovary` samples

    # %%
    important_biomarkers_normal_ovary = rf_normal_cancers(categories = categories, 
                                                        dfs = dfs, 
                                                        cancer1_category_index = 6, 
                                                        iterations = 100,
                                                        threshold = 0.04,
                                                        save_feature_importances_list = True)

    # %% [markdown]
    # And here's the list of the biomarker indices.

    # %%
    ovary_important_biomarker_indices_in_RF = list(important_biomarkers_normal_ovary.index)

    # %% [markdown]
    # ### 2.1.2. Filtering through descriptive statistics of biomarkers in `Normal + Ovary` samples

    # %% [markdown]
    # Now, we find that CA-125 is present in uniquely high levels in `Ovary` samples.

    # %%
    ovary_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories, 
                                                                    dfs = dfs, 
                                                                    cancer_important_biomarker_indices_in_RF = ovary_important_biomarker_indices_in_RF)

    # %% [markdown]
    # Let's see the biomarkers that pass through the uniquely high level criterion.

    # %% [markdown]
    # And, `Prolactin` and `HE4` pass through the higher side filtering criteria.

    # %%
    ovary_candidates_for_higher_side_filtering = ovary_important_biomarker_indices_in_RF.copy()
    ovary_candidates_for_higher_side_filtering.remove(ovary_biomarkers_uniquely_high[0])

    ovary_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 6,
                                                                        cancer_candidates_for_higher_side_filtering = ovary_candidates_for_higher_side_filtering)

    # %% [markdown]
    # Here's the list of biomarkers that pass through the higher side filtering criteria.

    # %% [markdown]
    # Let's collect the biomarker indices that pass through the two descriptive statistics-based filtering criteria.
    # 

    # %%
    ovary_selected_biomarkers = ovary_biomarkers_uniquely_high + ovary_biomarkers_higher_side

    # %% [markdown]
    # For example, let's see an overview of `CA-125` levels in all the cancer types, as well as the normal samples.

    # %% [markdown]
    # ### 2.1.3. Yuen-Welch's test of `CA-125`, `Prolactin` and `HE4` levels in `Ovary` samples versus all the other cancer types

    # %%
    ovary_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 6,
        cancer_selected_biomarkers = ovary_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # Now, we have the finalized list of biomarkers. Note that, `HE4` didn't pass the hypothesis test based criterion.

    # %% [markdown]
    # ## 2.2. Analysis of `Normal + Pancreas` samples

    # %% [markdown]
    # ### 2.2.1. Random forest classification for `Normal + Pancreas` samples

    # %%
    important_biomarkers_normal_pancreas = rf_normal_cancers(categories = categories, 
                                                            dfs = dfs, 
                                                            cancer1_category_index = 7, 
                                                            iterations = 100, 
                                                            threshold = 0.04,
                                                            save_feature_importances_list = True)

    # %%
    pancreas_important_biomarker_indices_in_RF = list(important_biomarkers_normal_pancreas.index)

    # %% [markdown]
    # ### 2.2.2. Filtering through descriptive statistics of biomarkers in `Normal + Pancreas` samples

    # %%
    pancreas_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_important_biomarker_indices_in_RF = pancreas_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = pancreas_important_biomarker_indices_in_RF.copy()
    candidates_for_higher_side_filtering.remove(pancreas_biomarkers_uniquely_high[0])

    pancreas_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                            dfs = dfs,
                                                                            cancer_category_index = 7,
                                                                            cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %%
    pancreas_selected_biomarkers = pancreas_biomarkers_uniquely_high + pancreas_biomarkers_higher_side

    # %% [markdown]
    # ### 2.2.3. Yuen-Welch's test of `CA19-9` and `SHER2/sEGFR2/sErbB2` levels in `Pancreas` samples versus all the other cancer types

    # %%
    pancreas_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 7,
        cancer_selected_biomarkers = pancreas_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # `IL-8` is immediately dropped. Because the p_values are greater than 0.05 for 3 comparisons.

    # %% [markdown]
    # ## 2.3. Analysis of `Normal + Liver` samples

    # %% [markdown]
    # ### 2.3.1. Random forest classification for `Normal + Liver` samples

    # %%
    important_biomarkers_normal_liver = rf_normal_cancers(categories = categories, 
                                                        dfs = dfs,
                                                        cancer1_category_index = 3,
                                                        iterations = 100,
                                                        threshold = 0.04,
                                                        save_feature_importances_list = True)

    # %%
    liver_important_biomarker_indices_in_RF = list(important_biomarkers_normal_liver.index)

    # %% [markdown]
    # ### 2.3.2. Filtering through descriptive statistics of biomarkers in `Normal + Liver` samples

    # %%
    liver_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = liver_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = liver_important_biomarker_indices_in_RF.copy()
    candidates_for_higher_side_filtering.remove(liver_biomarkers_uniquely_high[0])

    liver_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 3,
                                                                        cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %%
    liver_selected_biomarkers = liver_biomarkers_uniquely_high + liver_biomarkers_higher_side

    # %%
    # descriptive_statistics(categories = categories, dfs = dfs, biomarker_index = 0)

    # %% [markdown]
    # ### 2.3.3 Yuen-Welch's test of `AFP`, `OPN`, `Myeloperoxidase`, `HGF` and `GDF15` levels in `Liver` samples versus all the other cancer types

    # %%
    liver_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 3,
        cancer_selected_biomarkers = liver_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # Note that `GDF15` and `IL-6` are dropped.

    # %% [markdown]
    # ## 2.4. RandomForest accuracy scores for classifying liver, ovarian, and pancreatic cancers from normal ones  
    # 
    # In this section, we use a switch `roc = True` for generating the ROC curves with AUC scores. This switch also makes sure that the train-test split is stratified. Hence, we will see that the classification accuracy changes a little bit from the previous one. We report the present classification accuracy in the article.

    # %% [markdown]
    # ### 2.4.1. Normal + Liver with `AFP`, `OPN`, `Myeloperoxidase` and `HGF`

    # %%
    liver_finalized_biomarkers = [i for i, _ in liver_shared_nature_of_biomarkers]
    rf_normal_cancers(categories = categories, 
                    dfs = dfs,
                    cancer1_category_index = 3,
                    selected_biomarkers = np.array(liver_finalized_biomarkers),
                    test_size = 0.4,
                    iterations = 100,
                    threshold = 0.01,
                    debug = True,
                    roc = True)

    # %% [markdown]
    # ### 2.4.2. Normal + Ovary with `CA-125`, `Prolactin` and `HE4`

    # %%
    ovary_finalized_biomarkers = [i for i, _ in ovary_shared_nature_of_biomarkers]
    rf_normal_cancers(categories = categories, 
                    dfs = dfs,
                    cancer1_category_index = 6,
                    selected_biomarkers = np.array(ovary_finalized_biomarkers),
                    test_size = 0.4,
                    iterations = 100,
                    threshold = 0.01,
                    debug = True,
                    roc = True)

    # %% [markdown]
    # ### 2.4.3. Normal + Pancreas with `CA19-9`, `sHER2/sEGFR2/sErbB2`, `Midkine` and `GDF15`

    # %%
    pancreas_finalized_biomarkers = [i for i, _ in pancreas_shared_nature_of_biomarkers]
    rf_normal_cancers(categories = categories, 
                    dfs = dfs,
                    cancer1_category_index = 7,
                    selected_biomarkers = np.array(pancreas_finalized_biomarkers),
                    test_size = 0.4,
                    iterations = 100,
                    threshold = 0.05,
                    debug = True,
                    roc = True)

    # %% [markdown]
    # ### 2.4.4. Normal + Liver + Ovary + Pancreas

    # %%
    liver_ovary_pancreas_finalized_biomarkers = liver_finalized_biomarkers + ovary_finalized_biomarkers + pancreas_finalized_biomarkers
    rf_normal_cancers(categories = categories, 
                    dfs = dfs,
                    cancer1_category_index = 3,
                    cancer2_category_index = 6,
                    cancer3_category_index = 7,
                    selected_biomarkers = np.array(liver_ovary_pancreas_finalized_biomarkers),
                    test_size = 0.4,
                    iterations = 100,
                    threshold = 0.05)

    # %% [markdown]
    # # 3. Analysis of `Breast` and `Colorectum` samples, taken together with random subsamples of `Normal` samples
    # 
    # Now we see two cancer types for which the important biomarkers given by random forest classifier are not suitable in practical scenario for distinguishing between different cancer types from normal samples. None of the biomarkers display uniquely high level for the particular cancer type, and none can be found with Q3 level in top 2 among all the cancer types. 

    # %% [markdown]
    # ## 3.1. Analysis of `Normal + Breast` samples

    # %% [markdown]
    # ### 3.1.1. Random forest classification for `Normal + Breast` samples

    # %%
    important_biomarkers_normal_breast = rf_normal_cancers(categories = categories, 
                                                        dfs = dfs,
                                                        cancer1_category_index = 0,
                                                        iterations = 100,
                                                        threshold = 0.04,
                                                        save_feature_importances_list = True)

    # %% [markdown]
    # And here's list of biomarkers that were selected by random forest classifier for `Normal + Breast` samples.

    # %%
    breast_important_biomarker_indices_in_RF = list(important_biomarkers_normal_breast.index)

    # %% [markdown]
    # ### 3.1.2. Filtering through descriptive statistics of biomarkers in `Normal + Breast` samples

    # %% [markdown]
    # Now, we see that none of the biomarkers selected by Random Forest is present in uniquely high levels in the `Breast` samples.

    # %%
    breast_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = breast_important_biomarker_indices_in_RF)

    # %% [markdown]
    # And none satisfy the higher side filtering criterion.

    # %%
    candidates_for_higher_side_filtering = breast_important_biomarker_indices_in_RF.copy()
    if len(breast_biomarkers_uniquely_high) != 0:
        candidates_for_higher_side_filtering.remove(breast_biomarkers_uniquely_high[0])

    breast_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 0,
                                                                        cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %% [markdown]
    # So, we've got no biomarker left at the end of the filtering process.

    # %% [markdown]
    # ## 3.2. Analysis of `Normal + Colorectum` samples

    # %% [markdown]
    # ### 3.2.1. Random forest classification for `Normal + Colorectum` samples

    # %%
    important_biomarkers_normal_colorectum = rf_normal_cancers(categories = categories, 
                                                            dfs = dfs,
                                                            cancer1_category_index = 1,
                                                            iterations = 100,
                                                            threshold = 0.04,
                                                            save_feature_importances_list = True)

    # %%
    colorectum_important_biomarker_indices_in_RF = list(important_biomarkers_normal_colorectum.index)

    # %% [markdown]
    # ### 3.2.2. Filtering through descriptive statistics of biomarkers in `Normal + Colorectum` samples
    # 
    # * Uniquely high levels: None
    # * Higher side filtering: None
    # 
    # Note that, `IL-8` and `IL-6`, being important in regulating immune response and inflammation, are not specific to any one type of cancer. For example, the same two biomarkers are two of the most important ones in separating `Normal` and `Breast` samples, as shown above. And they can be found in higher levels in some other cancer types, such as `Esophagus`, `Liver` and `Lung` samples.
    # 
    # `OPN` Q3 levels are much higher in `Esophagus`, `Liver`, and `Stomach` samples than in `Colorectum` samples.
    # 
    # `HGF` Q3 levels are higher in `Pancreas` samples, and much higher in `Esophagus`, `Liver`, and `Stomach` samples than in `Colorectum` samples.
    # 
    # `GDF15` Q3 levels are close in `Colorectum` and Ovary samples, and higher in `Esophagus`, `Liver`, `Stomach` and `Pancreas` samples than in `Colorectum` samples.
    # 
    # `sFas` Q3 levels are higher in `Breast`, `Esophagus`, `Liver`, `Lung`, `Pancreas`, `Stomach` and even `Normal` samples than in `Colorectum` samples.
    # 
    # `Prolactin` levels are much higher in `Liver`, `Lung` and `Ovary` samples than in `Colorectum` samples.
    # 
    # 

    # %%
    colorectum_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = colorectum_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = colorectum_important_biomarker_indices_in_RF.copy()
    if len(colorectum_biomarkers_uniquely_high) != 0:
        candidates_for_higher_side_filtering.remove(colorectum_biomarkers_uniquely_high[0])

    colorectum_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                                dfs = dfs,
                                                                                cancer_category_index = 1,
                                                                                cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %% [markdown]
    # # 4. Analysis of `Esophagus`, `Lung` and `Stomach` samples, taken together with random subsamples of `Normal` samples

    # %% [markdown]
    # ## 4.1. Analysis of `Normal + Esophagus` samples

    # %% [markdown]
    # ### 4.1.1. Random forest classification for `Normal + Esophagus` samples

    # %%
    important_biomarkers_normal_esophagus = rf_normal_cancers(categories = categories, 
                                                            dfs = dfs,
                                                            cancer1_category_index = 2,
                                                            iterations = 100,
                                                            threshold = 0.04,
                                                            save_feature_importances_list = True)

    # %%
    esophagus_important_biomarker_indices_in_RF = list(important_biomarkers_normal_esophagus.index)

    # %% [markdown]
    # ### 4.1.2.  Filtering through descriptive statistics of biomarkers in `Normal + Esophagus` samples

    # %%
    esophagus_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = esophagus_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = esophagus_important_biomarker_indices_in_RF.copy()
    if len(esophagus_biomarkers_uniquely_high) != 0:
        candidates_for_higher_side_filtering.remove(esophagus_biomarkers_uniquely_high[0])

    esophagus_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 2,
                                                                        cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %%
    esophagus_selected_biomarkers = esophagus_biomarkers_uniquely_high + esophagus_biomarkers_higher_side

    # %% [markdown]
    # ### 4.1.3. Yuen-Welch's test of filtered biomarkers' levels in `Esophagus` samples versus all the other cancer types

    # %%
    esophagus_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 2,
        cancer_selected_biomarkers = esophagus_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # ## 4.2. Analysis of `Normal + Lung` samples

    # %% [markdown]
    # ### 4.2.1. Random forest classification for `Normal + Lung` samples

    # %%
    important_biomarkers_normal_lung = rf_normal_cancers(categories = categories, 
                                                        dfs = dfs,
                                                        cancer1_category_index = 4,
                                                        iterations = 100,
                                                        threshold = 0.04,
                                                        save_feature_importances_list = True)

    # %%
    lung_important_biomarker_indices_in_RF = list(important_biomarkers_normal_lung.index)

    # %% [markdown]
    # ### 4.2.2. Filtering through descriptive statistics of biomarkers in `Normal + Lung` samples

    # %%
    lung_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = lung_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = lung_important_biomarker_indices_in_RF.copy()
    if len(lung_biomarkers_uniquely_high) != 0:
        candidates_for_higher_side_filtering.remove(lung_biomarkers_uniquely_high[0])

    lung_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 4,
                                                                        cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %%
    lung_selected_biomarkers = lung_biomarkers_uniquely_high + lung_biomarkers_higher_side

    # %% [markdown]
    # ### 4.2.3. Yuen-Welch's test of filtered biomarkers' levels in `Lung` samples versus all the other cancer types

    # %%
    lung_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 4,
        cancer_selected_biomarkers = lung_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # ## 4.3. Analysis of `Normal + Stomach` samples

    # %% [markdown]
    # ### 4.3.1. Random forest classification for `Normal + Stomach` samples

    # %%
    important_biomarkers_normal_stomach = rf_normal_cancers(categories = categories, 
                                                            dfs = dfs,
                                                            cancer1_category_index = 8,
                                                            iterations = 100,
                                                            threshold = 0.04,
                                                            save_feature_importances_list = True)

    # %%
    stomach_important_biomarker_indices_in_RF = list(important_biomarkers_normal_stomach.index)

    # %% [markdown]
    # ### 4.3.2. Filtering through descriptive statistics of biomarkers in `Normal + Stomach` samples

    # %%
    stomach_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = stomach_important_biomarker_indices_in_RF)

    # %%
    candidates_for_higher_side_filtering = stomach_important_biomarker_indices_in_RF.copy()
    if len(stomach_biomarkers_uniquely_high) != 0:
        candidates_for_higher_side_filtering.remove(stomach_biomarkers_uniquely_high[0])

    stomach_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                        dfs = dfs,
                                                                        cancer_category_index = 8,
                                                                        cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

    # %%
    stomach_selected_biomarkers = stomach_biomarkers_uniquely_high + stomach_biomarkers_higher_side

    # %% [markdown]
    # ### 4.3.3. Yuen-Welch's test of filtered biomarkers' levels in `Stomach` samples versus all the other cancer types

    # %%
    stomach_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
        categories = categories,
        dfs = dfs,
        cancer_category_index = 8,
        cancer_selected_biomarkers = stomach_selected_biomarkers,
        p_threshold = 0.05,
        debug = True
    )

    # %% [markdown]
    # # 5. Summary of findings

    # %% [markdown]
    # Now, let's have a look at all the cancer types and their selected biomarkers, with shared nature. We make a few summarizing comments, looking at all the cancer types and their selected biomarkers, with shared nature.

    # %%
    print("\n\nSUMMARY OF FINDINGS:\n")

    # %% [markdown]
    # ## 5.1. `Ovary` biomarkers 
    # We make a comment that `HE4` might be shared with `Pancreas`.

    # %%
    print("\nBiomarkers selected for ovary:\n")
    print([(biomarkers[i],  shared) for i, shared in ovary_shared_nature_of_biomarkers])

    # %% [markdown]
    # ## 5.2. `Pancreas` biomarkers 
    # We make a comment that `GDF15` might be shared with `Liver`.

    # %%
    print("\nBiomarkers selected for pancreas:\n")
    print([(biomarkers[i], shared) for i, shared in pancreas_shared_nature_of_biomarkers])

    # %% [markdown]
    # ## 5.3. `Liver` biomarkers
    # Note that `HGF`, `OPN` and `Myloperoxidase` are hereby confirmed as shared between `Liver`, `Esophagus` and `Stomach`.

    # %%
    print("\nBiomarkers selected for liver:\n")
    print([(biomarkers[i], shared) for i, shared in liver_shared_nature_of_biomarkers])

    # %% [markdown]
    # ## 5.4. `Esophagus` biomarkers
    # Note the sharing of `OPN`, `HGF` and `Myloperoxidase` with `Liver` and `Stomach`.

    # %%
    print("\nBiomarkers selected for esophagus:\n")
    print([(biomarkers[i], shared) for i, shared in esophagus_shared_nature_of_biomarkers])

    # %% [markdown]
    # ## 5.5. `Stomach` biomarkers
    # Note the sharing of `OPN`, `HGF` and `Myloperoxidase` with `Liver` and `Esophagus`.

    # %%
    print("\nBiomarkers selected for stomach:\n")
    print([(biomarkers[i], shared) for i, shared in stomach_shared_nature_of_biomarkers])

    # %% [markdown]
    # ## 5.6. `Lung` biomarkers
    # Since `Prolactin` is reported in the higher side filtering of `Ovary` with no shared nature, and it's levels are the highest in `Ovary`, we do not consider it a reliable biomarker for `Lung` cancer.

    # %%
    print("\nBiomarkers selected for lung:\n")
    print([(biomarkers[i], shared) for i, shared in lung_shared_nature_of_biomarkers])
    
    
    # VISUALIZATIONS
    

    cbf_colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9', '#999999']
    importance_scores = [important_biomarkers_normal_breast, important_biomarkers_normal_colorectum, important_biomarkers_normal_esophagus, important_biomarkers_normal_liver, important_biomarkers_normal_lung, important_biomarkers_normal_ovary, important_biomarkers_normal_pancreas, important_biomarkers_normal_stomach]

    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharey=True, constrained_layout=True)
    axs.flatten()
    for i, ax in enumerate(axs.flatten()):
        sns.barplot(data=importance_scores[i], x="Biomarker", y="Importance", ax=ax, color=cbf_colors[i])
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_xlabel('')
        ax.set_ylabel('Importance', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        if i<5:
            # ax.set_title(f'Normal + {categories[i]}')
            ax.set_title(f'{categories[i]}', fontsize=18)
        else:
            # ax.set_title(f'Normal + {categories[i+1]}')
            ax.set_title(f'{categories[i+1]}', fontsize=18)

    # fig.suptitle(f"Important biomarkers in Random Forest classification based on MDI scores (cutoff 0.04)\n", fontsize=14)

    fig.savefig("FIG2.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    # %% [markdown]
    # ## 6.2. Descriptive statistics based filtering

    # %% [markdown]
    # ### 6.2.1. Uniquely high levels

    # %%
    data_rows = []
    rf_important_biomarkers = pd.concat(importance_scores, axis=0, ignore_index=True).iloc[:, 0].unique()
    for biomarker in rf_important_biomarkers:
        for i, cancer_type in enumerate(categories):
            df = dfs[i]
            desc_stats = df[biomarker].describe()
            mean = desc_stats['mean']
            std = desc_stats['std']
            cv = mean / std
            Q2 = desc_stats['50%']
            Q3 = desc_stats['75%']
            data_row_Q2 = [biomarker, Q2, 'Q2', cancer_type]
            data_row_Q3 = [biomarker, Q3, 'Q3', cancer_type]
            data_rows.append(data_row_Q2)
            data_rows.append(data_row_Q3)
            
    columns = ['Biomarker', 'Level', 'Quartile', 'Tumor_type']
    cv_boxplot_df = pd.DataFrame(data_rows, columns=columns).replace('sHER2/sEGFR2/sErbB2', 'sHER2').replace('CA-125', 'CA-125 ★').replace('CA19-9', 'CA19-9 ★').replace('AFP', 'AFP ★')

    # facet_kws = dict(col_wrap=3)
    # sns.catplot(data=cv_boxplot_df, x='Quartile', y='Level', hue='Tumor_type', col='Biomarker', aspect=0.5, col_wrap=3)

    biomarkers_0_to_200000_range = ['Prolactin', 'OPN', 'Myeloperoxidase', 'NSE', 'TIMP-1']
    biomarkers_0_to_10000_range = ['CYFRA 21-1', 'HE4', 'sEGFR', 'HGF', 'GDF15', 'sFas', 'sHER2', 'Midkine']
    biomarkers_TGFa = ['TGFa']
    biomarkers_0_to_200_range = ['IL-8', 'IL-6']
    biomarkers_AFP = ['AFP ★']
    biomarker_sets_together = biomarkers_0_to_200000_range + biomarkers_0_to_10000_range + biomarkers_TGFa + biomarkers_0_to_200_range + biomarkers_AFP
    cv_boxplot_df_1 = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(biomarkers_0_to_200000_range)]
    cv_boxplot_df_2 = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(biomarkers_0_to_10000_range)]
    cv_boxplot_df_TGFa = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(biomarkers_TGFa)]
    cv_boxplot_df_added = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(biomarkers_0_to_200_range)]
    cv_boxplot_df_3 = cv_boxplot_df[~cv_boxplot_df['Biomarker'].isin(biomarker_sets_together)]
    cv_boxplot_df_4 = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(biomarkers_AFP)]

    # %%
    boxplot_dfs = [cv_boxplot_df_1, cv_boxplot_df_TGFa, cv_boxplot_df_added, cv_boxplot_df_3, cv_boxplot_df_2, cv_boxplot_df_4]

    # Create figure with flexible layout
    fig = plt.figure(figsize=(12, 12), constrained_layout = True)
    gs = GridSpec(3, 20, figure=fig)
    text_font_size = 12

    # Define subplots (6:4 width ratio for the first row, 8:2 width ratio for the second row)
    ax1 = fig.add_subplot(gs[0, :13])  # First row, first subplot (occupying 6 columns)
    ax2 = fig.add_subplot(gs[0, 14:])
    ax3 = fig.add_subplot(gs[1, :8])
    ax4 = fig.add_subplot(gs[1, 10:])  # First row, second subplot (occupying 4 columns)
    ax5 = fig.add_subplot(gs[2, :13])  # Second row, first subplot (occupying 8 columns)
    ax6 = fig.add_subplot(gs[2, 15:])  # Second row, second subplot (occupying 2 columns)

    ax2.set_ylim(bottom=15, top=52)
    ax4.set_ylim(top=550)
    ax5.set_ylim(top=17500)
    ax6.set_ylim(top=650000)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    for i, ax in enumerate(axs):
        # sns.stripplot(data=boxplot_dfs[i], x = 'Biomarker', y = 'Level',  hue = 'Quartile', dodge=True, jitter=False, ax=ax)
        sns.boxplot(data=boxplot_dfs[i], x = 'Biomarker', y = 'Level',  hue = 'Quartile', ax=ax)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=12)
        # ax.spines['top'].set_visible(False)
        # Add outlier annotations
        for biomarker in boxplot_dfs[i]['Biomarker'].unique():
            biomarker_data = boxplot_dfs[i][boxplot_dfs[i]['Biomarker'] == biomarker]
            
            for quartile in biomarker_data['Quartile'].unique():
                data = biomarker_data[biomarker_data['Quartile'] == quartile]
                levels = data['Level']
                tumor_types = data['Tumor_type']
                
                q1 = levels.quantile(0.25)
                q3 = levels.quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[levels > upper_bound]
                for _, outlier_row in outliers.iterrows():
                    outlier_level = outlier_row['Level']
                    outlier_quartile = outlier_row['Quartile']
                    tumor_type = outlier_row['Tumor_type']
                    if outlier_quartile == 'Q2':
                        if biomarker == 'TGFa':
                            if tumor_type == 'Esophagus':
                                ax.annotate(f"{tumor_type}", 
                                            xy=(biomarker, outlier_level), 
                                            xytext=(-95, 5), 
                                            textcoords="offset points", 
                                            fontsize=text_font_size, 
                                            color='blue')
                            else:
                                ax.annotate(f"{tumor_type}", 
                                            xy=(biomarker, outlier_level), 
                                            xytext=(-60, 14), 
                                            textcoords="offset points", 
                                            fontsize=text_font_size, 
                                            color='blue')
                        
                        elif biomarker == 'AFP ★':
                            if tumor_type == 'Liver':
                                ax.annotate(f"{tumor_type}", 
                                            xy=(biomarker, outlier_level), 
                                            xytext=(-55, 14), 
                                            textcoords="offset points", 
                                            fontsize=text_font_size, 
                                            color='blue')
                            else:
                                ax.annotate(f"{tumor_type}", 
                                            xy=(biomarker, outlier_level), 
                                            xytext=(-80, 6), 
                                            textcoords="offset points", 
                                            fontsize=text_font_size, 
                                            color='blue')
                        elif biomarker == 'IL-6':
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(-70, 5), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                        elif biomarker == 'CA-125 ★':
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(-55, 4), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                        elif biomarker == 'NSE':
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(-52, 6), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                        else:
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(-35, 6), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                    else:
                        if biomarker == 'Midkine':
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(-28, 5), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                        elif biomarker == 'AFP ★':
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(10, 5), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')
                        else:
                            ax.annotate(f"{tumor_type}", 
                                        xy=(biomarker, outlier_level), 
                                        xytext=(1, 6), 
                                        textcoords="offset points", 
                                        fontsize=text_font_size, 
                                        color='blue')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.04))
    # Completely remove legends from all subplots
    for ax in axs:
        ax.get_legend().remove()  
        
    # Add explanatory text at the bottom to indicate what "★" means
    fig.text(0.5, 0.02, f'★ indicates biomarkers with \"uniquely high levels\" in the corresponding tumor type,\n as determined by the MAD-based outlier detection criteria', ha='center', va='center', fontsize=12, color='black')
    # fig.suptitle("Q2 and Q3 levels of the biomarkers across cancer types\n")

    fig.savefig("FIG3.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    # %% [markdown]
    # ### 6.2.2. Higher side filtering

    # %%
    q3_df = cv_boxplot_df[cv_boxplot_df['Quartile']=='Q3'].reset_index(drop=True)
    important_biomarkers = q3_df.Biomarker.unique()
    biomarker_q3_rank_list = []
    for biomarker in important_biomarkers:
        biomarker_q3_df = q3_df[q3_df['Biomarker'] == biomarker].sort_values(by='Level', ascending=False).reset_index(drop=True)
        row = [biomarker, (biomarker_q3_df.iloc[0].Tumor_type, biomarker_q3_df.iloc[1].Tumor_type, biomarker_q3_df.iloc[2].Tumor_type)]
        biomarker_q3_rank_list.append(row)
    biomarkers_q3_rank_df = pd.DataFrame(data=biomarker_q3_rank_list, columns=['Biomarker', 'Tumor_types_first_three_Q3']).set_index('Biomarker')

    biomarkers_q3_rank_df.head()

    # %%
    cancer_types = categories.copy()
    cancer_types.remove('Normal')

    higher_side_data_rows = []

    for i, tumor_type in enumerate(cancer_types):

        important_biomarkers_for_this_tumor_type = importance_scores[i].Biomarker.replace('sHER2/sEGFR2/sErbB2', 'sHER2').replace('CA-125', 'CA-125 ★').replace('CA19-9', 'CA19-9 ★').replace('AFP', 'AFP ★')

        for biomarker in important_biomarkers_for_this_tumor_type:
            # print(tumor_type, list(biomarkers_q3_rank_df.loc[biomarker])[0])
            ranked_tumor_types_for_this_biomarker = list(biomarkers_q3_rank_df.loc[biomarker])[0]
            if tumor_type in ranked_tumor_types_for_this_biomarker:
                q3_rank = ranked_tumor_types_for_this_biomarker.index(tumor_type) + 1
                row = [tumor_type, biomarker, q3_rank]
                higher_side_data_rows.append(row)

    columns = ['Tumor_type', 'Biomarker', 'Q3_rank']
    higher_side_data_rows_df = pd.DataFrame(data=higher_side_data_rows, columns=columns)
    higher_side_data_rows_df['Weight'] = 4-higher_side_data_rows_df['Q3_rank']


    # %%
    # Create your figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    axs = axs.flatten()

    # Get all unique tumor types
    tumor_types = higher_side_data_rows_df.Tumor_type.unique()

    # Get all unique Weight values and assign a color to each Weight for consistency
    unique_ranks = sorted(higher_side_data_rows_df.Q3_rank.unique())
    colors = plt.cm.Paired(np.linspace(0, 1, len(unique_ranks)))  # Generate colors
    weight_color_map = dict(zip(unique_ranks, colors))  # Map Weight values to colors
    rank_color_map = dict(zip(unique_ranks, colors[::-1]))

    for i, tumor_type in enumerate(tumor_types):
        tumor_type_filtered_df = higher_side_data_rows_df[
            higher_side_data_rows_df['Tumor_type'] == tumor_type
        ].reset_index(drop=True)
        ax = axs[i]
        
        # Extract weights and assign colors based on the weight->color map
        weights = tumor_type_filtered_df.Weight
        pie_colors = [weight_color_map[w] for w in weights]

        # Plot the pie chart
        ax.pie(weights, 
            labels=tumor_type_filtered_df.Biomarker, 
            colors=pie_colors, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
            textprops = {'fontsize':14})
        ax.set_title(tumor_type, fontsize=16, fontdict={'color':'indigo'})

    # Create a shared legend for unique Weight values
    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color=rank_color_map[r], linestyle='', markersize=10, label=f"Rank {r}"
        )
        for r in unique_ranks
    ]

    # Add the legend to the figure
    fig.legend(
        handles=legend_elements, 
        title="Q3 level rank",
        title_fontsize=18, 
        fontsize=16,
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=3
    )

    # fig.suptitle("Q3 ranks of the biomarkers (tumor types with the top 3 Q3 levels are considered)\n")

    fig.savefig("FIG4.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    # %% [markdown]
    # ### 6.2.3. Yuen-Welch's test

    # %%
    from stats_tests import heatmap_shared_nature

    # %%
    heatmap_shared_natures = []
    cancers_selected_biomarkers = [esophagus_selected_biomarkers, liver_selected_biomarkers, lung_selected_biomarkers, ovary_selected_biomarkers, pancreas_selected_biomarkers, stomach_selected_biomarkers]
    cancers_indices = [2, 3, 4, 6, 7, 8]
    for i in range(0,6):
        cancer_heatmap_shared_nature = heatmap_shared_nature(categories = categories,dfs = dfs,cancer_category_index = cancers_indices[i],cancer_selected_biomarkers = cancers_selected_biomarkers[i],p_threshold = 0.05)
        heatmap_shared_natures.append(cancer_heatmap_shared_nature)

    # %%
    # 4, 4, 1, 3, 4, 3
    # heatmap_shared_natures[5]

    # Define threshold and colormap bounds
    vmin = 0  # Minimum value
    vmax = 1  # Maximum value
    threshold = 0.05

    # Enhanced colormap to emphasize threshold
    cmap = sns.diverging_palette(250, 30, l=65, as_cmap=True)  # Custom continuous colormap

    # Create figure with flexible layout
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(10, 2, figure=fig)

    # Define subplots
    ax1 = fig.add_subplot(gs[:4, 0])  # First col, first subplot
    ax2 = fig.add_subplot(gs[4:8, 0])  # First col, second subplot
    ax3 = fig.add_subplot(gs[8:, 0])  # First col, third subplot
    ax4 = fig.add_subplot(gs[:3, 1])  # Second col, first subplot
    ax5 = fig.add_subplot(gs[3:7, 1])  # Second col, second subplot
    ax6 = fig.add_subplot(gs[7:, 1])  # Second col, third subplot

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Plot heatmaps
    for ax, df in zip(axs, heatmap_shared_natures):
        df.rename(index={'AFP':'AFP ★', 'CA19-9 ':'CA19-9 ★', 'CA-125':'CA-125 ★', 'sHER2/sEGFR2/sErbB2': 'sHER2 ', 'Myeloperoxidase':'MPO'}, inplace=True)
        df.rename(columns={'Normal':'Healthy'}, inplace=True)
        sns.heatmap(
            df, 
            # annot=True, # For including actual p-values in each cell 
            annot=False, # For not having actual p-values in each cell
            ax=ax, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            linewidths=0.1,
            cbar=False  # Disable individual colorbars
        )
        ax.set_title(df.name, fontsize=18, fontdict={'fontweight':'heavy', 'color':'indigo'})
        ax.tick_params(axis='x', labelsize=14, labelrotation=20)
        ax.tick_params(axis='y', labelsize=16, labelrotation=60)

    # Add shared colorbar
    cbar_ax = fig.add_axes([1.02, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Customize the colorbar to mark the threshold
    cbar.set_label("Shared Scale (Threshold: 0.05)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Annotate the color legend to emphasize regions below/above the threshold
    cbar.ax.hlines(y=threshold, xmin=0, xmax=1, colors="red", lw=2, label=f"Threshold \n {threshold}")
    cbar.ax.legend(loc="lower left", frameon=False, fontsize=16)

    # fig.suptitle("Shared nature of biomarkers indicated by p-values (threshold = 0.05)\n", fontsize=20)

    fig.savefig("FIG5.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    # %%
    important_biomarkers_normal_stomach.Biomarker.unique()

    # %%
    collection = []
    for sample_list in cancers_selected_biomarkers:
        collection = collection + sample_list
        

    # %%
    cancer_shared_nature_of_biomarkers_list = [esophagus_shared_nature_of_biomarkers, liver_shared_nature_of_biomarkers, lung_shared_nature_of_biomarkers, ovary_shared_nature_of_biomarkers, pancreas_shared_nature_of_biomarkers, stomach_shared_nature_of_biomarkers]
    reported_biomarkers = []
    for cancer_shared_nature_of_biomarkers in cancer_shared_nature_of_biomarkers_list: 
        reported_biomarkers = reported_biomarkers + [biomarkers[i] for i, _ in cancer_shared_nature_of_biomarkers]
        
    # %% [markdown]
    # ## 6.3. Heatmap of the selected Biomarkers

    # %%
    random_forest_biomarkers = list(cv_boxplot_df.Biomarker.unique())
    removed_biomarkers =['TGFa', 'IL-8', 'CYFRA 21-1', 'sFas', 'sEGFR', 'NSE', 'TIMP-1']
    finalized_biomarkers = [biomarker for biomarker in random_forest_biomarkers if biomarker not in removed_biomarkers]
    finalized_biomarkers

    # %%
    df = cv_boxplot_df[cv_boxplot_df['Biomarker'].isin(finalized_biomarkers)]

    # Pivot the data for Q2 and Q3 separately
    df_q2 = df[df["Quartile"] == "Q2"].pivot(index="Biomarker", columns="Tumor_type", values="Level")
    df_q3 = df[df["Quartile"] == "Q3"].pivot(index="Biomarker", columns="Tumor_type", values="Level")

    # Min-max normalization within each biomarker
    df_q2_norm = df_q2.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
    df_q3_norm = df_q3.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

    df_q2_norm.rename(columns={'Normal':'Healthy'}, inplace=True)
    df_q3_norm.rename(columns={'Normal':'Healthy'}, inplace=True)
    # # Set up the figure
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # # Create the heatmaps
    # sns.heatmap(df_q2_norm, cmap="coolwarm", annot=False, linewidths=0.5, ax=axes[0])
    # axes[0].set_title("Normalized Q2 (Median) Biomarker Levels")

    # sns.heatmap(df_q3_norm, cmap="coolwarm", annot=False, linewidths=0.5, ax=axes[1])
    # axes[1].set_title("Normalized Q3 (Third Quartile) Biomarker Levels")

    # Adjust layout and show the plot

    # Create the heatmaps
    ax1 = sns.heatmap(df_q2_norm,
                    cmap="coolwarm",
                    annot=False,
                    linewidths=0.5)
    plt.xlabel("Tumor type", fontsize=18)
    plt.ylabel("Biomarker", fontsize=18)
    plt.tick_params(axis='x', labelsize=14, labelrotation=90)
    plt.tick_params(axis='y', labelsize=14)
    # Customize the colorbar fontsize
    colorbar1 = ax1.collections[0].colorbar  # Access the colorbar
    colorbar1.ax.tick_params(labelsize=14)  # Set tick label size
    plt.tight_layout()
    plt.savefig("q2_heatmap.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    ax2 = sns.heatmap(df_q3_norm, 
                    cmap="coolwarm", 
                    annot=False, 
                    linewidths=0.5)
    plt.xlabel("Tumor type", fontsize=18)
    plt.ylabel("Biomarker", fontsize=18)
    plt.tick_params(axis='x', labelsize=14, labelrotation=90)
    plt.tick_params(axis='y', labelsize=14)
    # Customize the colorbar fontsize
    colorbar2 = ax2.collections[0].colorbar  # Access the colorbar
    colorbar2.ax.tick_params(labelsize=14)  # Set tick label size
    plt.tight_layout()
    plt.savefig("q3_heatmap.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()


    # %% [markdown]
    # # 7. Limitations

    # %%
    # Load the CSV file, skipping the first row
    df = pd.read_csv("feature_importance_list_Normal_Ovary.csv", skiprows=1)

    # Convert columns to numeric (in case they are read as strings)
    df = df.apply(pd.to_numeric)

    # Compute cumulative mean for each biomarker (column-wise)
    cumulative_mean = df.expanding().mean()

    # Plot convergence for a subset of biomarkers
    plt.figure(figsize=(10, 6))
    biomarker_indices = [3, 29, 18, 35, 31, 37, 16]
    # Plot only the specified biomarkers
    for index in biomarker_indices:
        col = df.columns[index]  # Get the column name corresponding to the index
        plt.plot(cumulative_mean.index, cumulative_mean[col], label=f"{biomarkers[index]}")


    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Cumulative Mean MDI", fontsize=18)
    # plt.title("Convergence of MDI Scores Over Iterations")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig("normal_ovary_convergence_of_MDI_scores.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

    # Compute change in mean between successive iterations
    delta_mean = cumulative_mean.diff().abs().mean(axis=1)

    # Plot the rate of change
    plt.figure(figsize=(10, 6))
    plt.plot(delta_mean, color="red", label="Mean Change Across Biomarkers")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Change in Mean", fontsize=18)
    # plt.title("Stability of Cumulative Mean MDI Scores")
    plt.axhline(y=0.001, color="gray", linestyle="--", label="Threshold (0.001)")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig("normal_ovary_stability_of_cumulative_mean_MDI_scores.pDF", dpi=600, bbox_inches='tight', format='pdf')
    plt.show()

def biomarker_screening(file_path = "data/aar3247_cohen_sm_tables-s1-s11.xlsx"):
    warnings.filterwarnings("ignore", category=UserWarning)
    extract_and_clean_data(file_path = file_path)
    biomarker_discovery_ROC_AUC_and_other_visualizations()
    
    
if __name__ == "__main__":
    biomarker_screening()