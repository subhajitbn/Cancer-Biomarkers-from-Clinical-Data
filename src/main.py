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

# Add the path to the src folder
# import sys
# sys.path.append('src')

# Import project functions
from data_preprocessing import load_data, feature_label_split
from random_forest_model import rf_normal_cancers, plot_important_biomarkers
from desc_stats import descriptive_statistics, cancer_biomarkers_uniquely_high, cancer_biomarkers_higher_side_filtering
from stats_tests import find_shared_nature_of_biomarkers

# %% [markdown]
# ## 1.2. Load the data

# %%
categories, dfs = load_data()

# list(enumerate(categories))

# %%
biomarkers = feature_label_split(dfs[0])[0].columns
# list(enumerate(biomarkers))

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
                                                      debug=False)

# %% [markdown]
# And here's the list of the biomarker indices.

# %%
ovary_important_biomarker_indices_in_RF = list(important_biomarkers_normal_ovary.index)
# ovary_important_biomarker_indices_in_RF

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

# %%
# [biomarkers[i] for i in ovary_biomarkers_uniquely_high]

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

# %%
# [biomarkers[i] for i in ovary_biomarkers_higher_side]

# %% [markdown]
# Let's collect the biomarker indices that pass through the two descriptive statistics-based filtering criteria.
# 

# %%
ovary_selected_biomarkers = ovary_biomarkers_uniquely_high + ovary_biomarkers_higher_side

# [biomarkers[i] for i in ovary_selected_biomarkers]

# %% [markdown]
# For example, let's see an overview of `CA-125` levels in all the cancer types, as well as the normal samples.

# %%
# descriptive_statistics(categories = categories, dfs = dfs, biomarker_index = 3)

# %% [markdown]
# ### 2.1.3. Yuen-Welch's test of `CA-125`, `Prolactin` and `HE4` levels in `Ovary` samples versus all the other cancer types

# %%
ovary_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
    categories = categories,
    dfs = dfs,
    cancer_category_index = 6,
    cancer_selected_biomarkers = ovary_selected_biomarkers,
    p_threshold = 0.05,
    debug = False
)

# %% [markdown]
# Now, we have the finalized list of biomarkers. Note that, `HE4` didn't pass the hypothesis test based criterion.

# %%
# [(biomarkers[i], shared) for i, shared in ovary_shared_nature_of_biomarkers]

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
                                                         debug=False)

# %%
pancreas_important_biomarker_indices_in_RF = list(important_biomarkers_normal_pancreas.index)
# pancreas_important_biomarker_indices_in_RF

# %% [markdown]
# ### 2.2.2. Filtering through descriptive statistics of biomarkers in `Normal + Pancreas` samples

# %%
pancreas_biomarkers_uniquely_high = cancer_biomarkers_uniquely_high(categories = categories,
                                                                    dfs = dfs,
                                                                    cancer_important_biomarker_indices_in_RF = pancreas_important_biomarker_indices_in_RF)

# %%
# pancreas_biomarkers_uniquely_high

# %%
candidates_for_higher_side_filtering = pancreas_important_biomarker_indices_in_RF.copy()
candidates_for_higher_side_filtering.remove(pancreas_biomarkers_uniquely_high[0])

pancreas_biomarkers_higher_side = cancer_biomarkers_higher_side_filtering(categories = categories,
                                                                          dfs = dfs,
                                                                          cancer_category_index = 7,
                                                                          cancer_candidates_for_higher_side_filtering = candidates_for_higher_side_filtering)

# %%
# pancreas_biomarkers_higher_side

# %%
pancreas_selected_biomarkers = pancreas_biomarkers_uniquely_high + pancreas_biomarkers_higher_side

# [biomarkers[i] for i in pancreas_selected_biomarkers]

# %%
# descriptive_statistics(categories = categories, dfs = dfs, biomarker_index = 5)

# %% [markdown]
# ### 2.2.3. Yuen-Welch's test of `CA19-9` and `SHER2/sEGFR2/sErbB2` levels in `Pancreas` samples versus all the other cancer types

# %%
pancreas_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
    categories = categories,
    dfs = dfs,
    cancer_category_index = 7,
    cancer_selected_biomarkers = pancreas_selected_biomarkers,
    p_threshold = 0.05,
    debug = False
)

# %% [markdown]
# `IL-8` is immediately dropped. Because the p_values are greater than 0.05 for 3 comparisons.

# %%
# [(biomarkers[i], shared) for i, shared in pancreas_shared_nature_of_biomarkers]

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
                                                      debug=False)

# %%
liver_important_biomarker_indices_in_RF = list(important_biomarkers_normal_liver.index)
# liver_important_biomarker_indices_in_RF

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

# [biomarkers[i] for i in liver_selected_biomarkers]

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
    debug = False
)

# %% [markdown]
# Note that `GDF15` and `IL-6` are dropped.

# %%
# liver_shared_nature_of_biomarkers

# %% [markdown]
# ## 2.4. RandomForest accuracy scores for classifying liver, ovarian, and pancreatic cancers from normal ones  

# %% [markdown]
# ### 2.4.1. Normal + Liver with `AFP`, `OPN`, `Myeloperoxidase` and `HGF`

# %%
liver_finalized_biomarkers = [i for i, _ in liver_shared_nature_of_biomarkers]
rf_normal_cancers(categories = categories, 
                  dfs = dfs,
                  cancer1_category_index = 3,
                  selected_biomarkers = np.array(liver_finalized_biomarkers),
                  test_size = 0.2,
                  iterations = 100,
                  threshold = 0.01,
                  debug=False)

# %% [markdown]
# ### 2.4.2. Normal + Ovary with `CA-125`, `Prolactin` and `HE4`

# %%
ovary_finalized_biomarkers = [i for i, _ in ovary_shared_nature_of_biomarkers]
rf_normal_cancers(categories = categories, 
                  dfs = dfs,
                  cancer1_category_index = 6,
                  selected_biomarkers = np.array(ovary_finalized_biomarkers),
                  test_size = 0.2,
                  iterations = 100,
                  threshold = 0.01,
                  debug=False)

# %% [markdown]
# ### 2.4.3. Normal + Pancreas with `CA19-9`, `sHER2/sEGFR2/sErbB2`, `Midkine` and `GDF15`

# %%
pancreas_finalized_biomarkers = [i for i, _ in pancreas_shared_nature_of_biomarkers]
rf_normal_cancers(categories = categories, 
                  dfs = dfs,
                  cancer1_category_index = 7,
                  selected_biomarkers = np.array(pancreas_finalized_biomarkers),
                  test_size = 0.2,
                  iterations = 100,
                  threshold = 0.05,
                  debug=False)

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
                  test_size = 0.2,
                  iterations = 100,
                  threshold = 0.05,
                  debug=False)

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
                                                       debug=False)

# %% [markdown]
# And here's list of biomarkers that were selected by random forest classifier for `Normal + Breast` samples.

# %%
breast_important_biomarker_indices_in_RF = list(important_biomarkers_normal_breast.index)
# breast_important_biomarker_indices_in_RF

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
                                                           debug=False)

# %%
colorectum_important_biomarker_indices_in_RF = list(important_biomarkers_normal_colorectum.index)
# colorectum_important_biomarker_indices_in_RF

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
                                                          debug=False)

# %%
esophagus_important_biomarker_indices_in_RF = list(important_biomarkers_normal_esophagus.index)
# esophagus_important_biomarker_indices_in_RF

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

# [biomarkers[i] for i in esophagus_selected_biomarkers]

# %% [markdown]
# ### 4.1.3. Yuen-Welch's test of filtered biomarkers' levels in `Esophagus` samples versus all the other cancer types

# %%
esophagus_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(
    categories = categories,
    dfs = dfs,
    cancer_category_index = 2,
    cancer_selected_biomarkers = esophagus_selected_biomarkers,
    p_threshold = 0.05,
    debug = False
)

# %%
# esophagus_shared_nature_of_biomarkers

# %%
# [(biomarkers[i], shared) for i, shared in esophagus_shared_nature_of_biomarkers]

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
                                                     debug=False)

# %%
lung_important_biomarker_indices_in_RF = list(important_biomarkers_normal_lung.index)
# lung_important_biomarker_indices_in_RF

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

# [biomarkers[i] for i in lung_selected_biomarkers]

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

# %%
# [(biomarkers[i], shared) for i, shared in lung_shared_nature_of_biomarkers]

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
                                                        debug=False)

# %%
stomach_important_biomarker_indices_in_RF = list(important_biomarkers_normal_stomach.index)
# stomach_important_biomarker_indices_in_RF

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

# [biomarkers[i] for i in stomach_selected_biomarkers]

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

# %%
# stomach_shared_nature_of_biomarkers

# %%
# [biomarkers[i] for i, _ in stomach_shared_nature_of_biomarkers]

# %% [markdown]
# # 5. Summary of findings

# %% [markdown]
# Now, let's have a look at all the cancer types and their selected biomarkers, with shared nature. We make a few summarizing comments, looking at all the cancer types and their selected biomarkers, with shared nature.

# %% [markdown]
# ## 5.1. `Ovary` biomarkers 
# We make a comment that `HE4` might be shared with `Pancreas`.

# %%
ovary_biomarkers = [(biomarkers[i],  shared) for i, shared in ovary_shared_nature_of_biomarkers]

# %% [markdown]
# ## 5.2. `Pancreas` biomarkers 
# We make a comment that `GDF15` might be shared with `Liver`.

# %%
pancreas_biomarkers = [(biomarkers[i], shared) for i, shared in pancreas_shared_nature_of_biomarkers]

# %% [markdown]
# ## 5.3. `Liver` biomarkers
# Note that `HGF`, `OPN` and `Myloperoxidase` are hereby confirmed as shared between `Liver`, `Esophagus` and `Stomach`.

# %%
liver_biomarkers = [(biomarkers[i], shared) for i, shared in liver_shared_nature_of_biomarkers]

# %% [markdown]
# ## 5.4. `Esophagus` biomarkers
# Note the sharing of `OPN`, `HGF` and `Myloperoxidase` with `Liver` and `Stomach`.

# %%
esophagus_biomarkers = [(biomarkers[i], shared) for i, shared in esophagus_shared_nature_of_biomarkers]

# %% [markdown]
# ## 5.5. `Stomach` biomarkers
# Note the sharing of `OPN`, `HGF` and `Myloperoxidase` with `Liver` and `Esophagus`.

# %%
stomach_biomarkers = [(biomarkers[i], shared) for i, shared in stomach_shared_nature_of_biomarkers]

# %% [markdown]
# ## 5.6. `Lung` biomarkers
# Since `Prolactin` is reported in the higher side filtering of `Ovary` with no shared nature, and it's levels are the highest in `Ovary`, we do not consider it a reliable biomarker for `Lung` cancer.

# %%
lung_biomarkers = [(biomarkers[i], shared) for i, shared in lung_shared_nature_of_biomarkers]


# Debug code
collection = {'ovary': ovary_biomarkers, 'pancreas': pancreas_biomarkers, 'liver': liver_biomarkers, 'esophagus': esophagus_biomarkers, 'stomach': stomach_biomarkers, 'lung': lung_biomarkers}

for cancer_type, biomarkers in collection.items():
    print(f"{cancer_type} biomarkers: {biomarkers}")