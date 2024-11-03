# Library imports
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from IPython.display import display

# Project imports
from data_preprocessing import load_data, feature_label_split

def ywtest(cancer_1_features, cancer_2_features, biomarker_index):
    
    # cancer_1_features and cancer_2_features are arrays or dataframes containing the levels of 39 biomarkers for each sample
    # We'll select the specific biomarker levels for cancer_1 and cancer_2
    cancer_1_biomarker = cancer_1_features.iloc[:, biomarker_index]  # Select biomarker levels from cancer 1 samples
    cancer_2_biomarker = cancer_2_features.iloc[:, biomarker_index]  # Select biomarker levels from cancer 2 samples

    # Perform the two-sample t-test
    t_stat, p_value = ttest_ind(cancer_1_biomarker, cancer_2_biomarker, equal_var=False, trim=0.1)  # Use Welch's t-test if variances differ
    return p_value

    # Output the results
    # print(f"T-statistic: {t_stat}")
    # print(f"P-value: {p_value}")

    # Interpret the p-value
    # alpha = 0.05  # Significance level
    # if p_value < alpha:
    #     print("The difference in biomarker levels is statistically significant.")
    # else:
    #     print("No statistically significant difference in biomarker levels.")

def utest(cancer_1_features, cancer_2_features, biomarker_index):
    # cancer_1_features and cancer_2_features are arrays or dataframes containing the levels of 39 biomarkers for each sample
    # We'll select the specific biomarker levels for cancer_1 and cancer_2
    cancer_1_biomarker = cancer_1_features.iloc[:, biomarker_index]  # Select biomarker levels from cancer 1 samples
    cancer_2_biomarker = cancer_2_features.iloc[:, biomarker_index]  # Select biomarker levels from cancer 2 samples

    # Perform the two-sample t-test
    u_stat, p_value = mannwhitneyu(cancer_1_biomarker, cancer_2_biomarker, alternative='two-sided')  # Use Welch's t-test if variances differ
    return p_value

def full_ywtest(cancer_category_index, biomarker_index, categories, dfs, test_type = 'ywtest', p_threshold = 0.05):
    test_dict = {'ywtest': ywtest, 'utest': utest}
    testfunction = test_dict[test_type]
    
    biomarker_levels = feature_label_split(dfs[0])[0]
    biomarkers = biomarker_levels.columns
    biomarker = biomarkers[biomarker_index]
    
    p_df = pd.DataFrame(index = [categories[cancer_category_index]])
    p_df.name = biomarker
    
    main_cancer_features = feature_label_split(dfs[cancer_category_index])[0]
    for i in range(9):
        if i != cancer_category_index:
            other_type_features = feature_label_split(dfs[i])[0]
            p_value = testfunction(main_cancer_features, other_type_features, biomarker_index = biomarker_index)
            p_df[categories[i]] = [p_value]
    
    # Find columns where the value in the first row is greater than 0.05
    categories_where_p_greater_than_threshold = list(p_df.columns[p_df.iloc[0] > p_threshold])

    # Get the column indices (locations)
    # categories_locations_where_p_greater_than_threshold = [p_df.columns.get_loc(col) for col in categories_where_p_greater_than_threshold]

    # has_p_values_greater_than_threshold = (p_df.loc[categories[cancer_category_index]] > p_threshold).any()        
    return categories_where_p_greater_than_threshold, p_df


def find_shared_nature_of_biomarkers(categories, dfs, cancer_category_index, cancer_selected_biomarkers, p_threshold = 0.05, debug = False):
    
    biomarker_levels = feature_label_split(dfs[0])[0]
    biomarkers = biomarker_levels.columns
    
    cancer_shared_nature_of_biomarkers = []
    for i in cancer_selected_biomarkers:
        categories_locations_where_p_greater_than_threshold, p_df = full_ywtest(cancer_category_index = cancer_category_index,
                                                                                biomarker_index = i,
                                                                                categories = categories,
                                                                                dfs = dfs,
                                                                                p_threshold = p_threshold)
        if len(categories_locations_where_p_greater_than_threshold) < 3:
            cancer_shared_nature_of_biomarkers.append((i, categories_locations_where_p_greater_than_threshold))
        
        if debug == True:
            if len(categories_locations_where_p_greater_than_threshold) == 0:
                print(f"\nBiomarker {i}: {biomarkers[i]}")
            elif len(categories_locations_where_p_greater_than_threshold) in [1, 2]:
                print(f"\nBiomarker {i}: {biomarkers[i]} didn't pass the p-value cutoff for one or two categories.")
            else:
                print(f"\nBiomarker {i}: {biomarkers[i]} didn't pass the p-value cutoff for three or more categories. It is dropped.")
            display(p_df)
        
    return cancer_shared_nature_of_biomarkers


if __name__ == "__main__":
    categories, dfs = load_data()
    cancer_1_features, _ = feature_label_split(dfs[6])
    cancer_2_features, _ = feature_label_split(dfs[0])
    # print(ttest(cancer_1_features, cancer_2_features, biomarker_index = 3))
    # categories_locations_where_p_greater_than_0_05, p_df = full_ywtest(cancer_category_index = 6, biomarker_index = 3, categories = categories, dfs = dfs)
    # print(categories_locations_where_p_greater_than_0_05)
    # print(p_df.name)
    # print(p_df)
    ovary_shared_nature_of_biomarkers = find_shared_nature_of_biomarkers(categories, 
                                                                         dfs, 
                                                                         cancer_category_index = 6, 
                                                                         cancer_selected_biomarkers = [3, 16, 29], 
                                                                         p_threshold = 0.05,
                                                                         debug = True)
    print(f"\nShared nature:\n{ovary_shared_nature_of_biomarkers}")