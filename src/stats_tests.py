# Library imports
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# Project imports
from data_preprocessing import load_data, feature_label_split

def ttest(cancer_1_features, cancer_2_features, biomarker_index):
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

def full_test(cancer_category_index, biomarker_index, categories, dfs, test_type = 'ttest'):
    test_dict = {'ttest': ttest, 'utest': utest}
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
    return p_df


if __name__ == "__main__":
    categories, dfs = load_data()
    cancer_1_features, _ = feature_label_split(dfs[6])
    cancer_2_features, _ = feature_label_split(dfs[0])
    # print(ttest(cancer_1_features, cancer_2_features, biomarker_index = 3))
    p_df = full_test(cancer_category_index = 6, biomarker_index = 0, categories = categories, dfs = dfs)
    print(p_df.name)
    print(p_df)