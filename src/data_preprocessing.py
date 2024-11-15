# We require the following packages:
# openpyxl
import pandas as pd
import numpy as np

def load_data(file_path = "../data/clinical_cancer_data.xlsx"):
    
    """
    Load the clinical cancer data from the given Excel file.
    
    Parameters
    ----------
    file_path : str, default "../data/clinical_cancer_data.xlsx"
        The path to the Excel file containing the clinical cancer data.
    
    Returns
    -------
    tuple
        A tuple containing the names of the datasheets, followed by the dataframes
        corresponding to the individual datasheets.
    """
    
    # Load the excel file
    xls = pd.ExcelFile(file_path)

    # The datasheet names. Note that the individual datasheets start from sheet 2, i.e., index 1.
    categories = xls.sheet_names[1:]

    # Load the individual datasheets into a list of dataframes
    dfs = [pd.read_excel(xls, sheet_name) for sheet_name in xls.sheet_names[1:]]
    
    return categories, dfs


def feature_label_split(df: pd.DataFrame, selected_biomarkers = np.arange(39)) -> tuple:
    """
    Split the dataframe into biomarker levels and cancer label.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    
    Returns
    -------
    tuple
        A tuple containing the biomarker levels and the cancer label.
    
    Notes
    -----
    The dataframe is assumed to have the following structure:
    - The first column is the sample id.
    - The second column is the Tumor type.
    - The third column is the AJCC Stages.
    - The columns from 4 to 43 are the biomarker levels.
    """
    selected_features = selected_biomarkers + 4
    biomarker_levels = df.iloc[:, selected_features]
    cancer_label = df.iloc[:, 2]
    return biomarker_levels, cancer_label

def feature_label_split_stage_I(df: pd.DataFrame, selected_biomarkers = np.arange(39)) -> tuple:
    """
    Split the dataframe into biomarker levels and cancer label for stage I data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    
    Returns
    -------
    tuple
        A tuple containing the biomarker levels and the cancer label for stage I data.
    """
    # Filter the dataframe to only include stage I data
    stage_I_df = df[df.iloc[:, 3] == 'I']
    return feature_label_split(stage_I_df, selected_biomarkers)


# Debug code
if __name__ == "__main__":
    categories, dfs = load_data()
    cancer_category_index = 3
    print(f"Tumor type: {categories[cancer_category_index]}")
    print(f"Unique valuse of stages: {dfs[cancer_category_index].iloc[:, 3].unique()}")
    print(f"Number of records: {dfs[cancer_category_index].shape[0]}")
    print(f"Number of stage I records: {feature_label_split_stage_I(dfs[cancer_category_index])[0].shape[0]}")