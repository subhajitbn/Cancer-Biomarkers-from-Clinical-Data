# We require the following packages:
# openpyxl
import pandas as pd

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