import pandas as pd
import re

def load_data_and_extract_Table_S6(file_path, 
                                   sheet_name = "Table S6",
                                   rows_to_trim_from_above = 2,
                                   rows_to_trim_from_bottom = 4):
    
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=rows_to_trim_from_above)
    return df.iloc[:-rows_to_trim_from_bottom].reset_index(drop=True)

def strip_units(col):
    # Remove anything in parentheses and extra spaces
    return re.sub(r"\s*\(.*?\)", "", col).strip()

def strip_stars(df):
    # Remove '*' from all string values in the DataFrame
    return df.map(lambda x: x.replace('*', '') if isinstance(x, str) else x)

def extract_blood_test_table(file_path = "data/aar3247_cohen_sm_tables-s1-s11.xlsx"):
    df = load_data_and_extract_Table_S6(file_path)
    df.columns = [strip_units(col) for col in df.columns]
    df = strip_stars(df)
    # Save to new Excel file
    df.to_excel("data/prelim_clinical_cancer_data.xlsx", index=False)
    print("Table S6 extracted to data/prelim_clinical_cancer_data.xlsx")

if __name__ == "__main__":
    extract_blood_test_table()