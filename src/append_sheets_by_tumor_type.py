import pandas as pd

def append_sheets_by_tumor_type(file_path = "data/prelim_clinical_cancer_data.xlsx"):
    df = pd.read_excel(file_path, sheet_name=0)  # Assumes first sheet has all data

    # List of tumor types to split into separate sheets
    tumor_types = [
        "Breast", "Colorectum", "Esophagus", "Liver", "Lung",
        "Normal", "Ovary", "Pancreas", "Stomach"
    ]

    # Create a new Excel file with multiple sheets
    with pd.ExcelWriter("data/clinical_cancer_data.xlsx", engine="openpyxl") as writer:
        # Write the full data as the first sheet
        df.to_excel(writer, sheet_name="All", index=False)
        
        # Write each tumor-specific subset into its own sheet
        for tumor in tumor_types:
            filtered_df = df[df["Tumor type"] == tumor]
            filtered_df.to_excel(writer, sheet_name=tumor, index=False)
    
    print("Data appended by tumor type to data/clinical_cancer_data.xlsx")
            
if __name__ == "__main__":
    append_sheets_by_tumor_type()
