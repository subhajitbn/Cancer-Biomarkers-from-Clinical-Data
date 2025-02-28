# Load required libraries
library(readxl)     # For reading Excel files
library(factoextra)
library(ggrepel)

# Step 1: Load the Excel file
file_path <- "clinical_cancer_data.xlsx"  # Replace with your file path
sheet_names <- excel_sheets(file_path)  # Get sheet names

# Verify if 'Normal' sheet exists
if (!"Normal" %in% sheet_names) {
  stop("Error: 'Normal' sheet is missing from the Excel file.")
}

normal_data <- read_excel(file_path, sheet = "Normal")
cancer_sheets <- sheet_names[sheet_names != "Normal"]
selected_cancer_sheets <- c("Liver", "Ovary", "Pancreas")

# Define a mapping of wrong (or long) feature names to correct (or short) feature names.
feature_name_mapping <- c(
  "sHER2.sEGFR2.sErbB2" = "sHER2",
  "CA.125" = "CA-125",
  "Kallikrein.6" = "Kallikrein-6",
  "TIMP.1" = "TIMP-1",
  "TIMP.2" = "TIMP-2",
  "sPECAM.1" = "sPECAM-1",
  "CYFRA.21.1" = "CYFRA 21-1",
  "Thrombospondin.2" = "Thrombospondin-2"
)

# Step 2: Iterate through the selected cancer sheets and
for (cancer in selected_cancer_sheets) {
  
  # Load cancer data
  cancer_data <- read_excel(file_path, sheet = cancer)
  
  # Ensure equal number of samples from normal and cancer (use the smaller sample size)
  n_samples <- min(nrow(normal_data), nrow(cancer_data))
  
  # if (n_samples < nrow(normal_data) || n_samples < nrow(cancer_data)) {
  #   message(sprintf("Sample sizes truncated to %d for group parity in '%s'.", n_samples, cancer))
  # }
  
  # seed 123 also produces good plots
  set.seed(12)  # Set a seed for reproducibility
  normal_sample <- normal_data[sample(1:nrow(normal_data), n_samples), ]
  cancer_sample <- cancer_data[sample(1:nrow(cancer_data), n_samples), ]
  
  # Standardize column names
  colnames(normal_sample) <- make.names(colnames(normal_sample))
  colnames(cancer_sample) <- make.names(colnames(cancer_sample))

  # Keep only common columns (if column names are mismatched)
  common_columns <- intersect(colnames(normal_sample), colnames(cancer_sample))
  normal_sample <- normal_sample[, common_columns]
  cancer_sample <- cancer_sample[, common_columns]
  
  # Combine data into a single dataframe
  combined_data <- rbind(normal_sample, cancer_sample)
  group_labels <- c(rep("Normal", n_samples), rep(cancer, n_samples))
  
  # Exclude the 1,2,4, and the last two columns
  numeric_data_with_label <- combined_data[, -(c(1, 2, 4, ncol(combined_data) - 1, ncol(combined_data)))]
  # Replace "Normal" with "Healthy" in the Tumor.type column of numeric_data_with_label
  numeric_data_with_label$Tumor.type <- gsub("Normal", "Healthy", numeric_data_with_label$Tumor.type)
  # Exclude the "Tumor type" column
  numeric_data <- numeric_data_with_label[, -1]
  
  # # Ensure column alignment is still intact
  # print("Final Numeric Data Column Names:")
  # print(colnames(numeric_data))
  
  # Replace biomarker names with the mapped names in the column names of numeric data
  colnames(numeric_data) <- ifelse(
    colnames(numeric_data) %in% names(feature_name_mapping),
    feature_name_mapping[colnames(numeric_data)],
    colnames(numeric_data)  # Keep the name unchanged if no mapping is found
  )
  
  # Perform PCA
  pca_result <- prcomp(numeric_data, scale. = TRUE)
  
  p1 <- fviz_pca_biplot(pca_result,
                        select.var = list(contrib = 10),
                        geom = "point",
                        addEllipses = TRUE, 
                        ggtheme = theme_gray(), 
                        alpha.var=0.3,
                        col.ind = numeric_data_with_label$Tumor.type,
                        col.var = "blue", 
                        repel=TRUE,
                        title = paste0(cancer, " + Healthy samples")) + theme(
                          plot.title = element_text(size = 16),         
                          axis.title = element_text(size = 14),        
                          axis.text = element_text(size = 12),         
                          legend.title = element_text(size = 13),      
                          legend.text = element_text(size = 12)        
                        )
  
  # Display or save the plot
  print(p1)  # Show in the RStudio viewer
  
  # Uncomment to save the plot as an image file
  file_name <- paste0(cancer, "_PCA_biplot_top_biomarkers.pdf")
  ggsave(filename = file_name, plot = p1, dpi = 600, width = 16, height = 12)
}

