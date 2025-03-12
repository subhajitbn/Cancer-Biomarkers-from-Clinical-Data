# Load required libraries
library(readxl)     # For reading Excel files
library(ggplot2)
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
  
  # Percentage of variance explained for each PC
  variance_explained <- summary(pca_result)$importance[2, ] * 100  # The second row stores Proportion of Variance
  
  # Get specific values for PC1 and PC2
  percentage_pc1 <- round(variance_explained[1], 2)  # Variance explained by PC1
  percentage_pc2 <- round(variance_explained[2], 2)  # Variance explained by PC2
  
  # Extract PCA results
  var_coords <- as.data.frame(pca_result$rotation)  # Extract variable loadings
  var_coords$Variable <- rownames(var_coords)  # Add feature names as a separate column
  rownames(var_coords) <- NULL  # Remove rownames for clean data manipulation
  
  # Compute "contributions" of variables as the sum of squared loadings for PC1 and PC2
  var_coords$Contribution <- var_coords$PC1^2 + var_coords$PC2^2
  
  # Select the top 10 contributing variables
  top_var_coords <- var_coords[order(-var_coords$Contribution), ][1:10, ]  # Sort and pick top 10
  
  # Extract sample (individual) coordinates for PC1 and PC2
  ind_coords <- as.data.frame(pca_result$x)  # Coordinates of individuals (samples)
  ind_coords <- ind_coords[, c("PC1", "PC2")]  # Retain only the first two PCs for plotting
  ind_coords$Group <- numeric_data_with_label$Tumor.type  # Add group labels for coloring
  
  # Create PCA biplot using ggplot2 and ggrepel
  arrow_scaling_factor = 30
  
  p1 <- ggplot(ind_coords, aes(x = PC1, y = PC2, color = Group)) +
    geom_point(size = 2, alpha = 0.6) +  # Plot individuals
    stat_ellipse(geom = "polygon", aes(fill = Group), 
                 alpha = 0.2, level = 0.95, show.legend = FALSE) + 
    geom_segment(data = top_var_coords, 
                 aes(x = 0, y = 0, 
                     xend = PC1 * arrow_scaling_factor, yend = PC2 * arrow_scaling_factor),
                 arrow = arrow(length = unit(0.3, "cm")), 
                 color = "blue", size = 0.5, alpha = 0.3) +  # Plot variable arrows
    geom_text_repel(data = top_var_coords, 
                    aes(x = PC1 * arrow_scaling_factor, 
                        y = PC2 * arrow_scaling_factor, label = Variable),
                    size = 6, color = "blue",
                    max.overlaps = 20,
                    box.padding = 0.5,
                    point.padding = 0.5,
                    force = 2) +  # Add ggrepel for label placement
    # Flip colors for the groups
    scale_color_manual(values = rev(scales::hue_pal()(2))) +  # Reverse the default color palette
    scale_fill_manual(values = rev(scales::hue_pal()(2))) +   # Reverse the default fill palette
    labs(# title = paste0(cancer, " + Healthy Samples"), 
         x = paste0("PC1 (", percentage_pc1, "%)"), 
         y = paste0("PC2 (", percentage_pc2, "%)"), 
         color = "Group") +
    theme_gray() +
    theme(
      plot.title = element_text(size = 18),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 12),
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14)
    )
  
  # Display or save the plot
  print(p1)  # View in the RStudio viewer
  
  # Uncomment to save the plot as an image file
  file_name <- paste0(cancer, "_PCA_biplot_top_biomarkers.pdf")
  ggsave(filename = file_name, plot = p1, dpi = 600, width = 8, height = 6)
}
