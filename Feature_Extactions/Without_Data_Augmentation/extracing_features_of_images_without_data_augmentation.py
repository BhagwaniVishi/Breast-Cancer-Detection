import pandas as pd

# Load the data from the Excel file
file_path = 'C:\Users\dell\Desktop\Manit_research_intern\Feature_Extractions_Data\Vgg16\40x\vgg16_40x_adenosis.csv'
data = pd.read_excel(file_path)

# Determine the number of rows to skip (9 rows per image)
rows_per_image = 10  # 9 augmentations + 1 original row

# Extract rows where index is a multiple of `rows_per_image`
original_features = data.iloc[::rows_per_image]

# Save the extracted rows to a new Excel file
output_file_path = 'C:\Users\dell\Desktop\Manit_research_intern\Feature_Extractions_Data\Without_Augmentation/original_features.xlsx'
original_features.to_excel(output_file_path, index=False)

print(f"Original features saved to {output_file_path}")
