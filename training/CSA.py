import os
import nibabel as nib
import numpy as np
import pandas as pd
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Calculate cross-sectional areas from .nii segmentation masks.')
parser.add_argument('--input_directory', type=str, required=True, help='Path to the directory containing .nii files.')  # Updated to use --input_directory
parser.add_argument('--output_file', type=str, required=True, help='Name of the output CSV file.')

args = parser.parse_args()

# List to store results
results = []

# Iterate through each file in the input directory
for filename in os.listdir(args.input_directory):
    if filename.endswith('.nii.gz'):  # Check for .nii.gz files
        # Load the NIfTI file
        nii_image = nib.load(os.path.join(args.input_directory, filename))
        mask_data = nii_image.get_fdata()

        # Get voxel dimensions (in mm)
        voxel_size = nii_image.header.get_zooms()[:2]  # Get the first two dimensions (x, y)
        voxel_area = voxel_size[0] * voxel_size[1]  # Area in mm²

        # Count the number of pixels for label 1
        label_1_pixel_count = np.sum(mask_data == 1)

        # Calculate the cross-sectional area for label 1
        cross_sectional_area_mm2 = label_1_pixel_count * voxel_area

        # Convert to cm²
        cross_sectional_area_cm2 = cross_sectional_area_mm2 / 100  # 1 cm² = 100 mm²

        # Append results with image identifier
        results.append({'Image Identifier': filename, 'Cross-Sectional Area (cm²)': cross_sectional_area_cm2})

# Create a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(args.output_file, index=False)

print(f'Results saved to {args.output_file}')
