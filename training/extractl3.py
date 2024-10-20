import SimpleITK as sitk
import numpy as np
import argparse
import os

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Extract L3 vertebra slice from multiple CT volumes.")
    parser.add_argument('--ct_volume_dir', type=str, required=True, help="Directory containing the 3D CT volumes (NIfTI files).")
    parser.add_argument('--l3_mask_dir', type=str, required=True, help="Directory containing the L3 vertebra segmentation masks (NIfTI files).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output slices.")
    return parser.parse_args()

# Function to process each CT volume and corresponding L3 mask
def process_ct_volume(ct_volume_path, l3_mask_path, output_dir):
    # Load the original 3D CT volume
    ct_volume = sitk.ReadImage(ct_volume_path)
    ct_array = sitk.GetArrayFromImage(ct_volume)

    # Load the segmented L3 mask (binary: 1 for L3 and 0 for background)
    l3_mask = sitk.ReadImage(l3_mask_path)
    l3_mask_array = sitk.GetArrayFromImage(l3_mask)

    # Find the axial slice index that contains the L3 vertebra
    l3_slice_sums = np.sum(l3_mask_array, axis=(1, 2))
    l3_slice_index = np.argmax(l3_slice_sums)  # Find the slice with the most L3 voxels

    # Extract the 2D slice from the CT volume
    ct_slice = ct_array[l3_slice_index, :, :]

    # Create SimpleITK image from the slice
    ct_slice_image = sitk.GetImageFromArray(ct_slice)

    # Generate output filename based on the input CT volume filename
    base_filename = os.path.basename(ct_volume_path)  # Keep the original name
    ct_slice_nifti_path = os.path.join(output_dir, f"{base_filename}")  # Save with the same name

    # Save the extracted CT slice in NIfTI format
    sitk.WriteImage(ct_slice_image, ct_slice_nifti_path)

# Main function to process all CT volumes in the directory
def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop through all NIfTI files in the CT volume directory
    for ct_file in os.listdir(args.ct_volume_dir):
        if ct_file.endswith(".nii.gz"):
            ct_volume_path = os.path.join(args.ct_volume_dir, ct_file)

            # Assume the L3 mask has the same base filename as the CT volume
            l3_mask_path = os.path.join(args.l3_mask_dir, ct_file)
            if os.path.exists(l3_mask_path):
                print(f"Processing {ct_file}...")
                process_ct_volume(ct_volume_path, l3_mask_path, args.output_dir)
            else:
                print(f"L3 mask not found for {ct_file}. Skipping.")

if __name__ == "__main__":
    main()
