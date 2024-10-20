#!/bin/bash
# Usage: bash sarcopenia_pipeline.sh -i /path/to/ct_volume_dir -o /path/to/l3_masks_dir -s /path/to/output_l3_slice_dir -p /path/to/output_smi_dir -c /path/to/output_file.csv

# Function to display usage
usage() {
    echo "Usage: $0 -i <input_dir> -o <output_dir> -s <output_l3_slice_dir> -p <output_smi_dir> -c <output_csv_file>"
    exit 1
}

# Parse command-line options
while getopts "i:o:s:p:c:" opt; do
    case ${opt} in
        i) input_dir="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;  # This will serve as both the output masks directory and l3_mask_dir
        s) output_l3_slice_dir="$OPTARG" ;;
        p) output_smi_dir="$OPTARG" ;;
        c) output_csv_file="$OPTARG" ;;  # New option for CSV output
        *) usage ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$input_dir" ] || [ -z "$output_dir" ] || [ -z "$output_l3_slice_dir" ] || [ -z "$output_smi_dir" ] || [ -z "$output_csv_file" ]; then
    usage
fi

# Set environment variables
export nnUNet_results="nnUNet_training/nnUNet_results"
export nnUNet_raw="nnUNet_training/nnUNet_raw"
export nnUNet_preprocessed="nnUNet_training/nnUNet_preprocessed"
export nnUNet_predictions="nnUNet_training/nnUNet_predictions"
export nnUNet_eval="nnUNet_training/nnUNet_eval"

GPU_ID=1  # Put the GPU IDs here
DATASET=560
DATASET_NAME=Dataset560_BCA_2d_regions
TRAINER=nnUNetTrainer_100epochs

# Loop through all .nii.gz files in the input directory
for file in "$input_dir"/*.nii.gz; do
    # Get the base name of the file (without directory and extension)
    base_name=$(basename "$file" .nii.gz)
    
    # Create a temporary output folder for each file
    temp_output_dir="$output_dir/$base_name"
    
    echo "Processing file: $file"
    echo "Saving segmentation to temporary folder: $temp_output_dir"
    
    # Run TotalSegmentator and save output to temporary folder
    if ! TotalSegmentator -i "$file" -o "$temp_output_dir" --roi_subset vertebrae_L3; then
        echo "Error processing $file with TotalSegmentator"
        continue
    fi
    
    # Move the segmentation file and rename it to match the input CT volume (with original name)
    if ! mv "$temp_output_dir/vertebrae_L3.nii.gz" "$output_dir/$base_name.nii.gz"; then
        echo "Error moving file to $output_dir"
        continue
    fi
    
    # Optionally remove the temporary folder (since it's now empty)
    rmdir "$temp_output_dir" || echo "Warning: Could not remove temporary folder $temp_output_dir"
done

echo "All CT volumes processed. Proceeding to extract L3 slices..."

# Call Python scripts
if ! python training/extractl3.py --ct_volume_dir "$input_dir" --l3_mask_dir "$output_dir" --output_dir "$output_l3_slice_dir"; then
    echo "Error extracting L3 slices"
    exit 1
fi

# Run nnU-Net prediction
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_predict -d $DATASET -i "$output_l3_slice_dir" -o "$output_smi_dir" -tr $TRAINER -c 2d -p nnUNetPlans

# Call CSA script with updated argument names
if ! python training/CSA.py --input_directory "$output_smi_dir" --output_file "$output_csv_file"; then
    echo "Error calculating cross-sectional areas"
    exit 1
fi

echo "Processing complete."
