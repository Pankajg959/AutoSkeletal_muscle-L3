# AutoSkeletal_muscle-L3
# This repository contains the code base for the automatic calculation of skeletal muscle cross-sectional area at L3 from a 3D CT volume
This is the repository for automatic deep learning-based calculation of skeletal muscle cross-sectional area (CSA) at the L3 level from a 3D CT volume. L3 skeletal muscle CSA is commonly used in clinical practice to calculate the skeletal muscle index (CSA(cm2)/height (m2)) and is widely accepted as a reliable marker of sarcopenia. The manual segmentation of the skeletal muscles is challenging and time-consuming. 
We train the nnUNet model on 600 CT volume from the SAROS dataset https://github.com/UMEssen/saros-dataset for automatic detection of skeletal muscle only. The rest of the body regions/ parts were not segmented.
The **complete pipeline** comprises using the TotalSegmentor https://github.com/wasserth/TotalSegmentator to segment the L3 vertebral body. This helps extract a single CT slice at the L3 level. The trained skeletal muscle segmentation models then segment the skeletal muscle. Finally, the cross-sectional area of the skeletal muscle pixels is calculated and output as a .csv file.

# Usage
1. install nnUnet "git clone https://github.com/MIC-DKFZ/nnUNet.git", cd nnUNet, pip install -e . 
2. create nnUNet_training/nnUNet_eval, nnUNet_training/predictions, nnUNet_training/preprocessed, nnUNet_training/raw, nnUNet_training/nnUNet_results folders
3. download the model checkpoints from "https://drive.google.com/drive/folders/1QNhIE2QYf7Z5CXueBFL_c62cW8q-9LpW?usp=sharing" and place in nnUNet_training/nnUNet_results 
4. download the sarcopenia_test folder from "https://drive.google.com/drive/folders/1AsAi1goIVaCzryF9_v2oPc7T2HBiU7s3?usp=sharing" and place in the main directory. Use the directories in this folder for running the dummy_test pipeline
5. pip install TotalSegmentator

General_usage
`bash sarcopenia_pipeline.sh -i /path/to/ct_volume_dir -o /path/to/l3_masks_dir -s /path/to/output_l3_slice_dir -p /path/to/output_smi_dir -c /path/to/output_file.csv`

dummy_test_usage: 
`bash sarcopenia_pipeline.sh -i /home/pankaj/AutoSkeletal_muscle-L3/sarcopenia_test/input_nii_image_dir -o /home/pankaj/AutoSkeletal_muscle-L3/sarcopenia_test/L3_mask_dir -s /home/pankaj/AutoSkeletal_muscle-L3/sarcopenia_test/L3_slice_dir -p /home/pankaj/AutoSkeletal_muscle-L3/sarcopenia_test/smi_dir -c /home/pankaj/AutoSkeletal_muscle-L3/sarcopenia_test/CSA_file.csv` (change the path according to your folder location)

Data should be in nnUNet format (files should be in the format 0000_0000.nii.gz, 0001_0000.nii.gz, and so on)
