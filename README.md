# AutoSkeletal_muscle-L3
# This repository contains the code base for automatic calculation of skeletal muscle cross-sectional area at L3 from a 3D CT volume
This is the repository for automatic deep learning based calculation of skeletal muscle cross sectional area (CSA) at L3 level from a 3D CT volume. L3 skeletal muscle CSA is commonly used in clinical practice to calculate the skeletal muscle index (CSA(cm2)/height (m2)), widely accepted as a reliable marker of sarcopenia. The manual segmentation of the skeletal muscles is challenging and time consuming. 
We train nnUNet model on 600 CT volume from the SAROS dataset https://github.com/UMEssen/saros-dataset for automatic detection of skeletal muscle only. The rest of the body regions/ parts were not segmented.
The **complete pipeline** comprises using the TotalSegmentor https://github.com/wasserth/TotalSegmentator for segmentation of L3 vertebral body. This helps extracting a single CT slice at L3 level. The trained skeletal muscle segmentation models then segments the skeletal muscle. Finally, the cross-sectional area of the skeletal muscle pixels is calculated and output as a .csv file.

# Usage
`bash sarcopenia_pipeline.sh -i /path/to/ct_volume_dir -o /path/to/l3_masks_dir -s /path/to/output_l3_slice_dir -p /path/to/output_smi_dir -c /path/to/output_file.csv`
