import os
import pickle
import pandas as pd
import numpy as np

def save_filtered_dataset_splits_to_csv(pkl_file, all_sample_file, output_dir):
    """
    Saves filtered dataset splits (train and test) for different folds into CSV files.

    Parameters:
        pkl_file (str): Path to the pickle file containing fold indices.
        all_sample_file (str): Path to the CSV file containing all sample information.
        output_dir (str): Directory to save the output CSV files.
    """
    # Load fold indices from the pickle file
    with open(pkl_file, 'rb') as fh:
        fold_indices = pickle.load(fh)
    
    # Define folds dictionary
    folds = {
        0: {'train': np.concatenate((fold_indices[1], fold_indices[2])), 'test': fold_indices[0]},
        1: {'train': np.concatenate((fold_indices[0], fold_indices[2])), 'test': fold_indices[1]},
        2: {'train': np.concatenate((fold_indices[0], fold_indices[1])), 'test': fold_indices[2]},
        'holdout': {'train': [], 'test': fold_indices[3]}
    }
    
    # Load all samples CSV
    all_samples = pd.read_csv(all_sample_file).drop(['Unnamed: 0'], axis=1)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each fold and save to CSV
    for fold_num, split in folds.items():
        train_idx = split['train']
        test_idx = split['test']
        
        # # Train split
        train_samples = all_samples.iloc[train_idx]
        output_patch_dir = '/mnt/bulk-ganymede/vidhya/crick/docker/Patch_Data/WSI'
        train_hdf5_list = [
            os.path.join(output_patch_dir, f.replace('.svs', '.hdf5')) for f in train_samples.svs.values
        ]
        is_present = np.array([os.path.exists(f) for f in train_hdf5_list])
        train_samples_filtered = train_samples[is_present]

        train_csv_path = os.path.join(output_dir, f'fold{fold_num}_train.csv')
        train_samples.to_csv(train_csv_path, index=False)
        
        # Test split
        test_samples = all_samples.iloc[test_idx]
        test_hdf5_list = [
            os.path.join(output_patch_dir, f.replace('.svs', '.hdf5')) for f in test_samples.svs.values
        ]
        is_present = np.array([os.path.exists(f) for f in test_hdf5_list])
        test_samples_filtered = test_samples[is_present]

        test_csv_path = os.path.join(output_dir, f'fold{fold_num}_test.csv')
        test_samples.to_csv(test_csv_path, index=False)
    
    print(f"Filtered dataset splits saved to {output_dir}.")

save_filtered_dataset_splits_to_csv(
    "/mnt/bulk-ganymede/vidhya/crick/docker/cancres-2022-intratumoral-heterogeneity-dl-paper/Data_Files/WSI/allGene_Updated_3FoldIndex.pkl",
    "/mnt/bulk-ganymede/vidhya/crick/docker/cancres-2022-intratumoral-heterogeneity-dl-paper/Data_Files/WSI/allSamples_UpdatedCV_BL.csv",
    "/mnt/bulk-ganymede/vidhya/crick/models/acosta/Folds"
)
