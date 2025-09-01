import os
import h5py
import torch
from transformers import AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_cache_dir = '/data/horse/ws/srsa552c-CONCHLLAVA/hf-TITAN'

model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True, cache_dir=custom_cache_dir)
model = model.to(device)
model.eval()  # Set the model to evaluation mode
print('Loaded Model')

# Path configurations
input_dir = "/data/horse/ws/srsa552c-CONCHLLAVA/TCGA/CONCHv1_5/PENDING/conch1_5-778e1572"  # Replace with your input folder path
output_dir = "/data/horse/ws/srsa552c-CONCHLLAVA/TCGA/TITAN"  # Replace with your output folder path
os.makedirs(output_dir, exist_ok=True)

# Parameters
patch_size_lv0 = 512  # Patch size

# Get a list of input files
input_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]

# Process files with progress bar
for h5_file in tqdm(input_files, desc="Processing H5 files"):
    input_path = os.path.join(input_dir, h5_file)
    output_path = os.path.join(output_dir, h5_file)

    # Skip if the file already exists in the output directory
    if os.path.exists(output_path):
        tqdm.write(f"Skipping {h5_file}: Output already exists.")
        continue

    with h5py.File(input_path, 'r') as file:
        # Load features and coords
        features = torch.from_numpy(file['feats'][:]).to(device)
        coords = torch.from_numpy(file['coords'][:]).to(device)
        coords = coords.to(torch.long)

        # Extract slide embedding
        features = features.float()
        with torch.inference_mode():
            slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
        #with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
            #slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

        # Save the embedding
        with h5py.File(output_path, 'w') as output_file:
            output_file.create_dataset('feats', data=slide_embedding.cpu().numpy())

        tqdm.write(f"Processed {h5_file} and saved to output.")

print("Processing complete. Embeddings saved to:", output_dir)
