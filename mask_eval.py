import pickle
import torch
import io
import matplotlib.pyplot as plt
import numpy as np
import re

# Metrics definitions
def jaccard_similarity(tensor1, tensor2):
    intersection = (tensor1 & tensor2).float().sum()
    union = (tensor1 | tensor2).float().sum()
    return (intersection / union).item()

def hamming_similarity(tensor1, tensor2):
    return (tensor1 == tensor2).float().mean().item()

def cosine_similarity(tensor1, tensor2):
    dot_product = (tensor1.float() * tensor2.float()).sum()
    norm1 = tensor1.float().norm()
    norm2 = tensor2.float().norm()
    return (dot_product / (norm1 * norm2)).item()

def euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1.float() - tensor2.float()).item()

# Function to sort layer data based on layer number
def sort_layer_data(results_dict):
    for metric, data in results_dict.items():
        results_dict[metric]['layer'] = sorted(data['layer'], key=lambda x: int(re.search(r"Layer (\d+)", x['name']).group(1)))

# Load data from both files
mask_path1 = "/home/alireza/wanda/wanda_mask_c4.pkl"
mask_path2 = "/home/alireza/wanda/wanda_mask_alpaca.pkl"

with open(mask_path1, "rb") as f:
    data1 = pickle.load(f)

with open(mask_path2, "rb") as f:
    data2 = pickle.load(f)

if isinstance(data1, bytes):
    data1 = torch.load(io.BytesIO(data1), map_location=torch.device('cpu'))

if isinstance(data2, bytes):
    data2 = torch.load(io.BytesIO(data2), map_location=torch.device('cpu'))

# Extracting mask names for saved file names
mask_name1 = mask_path1.split('/')[-1].split('.')[0]
mask_name2 = mask_path2.split('/')[-1].split('.')[0]
combined_mask_name = f"{mask_name1}_vs_{mask_name2}"

# Metrics accumulators
results = {
    'jaccard': {'layer': [], 'sublayer': [], 'whole': None},
    'hamming': {'layer': [], 'sublayer': [], 'whole': None},
    'cosine': {'layer': [], 'sublayer': [], 'whole': None},
    'euclidean': {'layer': [], 'sublayer': [], 'whole': None}
}

whole_data1 = []
whole_data2 = []

for layer, sublayers in data1.items():
    layer_data1 = []
    layer_data2 = []

    # For debugging purposes
    # if layer != 2:
    #     continue

    for sublayer_name, mask1 in sublayers.items():
        mask2 = data2[layer][sublayer_name]

        # Move masks to CPU
        mask1 = mask1.to('cpu')
        mask2 = mask2.to('cpu')

        # Sublayer metrics
        results['jaccard']['sublayer'].append({
            'name': f"Layer {layer} Sublayer {sublayer_name} (Shape: {mask1.shape})",
            'value': jaccard_similarity(mask1, mask2)
        })

        results['hamming']['sublayer'].append({
            'name': f"Layer {layer} Sublayer {sublayer_name} (Shape: {mask1.shape})",
            'value': hamming_similarity(mask1, mask2)
        })

        results['cosine']['sublayer'].append({
            'name': f"Layer {layer} Sublayer {sublayer_name} (Shape: {mask1.shape})",
            'value': cosine_similarity(mask1.view(-1), mask2.view(-1))
        })

        results['euclidean']['sublayer'].append({
            'name': f"Layer {layer} Sublayer {sublayer_name} (Shape: {mask1.shape})",
            'value': euclidean_distance(mask1, mask2)
        })


        # Accumulate for layer
        layer_data1.append(mask1.view(-1))
        layer_data2.append(mask2.view(-1))

    # Combine sublayer tensors for layer
    layer_data1 = torch.cat(layer_data1)
    layer_data2 = torch.cat(layer_data2)

    # Layer metrics
    results['jaccard']['layer'].append({
        'name': f"Layer {layer}",
        'value': jaccard_similarity(layer_data1, layer_data2)
    })

    results['hamming']['layer'].append({
        'name': f"Layer {layer}",
        'value': hamming_similarity(layer_data1, layer_data2)
    })

    results['cosine']['layer'].append({
        'name': f"Layer {layer}",
        'value': cosine_similarity(layer_data1, layer_data2)
    })

    results['euclidean']['layer'].append({
        'name': f"Layer {layer}",
        'value': euclidean_distance(layer_data1, layer_data2)
    })

    # Accumulate for whole masks
    whole_data1.append(layer_data1)
    whole_data2.append(layer_data2)

# Combine layer tensors for whole masks
whole_data1 = torch.cat(whole_data1)
whole_data2 = torch.cat(whole_data2)

# Whole masks metrics
results['jaccard']['whole'] = jaccard_similarity(whole_data1, whole_data2)
results['hamming']['whole'] = hamming_similarity(whole_data1, whole_data2)
results['cosine']['whole'] = cosine_similarity(whole_data1, whole_data2)
results['euclidean']['whole'] = euclidean_distance(whole_data1, whole_data2)

print(results)

# Apply the sorting function to your results
sort_layer_data(results)

# Visualization and Saving
for metric, data in results.items():
    # Sort sublayer data based on the metric values
    sorted_sublayer_data = sorted(data['sublayer'], key=lambda x: x['value'], reverse=False)

    # 1. A figure comparing the metrics for different layers
    plt.figure(figsize=(10, 6))
    layer_values = [layer['value'] for layer in data['layer']]
    layer_names = [layer['name'] for layer in data['layer']]
    plt.bar(layer_names, layer_values, edgecolor='k', alpha=0.7)
    plt.title(f'{metric.capitalize()} Similarity Distribution - Layer')
    plt.xlabel('Layer')
    plt.ylabel(f'{metric.capitalize()} Similarity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if metric in ['jaccard', 'hamming']:
        plt.ylim(0, 1)
    elif metric == 'cosine':
        plt.ylim(-1, 1)
    plt.savefig(f"{combined_mask_name}_{metric}_layers_comparison.png")
    plt.close()

    # 2. A figure comparing the whole metrics for each mask
    plt.figure(figsize=(10, 6))
    plt.bar(['whole'], [data['whole']])
    plt.title(f'{metric.capitalize()} Similarity - Whole')
    plt.xlabel('Granularity')
    plt.ylabel(f'{metric.capitalize()} Similarity')
    if metric in ['jaccard', 'hamming']:
        plt.ylim(0, 1)
    elif metric == 'cosine':
        plt.ylim(-1, 1)
    plt.savefig(f"{combined_mask_name}_{metric}_whole_comparison.png")
    plt.close()

    # 3. A figure per layer comparing sublayers
    # First, group the sublayers by layer
    sublayer_groups = {}
    for sublayer in sorted_sublayer_data:
        layer_match = re.search(r'Layer (\d+)', sublayer['name'])
        layer_number = layer_match.group(1) if layer_match else 'unknown'

        if layer_number not in sublayer_groups:
            sublayer_groups[layer_number] = []
        sublayer_groups[layer_number].append(sublayer)

    for layer_number, sublayers in sublayer_groups.items():
        plt.figure(figsize=(12, 7))
        for sublayer in sublayers:
            plt.bar(sublayer['name'], sublayer['value'], edgecolor='k', alpha=0.7, label=sublayer['name'])
        plt.title(f'{metric.capitalize()} Similarity Distribution - Layer {layer_number} Sublayers')
        plt.xlabel('Sublayer')
        plt.ylabel(f'{metric.capitalize()} Similarity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if metric in ['jaccard', 'hamming']:
            plt.ylim(0, 1)
        elif metric == 'cosine':
            plt.ylim(-1, 1)
        plt.savefig(f"{combined_mask_name}_layer_{layer_number}_{metric}_sublayer_comparison.png")
        plt.close()