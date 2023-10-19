import pickle
import torch
import io
import matplotlib.pyplot as plt

# Metrics definitions
def jaccard_similarity(tensor1, tensor2):
    intersection = (tensor1 & tensor2).float().sum()
    union = (tensor1 | tensor2).float().sum()
    return intersection / union

def hamming_similarity(tensor1, tensor2):
    return (tensor1 == tensor2).float().mean()

def cosine_similarity(tensor1, tensor2):
    dot_product = (tensor1.float() * tensor2.float()).sum()
    norm1 = tensor1.float().norm()
    norm2 = tensor2.float().norm()
    return dot_product / (norm1 * norm2)

def euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1.float() - tensor2.float())

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
    # 'hamming': {'layer': [], 'sublayer': [], 'whole': None},
    # 'cosine': {'layer': [], 'sublayer': [], 'whole': None},
    # 'euclidean': {'layer': [], 'sublayer': [], 'whole': None}
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

        # Sublayer metrics
        results['jaccard']['sublayer'].append({
            'name': f"Layer {layer} {sublayer_name}",
            'value': jaccard_similarity(mask1, mask2).cpu().item()
        })

        # results['hamming']['sublayer'].append({
        #     'name': f"Layer {layer} {sublayer_name}",
        #     'value': hamming_similarity(mask1, mask2).cpu().item()
        # })

        # results['cosine']['sublayer'].append({
        #     'name': f"Layer {layer} {sublayer_name}",
        #     'value': cosine_similarity(mask1.view(-1), mask2.view(-1)).cpu().item()
        # })

        # results['euclidean']['sublayer'].append({
        #     'name': f"Layer {layer} {sublayer_name}",
        #     'value': euclidean_distance(mask1, mask2).cpu().item()
        # })


        # Accumulate for layer
        layer_data1.append(mask1.view(-1))
        layer_data2.append(mask2.view(-1))

    # Combine sublayer tensors for layer
    layer_data1 = torch.cat(layer_data1)
    layer_data2 = torch.cat(layer_data2)

    # Layer metrics
    results['jaccard']['layer'].append({
        'name': f"Layer {layer}",
        'value': jaccard_similarity(layer_data1, layer_data2).cpu().item()
    })

    # results['hamming']['layer'].append({
    #     'name': f"Layer {layer}",
    #     'value': hamming_similarity(layer_data1, layer_data2).cpu().item()
    # })

    # results['cosine']['layer'].append({
    #     'name': f"Layer {layer}",
    #     'value': cosine_similarity(layer_data1, layer_data2).cpu().item()
    # })

    # results['euclidean']['layer'].append({
    #     'name': f"Layer {layer}",
    #     'value': euclidean_distance(layer_data1, layer_data2).cpu().item()
    # })

    # Accumulate for whole masks
    whole_data1.append(layer_data1)
    whole_data2.append(layer_data2)

# Combine layer tensors for whole masks
whole_data1 = torch.cat(whole_data1)
whole_data2 = torch.cat(whole_data2)

# Whole masks metrics
results['jaccard']['whole'] = jaccard_similarity(whole_data1, whole_data2).cpu().item()
# results['hamming']['whole'] = hamming_similarity(whole_data1, whole_data2).cpu().item()
# results['cosine']['whole'] = cosine_similarity(whole_data1, whole_data2).cpu().item()
# results['euclidean']['whole'] = euclidean_distance(whole_data1, whole_data2).cpu().item()

print(results)

# Visualization and Saving
for metric, data in results.items():
    # Sublayer
    plt.figure(figsize=(10, 6))
    for sublayer in data['sublayer']:
        plt.hist(sublayer['value'], bins=20, edgecolor='k', alpha=0.7, label=sublayer['name'])
    plt.title(f'{metric.capitalize()} Similarity Distribution - Sublayer')
    plt.xlabel(f'{metric.capitalize()} Similarity')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    if metric in ['jaccard', 'hamming']:
        plt.xlim(0, 1)
    elif metric == 'cosine':
        plt.xlim(-1, 1)
    plt.savefig(f"{combined_mask_name}_{metric}_sublayer.png")
    plt.close()

    # Layer
    plt.figure(figsize=(10, 6))
    values = [layer['value'] for layer in data['layer']]
    names = [layer['name'] for layer in data['layer']]
    plt.bar(names, values, edgecolor='k', alpha=0.7)
    plt.title(f'{metric.capitalize()} Similarity Distribution - Layer')
    plt.xlabel('Layer')
    plt.ylabel(f'{metric.capitalize()} Similarity')
    if metric in ['jaccard', 'hamming']:
        plt.ylim(0, 1)
    elif metric == 'cosine':
        plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{combined_mask_name}_{metric}_layer.png")
    plt.close()

    # Whole
    plt.figure(figsize=(10, 6))
    plt.bar(['whole'], [data['whole']])
    plt.title(f'{metric.capitalize()} Similarity - Whole')
    plt.xlabel('Granularity')
    plt.ylabel(f'{metric.capitalize()} Similarity')
    
    if metric in ['jaccard', 'hamming']:
        plt.xlim(0, 1)
    elif metric == 'cosine':
        plt.xlim(-1, 1)
    
    plt.savefig(f"{combined_mask_name}_{metric}_whole.png")
    plt.close()
