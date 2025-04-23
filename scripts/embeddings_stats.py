from itertools import combinations

import numpy as np
import argparse
import os

from torch.nn.functional import cosine_similarity
import torch


def load_tensor(file_path, file_name):
    path = os.path.join(file_path, file_name)
    tensor = torch.load(path, map_location=torch.device('cpu'))
    return tensor.squeeze(0)


def analyze_embeddings(tensors):
    all_embeddings = torch.cat(tensors, dim=0)  # [num_tensors*20, 512]

    # base stats
    mean_val = all_embeddings.mean().item()
    std_val = all_embeddings.std().item()

    # covariance
    emb_np = all_embeddings.numpy()
    cov_matrix = np.cov(emb_np, rowvar=False)  # [512 x 512]

    cov_trace = np.trace(cov_matrix)
    cov_fro = np.linalg.norm(cov_matrix, ord='fro')
    cov_mean_diag = np.mean(np.diag(cov_matrix))
    cov_mean_offdiag = (np.sum(cov_matrix) - np.trace(cov_matrix)) / (cov_matrix.size - 512)

    # vector L2 norm
    l2_norms = torch.norm(all_embeddings, dim=1)
    l2_mean = l2_norms.mean().item()
    l2_std = l2_norms.std().item()

    # cosine similarity intra-tensor
    cosine_sims = []
    for t in tensors:
        pairs = list(combinations(t, 2))  # tensor's embedding couples[20, 512]
        sims = [cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item() for a, b in pairs]
        cosine_sims.extend(sims)

    cosine_sims = np.array(cosine_sims)
    cos_mean = float(np.mean(cosine_sims))
    cos_std = float(np.std(cosine_sims))
    cos_max = float(np.max(cosine_sims))

    return {
        'num_tensors': len(tensors),
        'mean_value': mean_val,
        'std_value': std_val,
        'cov_trace': cov_trace,
        'cov_frobenius_norm': cov_fro,
        'cov_mean_diag': cov_mean_diag,
        'cov_mean_offdiag': cov_mean_offdiag,
        'l2_norm_mean': l2_mean,
        'l2_norm_std': l2_std,
        'cosine_similarity_mean': cos_mean,
        'cosine_similarity_std': cos_std,
        'cosine_similarity_max': cos_max,
    }


def save_stats(stats, save_path):
    with open(save_path, 'w') as f:
        for key, val in stats.items():
            f.write(f"{key}: {val:.6f}\n")


def main(args):
    base_dir = './artifacts'
    env_dir = args.env
    model_dir = args.model
    tensor_dir = 'tensors'
    artifact_type = args.artifact_type

    model_artifacts_path = os.path.join(base_dir, env_dir, model_dir)
    tensor_path = os.path.join(model_artifacts_path, tensor_dir)

    tensors = []
    for tensor_file in sorted(os.listdir(tensor_path)):
        if tensor_file.startswith(artifact_type) and tensor_file.endswith('.pt'):
            tensor = load_tensor(tensor_path, tensor_file)  # [20, 512]
            tensors.append(tensor)

    stats = analyze_embeddings(tensors)

    stats_path = os.path.join(model_artifacts_path, f'stats_{args.artifact_type}.txt')
    save_stats(stats, stats_path)
    print(f'completed -> stats saved in: {stats_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding analysis')

    parser.add_argument(
        '--env',
        type=str,
        choices=['ant', 'halfcheetah', 'hopper', 'walker2d'],
        help='Environment folder (one of: ant, halfcheetah, hopper, walker2d)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'sil_3l', 'til_3l'],
        help='Model folder (one of: baseline, sil_3l, til_3l)'
    )

    parser.add_argument(
        '--artifact_type',
        type=str,
        choices=['state_embeddings'],
        help='Artifact type (one of: state_embeddings)'
    )

    main_args = parser.parse_args()
    main(main_args)