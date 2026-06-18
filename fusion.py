import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple


class MatrixLevelAttentionFusion(nn.Module):
    def __init__(self, num_matrices, stat_features=5):
        super().__init__()
        self.num_matrices = num_matrices
        self.attention = nn.Sequential(
            nn.Linear(stat_features, 1),
            nn.Softmax(dim=0)
        )

    def _extract_global_stats(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[
            torch.mean(matrix), torch.max(matrix),
            torch.min(matrix), torch.var(matrix), torch.std(matrix)
        ]], dtype=matrix.dtype, device=matrix.device)

    def forward(self, matrices: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mat_shape = matrices[0].shape
        for mat in matrices:
            assert mat.shape == mat_shape, f"矩阵维度不一致：{mat.shape} vs {mat_shape}"

        global_features = torch.cat([self._extract_global_stats(mat) for mat in matrices], dim=0)
        att_weights = self.attention(global_features)

        fusion_matrix = torch.zeros_like(matrices[0])
        for weight, mat in zip(att_weights, matrices):
            fusion_matrix += weight * mat

        return fusion_matrix, att_weights


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    min_val, max_val = matrix.min(), matrix.max()
    return (matrix - min_val) / (max_val - min_val + 1e-8)


def read_matrix(file_path: str) -> np.ndarray:
    df = pd.read_csv(file_path, index_col=0)
    return df.values


def fuse_similarity_matrices(
    drug_matrix_paths: List[str],
    side_matrix_paths: List[str]
) -> Tuple[np.ndarray, np.ndarray]:

    drug_matrices = [read_matrix(path) for path in drug_matrix_paths]
    drug_tensors = [torch.tensor(normalize_matrix(mat), dtype=torch.float32) for mat in drug_matrices]
    drug_fuser = MatrixLevelAttentionFusion(num_matrices=len(drug_matrix_paths))
    fusion_drug_mat, drug_weights = drug_fuser(drug_tensors)

    side_matrices = [read_matrix(path) for path in side_matrix_paths]
    side_tensors = [torch.tensor(normalize_matrix(mat), dtype=torch.float32) for mat in side_matrices]
    side_fuser = MatrixLevelAttentionFusion(num_matrices=len(side_matrix_paths))
    fusion_side_mat, side_weights = side_fuser(side_tensors)

    print("\n" + "="*60)
    print("【药物相似度矩阵 - 注意力权重】")
    print("（权重越高，对融合结果的贡献越大）")
    for i, path in enumerate(drug_matrix_paths):
        file_name = path.split("/")[-1]  # 提取文件名
        print(f"矩阵 {i+1}：{file_name} → 权重 = {drug_weights[i].item():.4f}")

    print("\n" + "="*60)
    print("【副作用相似度矩阵 - 注意力权重】")
    print("（权重越高，对融合结果的贡献越大）")
    for i, path in enumerate(side_matrix_paths):
        file_name = path.split("/")[-1]
        print(f"矩阵 {i+1}：{file_name} → 权重 = {side_weights[i].item():.4f}")

    print("\n" + "="*60)
    print(f"药物矩阵权重之和：{torch.sum(drug_weights).item():.4f}（应为1.0）")
    print(f"副作用矩阵权重之和：{torch.sum(side_weights).item():.4f}（应为1.0）")

    return fusion_drug_mat.detach().cpu().numpy(), fusion_side_mat.detach().cpu().numpy()
