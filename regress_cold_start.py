import argparse
import random
import numpy as np
import pandas as pd
import torch
from regress_model import drugse, Optimizer
from myutils import *
from fusion import fuse_similarity_matrices
import numpy as np

def generate_balanced_kfold_masks(total_rows=664, total_cols=994, n_splits=10, similarity_threshold=0.6, similarity_matrix_path="./Data/drug/664_drug_fingerprint_jaccard_similarity_matrix_new.csv"):
    similarity_matrix = pd.read_csv(similarity_matrix_path, header=None)
    similarity_matrix = similarity_matrix.apply(pd.to_numeric, errors='coerce').to_numpy()
    mask_matrices = []
    removed_drug_records = []
    indices = np.arange(total_rows)
    np.random.shuffle(indices)
    fold_sizes = [total_rows // n_splits] * n_splits
    for i in range(total_rows % n_splits):
        fold_sizes[i] += 1

    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        mask = np.zeros((total_rows, total_cols), dtype=bool)
        current_fold_drugs = indices[start:end]
        train_drugs = np.setdiff1d(indices, current_fold_drugs)
        test_drugs_candidate = current_fold_drugs
        test_drugs_final = []
        removed_drugs = []
        for test_idx in test_drugs_candidate:
            is_too_similar = False
            for train_idx in train_drugs:
                if similarity_matrix[test_idx, train_idx] > similarity_threshold:
                    is_too_similar = True
                    break

            if is_too_similar:
              removed_drugs.append(test_idx)
            else:
                test_drugs_final.append(test_idx)

        test_drugs_final = np.array(test_drugs_final)
        print(f"Fold 训练集药物数：{len(train_drugs)}, 测试集药物数：{len(test_drugs_final)}, 移除药物数：{len(removed_drugs)}")
        mask[train_drugs, :] = True
        if len(test_drugs_final) > 0:
            mask[test_drugs_final, :] = False
        mask_matrices.append(mask)
        removed_drug_records.append(set(removed_drugs))
        start = end
    return mask_matrices,removed_drug_records

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_random_seeds(42)
    parser = argparse.ArgumentParser(description="Run DrugSE Cold Start Experiment")
    parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="the weight decay for l2 normalization")
    parser.add_argument('--layer_size', nargs='*', type=int, default=[2048, 4096], help='Output sizes of every layer')
    parser.add_argument('--alpha', type=float, default=0.2, help="the scale for balance gcn and ni")
    parser.add_argument('--gamma', type=float, default=8, help="the scale for sigmoid")
    parser.add_argument('--epochs', type=int, default=3000, help="the epochs for model")
    args = parser.parse_args()


    ideal_kernel_df, ideal_kernel_values = read_csv_with_labels(
        './Data/Frequency_664/Drug-Side_Effect_Frequency664.csv')
    drug_names = ideal_kernel_df.index.tolist()
    side_names = ideal_kernel_df.columns.tolist()

    drug_paths = [
        "./Data/drug/664_drug_drug_scores.csv",
        "./Data/drug/664_drug_fingerprint_jaccard_similarity_matrix_new.csv",
    ]

    # 副作用矩阵路径
    side_paths = [
        "./Data/side/kvplm_Side_Effect_Similarity_Matrix.csv",
        "./Data/side/semantic.csv",
        "./Data/side/word_new.csv"
    ]

    mask_matrices ,removed_drug_records= generate_balanced_kfold_masks(total_rows=664, total_cols=994, n_splits=10)
    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()

    RMSE = 0
    MAE = 0
    PCC = 0

    for fold_idx, mask in enumerate(mask_matrices):
        print(f"Fold {fold_idx + 1}/10")
        ideal_kernel_values_masked = ideal_kernel_values.copy()
        ideal_kernel_values_masked[mask == 0] = 0

        ideal_kernel_drugs = np.dot(ideal_kernel_values_masked, ideal_kernel_values_masked.T)
        ideal_kernel_drugs = kernel_normalized(ideal_kernel_drugs)

        ideal_kernel_sides = np.dot(ideal_kernel_values_masked.T, ideal_kernel_values_masked)
        ideal_kernel_sides = kernel_normalized(ideal_kernel_sides)

        fused_drug_sim, fused_side_sim = fuse_similarity_matrices(drug_paths, side_paths)
        print("mask",mask)
        train_mask = mask
        test_mask = ~mask
        real_test_mask = test_mask.copy()
        for removed_idx in removed_drug_records[fold_idx]:
            real_test_mask[removed_idx, :] = False

        test_mask = real_test_mask

        num_train = np.sum(train_mask[:, 0])
        num_test = np.sum(test_mask[:, 0])
        num_removed = len(removed_drug_records[fold_idx])
        print(f"Fold {fold_idx + 1}: 训练集={num_train}, 测试集={num_test}, 被剔除={num_removed}")
        print(f"  验证：{num_train} + {num_test} + {num_removed} = {num_train + num_test + num_removed} (应该=664)")
        print("test_mask", test_mask)
        train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool).to(args.device)
        test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool).to(args.device)
        num_true_train_mask = torch.sum(train_mask_tensor).item()
        num_true_test_mask = torch.sum(test_mask_tensor).item()

        print(f"train_mask_tensor 中 True 的数量: {num_true_train_mask}")
        print(f"test_mask_tensor 中 True 的数量: {num_true_test_mask}")

        train_data_tensor = torch.from_numpy(ideal_kernel_values_masked).float().to(args.device) * torch.from_numpy(mask).float().to(args.device)
        test_data_tensor = torch.from_numpy(ideal_kernel_values_masked).float().to(args.device) * torch.from_numpy(test_mask).float().to(args.device)

        freq_values_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device)

        model = drugse(adj_mat=train_data_tensor, drug_sim=fused_drug_sim, side_sim=fused_side_sim,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(args.device)

        opt = Optimizer(model, ideal_kernel_values,train_data_tensor, test_data_tensor,
                        train_mask=torch.from_numpy(train_mask).bool().to(args.device),
                        test_mask=torch.from_numpy(test_mask).bool().to(args.device),
                        ap_fun=roc_auc, aupr_fun=aupr, rmse_fun=rmse, mae_fun=mae,pcc_fun=pcc,freq_values=freq_values_tensor,
                        lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)

        true_data, predict_data,rmse_data, mae_data, pcc_data = opt()

        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))


        RMSE += rmse_data
        MAE += mae_data
        PCC += pcc_data

    print("Best RMSE: %.4f" % (RMSE / 10),"Best MAE: %.4f" % (MAE / 10),"Best PCC: %.4f" % (PCC / 10))
