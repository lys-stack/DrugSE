import argparse
from sklearn.model_selection import KFold
from regress_model import drugse, Optimizer
from myutils import *
import numpy as np
import random
from fusion import fuse_similarity_matrices

def generate_balanced_kfold_masks(DAL, n_splits=10):
    positive_samples = np.array([(i, j) for i in range(DAL.shape[0]) for j in range(DAL.shape[1]) if DAL[i, j] != 0])
    negative_samples = np.array([(i, j) for i in range(DAL.shape[0]) for j in range(DAL.shape[1]) if DAL[i, j] == 0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_masks, test_masks = [], []

    for train_idx, test_idx in kf.split(positive_samples):
        train_mask = np.zeros_like(DAL, dtype=bool)
        test_mask = np.zeros_like(DAL, dtype=bool)
        train_pos = positive_samples[train_idx]
        test_pos = positive_samples[test_idx]
        for i, j in train_pos:
            train_mask[i, j] = True

        test_pos_rows = set(i for i, j in test_pos)
        extra_neg = [(i, j) for i in test_pos_rows for j in range(DAL.shape[1]) if DAL[i, j] == 0]
        for i, j in negative_samples:
            train_mask[i, j] = True
        for i, j in extra_neg:
            train_mask[i, j] = True
        for i, j in test_pos:
            test_mask[i, j] = True
        for i, j in negative_samples:
            test_mask[i, j] = True

        train_masks.append(train_mask)
        test_masks.append(test_mask)

    return train_masks, test_masks

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
    parser = argparse.ArgumentParser(description="Run DrugSE Hot Start Experiment")
    parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="the weight decay for l2 normalization")
    parser.add_argument('--layer_size', nargs='*', type=int, default=[2048, 4096], help='Output sizes of every layer')
    parser.add_argument('--alpha', type=float, default=0.2, help="the scale for balance gcn and ni")
    parser.add_argument('--gamma', type=float, default=8, help="the scale for sigmoid")
    parser.add_argument('--epochs', type=int, default=3000, help="the epochs for model")
    args = parser.parse_args()


    ideal_kernel_df, ideal_kernel_values = read_csv_with_labels('./Data/Frequency_664/Drug-Side_Effect_Frequency664.csv')
    drug_names = ideal_kernel_df.index.tolist()
    side_names = ideal_kernel_df.columns.tolist()
    drug_paths = [
        "./Data/drug/664_drug_drug_scores.csv",
        "./Data/drug/664_drug_fingerprint_jaccard_similarity_matrix_new.csv",
    ]

    side_paths = [
        "./Data/side/kvplm_Side_Effect_Similarity_Matrix.csv",
        "./Data/side/semantic.csv",
        "./Data/side/word_new.csv"
    ]

    train_masks, test_masks = generate_balanced_kfold_masks(ideal_kernel_values, n_splits=10)
    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()
    RMSE = 0
    MAE = 0
    PCC = 0
    n_splits=10

    for i in range(n_splits):
        print(f"Fold {i + 1}/10")
        train_mask_i = train_masks[i]
        test_mask_i = test_masks[i]

        ideal_kernel_values_masked = ideal_kernel_values.copy()
        ideal_kernel_values_masked[test_mask_i == 1] = 0

        ideal_kernel_drugs = np.dot(ideal_kernel_values_masked, ideal_kernel_values_masked.T)
        ideal_kernel_drugs = kernel_normalized(ideal_kernel_drugs)

        ideal_kernel_sides = np.dot(ideal_kernel_values_masked.T, ideal_kernel_values_masked)
        ideal_kernel_sides = kernel_normalized(ideal_kernel_sides)

        fused_drug_sim, fused_side_sim = fuse_similarity_matrices(drug_paths, side_paths)

        train_mask_tensor = torch.tensor(train_mask_i, dtype=torch.bool).to(args.device)
        test_mask_tensor = torch.tensor(test_mask_i, dtype=torch.bool).to(args.device)
        num_true_train_mask = torch.sum(train_mask_tensor).item()
        num_true_test_mask = torch.sum(test_mask_tensor).item()

        print(f"train_mask_tensor 中 True 的数量: {num_true_train_mask}")
        print(f"test_mask_tensor 中 True 的数量: {num_true_test_mask}")

        train_data_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device) * torch.from_numpy(train_mask_i).float().to(args.device)
        test_data_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device) * torch.from_numpy(test_mask_i).float().to(args.device)

        freq_values_tensor = torch.from_numpy(ideal_kernel_values).float().to(args.device)

        model = drugse(adj_mat=train_data_tensor, drug_sim=fused_drug_sim, side_sim=fused_side_sim,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device).to(args.device)

        opt = Optimizer(model, ideal_kernel_values,train_data_tensor, test_data_tensor,
                        train_mask=torch.from_numpy(train_mask_i).bool().to(args.device),
                        test_mask=torch.from_numpy(test_mask_i).bool().to(args.device),
                        ap_fun=roc_auc, aupr_fun=aupr, rmse_fun=rmse, mae_fun=mae,pcc_fun=pcc,freq_values=freq_values_tensor,
                        lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)

        true_data, predict_data, rmse_data, mae_data, pcc_data = opt()

        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))
        RMSE += rmse_data
        MAE += mae_data
        PCC+=pcc_data
    print("Best RMSE: %.4f" % (RMSE / 10),"Best MAE: %.4f" % (MAE / 10),"Best PCC: %.4f" % (PCC / 10))


