import torch.optim as optim
from abc import ABC
from myutils import *
import dgl.function as fn
import torch.nn.functional as F
import dgl
import numpy as np
import math


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(out_dim * num_heads)
        self.out_channels = out_dim * num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        scale = math.sqrt(self.out_dim)
        g.apply_edges(lambda edges: {'exp_score': torch.exp((edges.data['score'] / scale).clamp(-5, 5))})
        g.update_all(fn.src_mul_edge('V_h', 'exp_score', 'weighted_V'), fn.sum('weighted_V', 'wV'))
        g.update_all(fn.copy_edge('exp_score', 'exp_score'), fn.sum('exp_score', 'z'))

    def forward(self, g, h):
        h_in = h
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))

        h_att = head_out.view(-1, self.out_dim * self.num_heads)
        h_att = h_att + h_in
        return self.layer_norm(h_att)


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)

        if self.residual:
            h = h_in1 + h
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h
        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)

class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cuda:0"):
        super(ConstructAdjMatrix, self).__init__()
        self.device = device
        self.adj = torch.where(original_adj_mat > 0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        self.adj = self.adj.float()

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1) + 1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0) + 1, -0.5))
        agg_drug_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_side_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1) + 1, -1)
        self_drug_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0) + 1, -1)
        self_side_lp = torch.diag(torch.add(d_d, 1))
        return agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp

# 加载特征类
class LoadFeature(nn.Module, ABC):
    def __init__(self, drug_sim, side_sim, device="cpu"):
        super(LoadFeature, self).__init__()
        drug_sim = torch.from_numpy(drug_sim).to(device).float()
        self.drug_feat = torch_z_normalized(drug_sim, dim=1).to(device).float()
        self.side_feat = torch.from_numpy(side_sim).to(device).float()

    def forward(self):
        drug_feat = self.drug_feat
        side_feat = self.side_feat
        return drug_feat, side_feat

class GEncoder(nn.Module, ABC):
    def __init__(self, agg_d_lp, agg_s_lp, self_d_lp, self_s_lp, drug_feat, side_feat, layer_size, alpha,num_heads, num_layers, dropout=0.0):
        super(GEncoder, self).__init__()
        self.agg_d_lp = agg_d_lp.float()
        self.agg_s_lp = agg_s_lp.float()
        self.self_d_lp = self_d_lp.float()
        self.self_s_lp = self_s_lp.float()
        self.dropout = dropout

        self.layers = layer_size
        self.alpha = alpha
        self.drug_feat = drug_feat.float()
        self.side_feat = side_feat.float()

        self.fc_drug = nn.Linear(self.drug_feat.shape[1], layer_size[0], bias=True)
        self.fc_side = nn.Linear(self.side_feat.shape[1], layer_size[0], bias=True)
        self.ld = nn.BatchNorm1d(layer_size[0])
        self.ls = nn.BatchNorm1d(layer_size[0])
        self.lm_drug = nn.Linear(layer_size[0], layer_size[1], bias=True)
        self.lm_side = nn.Linear(layer_size[0], layer_size[1], bias=True)

        self.drug_transformer_layers = nn.ModuleList()
        self.side_transformer_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.drug_transformer_layers.append(
                GraphTransformerLayer(
                    in_dim=layer_size[0],
                    out_dim=layer_size[0],
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=True,
                    batch_norm=True,
                    residual=True
                )
            )
            self.side_transformer_layers.append(
                GraphTransformerLayer(
                    in_dim=layer_size[0],
                    out_dim=layer_size[0],
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=True,
                    batch_norm=True,
                    residual=True
                )
            )

        num_drugs = drug_feat.shape[0]
        num_sides = side_feat.shape[0]
        device = drug_feat.device

        drug_edges = torch.nonzero(self.self_d_lp > 0, as_tuple=False).T
        self.drug_graph = dgl.graph(
            (drug_edges[0].to(device), drug_edges[1].to(device)),
            num_nodes=num_drugs,
            device=device
        )

        side_edges = torch.nonzero(self.self_s_lp > 0, as_tuple=False).T
        self.side_graph = dgl.graph(
            (side_edges[0].to(device), side_edges[1].to(device)),
            num_nodes=num_sides,
            device=device
        )

    def forward(self):
        drug_fc = self.ld(self.fc_drug(self.drug_feat))
        side_fc = self.ls(self.fc_side(self.side_feat))

        drug_gcn = torch.mm(self.self_d_lp, drug_fc) + torch.mm(self.agg_d_lp, side_fc)
        side_gcn = torch.mm(self.self_s_lp, side_fc) + torch.mm(self.agg_s_lp, drug_fc)

        drug_gcn_out = F.relu(drug_gcn)
        side_gcn_out = F.relu(side_gcn)

        self.drug_graph.ndata['h'] = drug_gcn_out
        self.side_graph.ndata['h'] = side_gcn_out

        for i in range(len(self.drug_transformer_layers)):
            drug_h_gt = self.drug_transformer_layers[i](self.drug_graph, self.drug_graph.ndata['h'])
            self.drug_graph.ndata['h'] = drug_h_gt

            side_h_gt = self.side_transformer_layers[i](self.side_graph, self.side_graph.ndata['h'])
            self.side_graph.ndata['h'] = side_h_gt

        drug_gt_out = self.drug_graph.ndata['h']
        side_gt_out = self.side_graph.ndata['h']

        drug_features = (1 - self.alpha) * drug_gcn_out + self.alpha * drug_gt_out
        side_features = (1 - self.alpha) * side_gcn_out + self.alpha * side_gt_out

        drug_emb = F.relu(self.lm_drug(drug_features))
        side_emb = F.relu(self.lm_side(side_features))


        return drug_emb, side_emb

class GDecoder(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, drug_emb, side_emb):
        Corr = torch_corr_x_y(drug_emb, side_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output


class drugse(nn.Module, ABC):
    def __init__(self, adj_mat, drug_sim, side_sim, layer_size, alpha, gamma, device="cuda:0"):
        super(drugse, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(drug_sim, side_sim, device=device)
        agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp = construct_adj_matrix()
        drug_feat, side_feat = loadfeat()
        self.encoder = GEncoder(agg_drug_lp, agg_side_lp, self_drug_lp, self_side_lp,
                                drug_feat, side_feat, layer_size, alpha,num_heads=8, num_layers=2, dropout=0.1)
        self.decoder = GDecoder(gamma=gamma)

    def forward(self):
        drug_emb, side_emb = self.encoder()
        output = self.decoder(drug_emb, side_emb)
        return output

class Optimizer(nn.Module, ABC):
    def __init__(self, model,adj, train_data, test_data, test_mask, train_mask, ap_fun, aupr_fun, rmse_fun, mae_fun, pcc_fun,lr=0.0001, wd=1e-05, epochs=800, test_freq=1000, device="cpu", lam=0.01, eps=1e-8,freq_values=None):
        super(Optimizer, self).__init__()
        self.adj= torch.tensor(adj).to(device).float()
        self.model = model.to(device)
        self.train_data = train_data.to(device).float()
        self.test_data = test_data.to(device).float()
        self.test_mask = test_mask.to(device).float().bool()
        self.train_mask = train_mask.to(device).float().bool()

        self.ap_fun = ap_fun
        self.aupr_fun = aupr_fun
        self.rmse_fun = rmse_fun
        self.mae_fun = mae_fun
        self.pcc_fun = pcc_fun
        self.freq_values = freq_values.to(device)
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.lam = lam
        self.eps = eps
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        for epoch in torch.arange(self.epochs):
            true_data = torch.masked_select(self.adj, self.train_mask).long()
            true_data_label = torch.where(true_data > 0, torch.tensor(1, device=true_data.device), true_data)
            best_predict = 0
            best_auc = 0
            best_aupr = 0
            best_rmse = float('inf')
            best_mae = float('inf')
            best_pcc = float('inf')
            predict_data = self.model()
            non_zero_mask = (self.train_data > 0) & self.train_mask.bool()
            train_data_selected = self.train_data * non_zero_mask
            predict_data_selected = predict_data * non_zero_mask
            mask_selected = non_zero_mask.float()
            freq_values_selected = self.freq_values * non_zero_mask

            loss = weighted_mse_loss(
                train_data_selected,
                predict_data_selected,
                mask_selected,
                freq_values_selected
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            predict_data_masked = torch.masked_select(predict_data, self.train_mask)

            auc = self.ap_fun(true_data_label, predict_data_masked)

            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)

            if epoch % self.test_freq == 0:
                print(f"epoch:{epoch.item():4d} loss:{loss.item():.6f}")
        with torch.no_grad():
            self.model.eval()
            predict_data = self.model()

            self.test_mask = self.test_mask.bool()
            indices = torch.nonzero(self.test_mask, as_tuple=True)
            row_indices, col_indices = indices
            true_test_data = self.adj[row_indices, col_indices].long()

            test_mask_np = self.test_mask.cpu().numpy()
            true_test_data_np = true_test_data.cpu().numpy()
            has_positive = (true_test_data > 0).sum().item() > 0
            num_true = self.test_mask.sum().item()
            print(num_true)

            true_data_label = torch.where(true_test_data > 0, torch.tensor(1, device=true_test_data.device), true_test_data)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)

            non_zero_indices = torch.nonzero(true_test_data, as_tuple=True)
            filtered_true_data = true_test_data[non_zero_indices]
            filtered_predict_data = predict_data_masked[non_zero_indices]
            rmse = self.rmse_fun(filtered_true_data, filtered_predict_data)
            mae = self.mae_fun(filtered_true_data, filtered_predict_data)
            pcc=self.pcc_fun(filtered_true_data, filtered_predict_data)
            print(f"Final Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC:{pcc:.4f}")

            best_rmse = rmse
            best_mae = mae
            best_pcc = pcc
            best_predict = predict_data_masked

        print("Fit finished.")

        return true_test_data, best_predict, best_rmse, best_mae, best_pcc