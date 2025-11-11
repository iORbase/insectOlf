# -*- coding: utf-8 -*-
import os
import pandas as pd
import utils
import h5py
import numpy as np
import torch
import random
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pickle
import json
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from utils import set_random_seed, read_embeddings, preprocess_data, save_hyperparameters, setup_fold_directory, save_data_to_csv, setup_tensorboard_writer, MLP, train_mlp, SEED

# 固定随机种子
set_random_seed()

class MLP(nn.Module):
    """
    定义多层感知机模型
    """
    def __init__(self, input_dim=1792, hidden_dim=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def preprocess_data(binding_csv_path, protein_dict=None, smile_dict=None):
    """
    数据预处理函数
    :param binding_csv_path: 绑定数据的 CSV 文件路径
    :param protein_dict: 蛋白质嵌入向量字典
    :param smile_dict: 小分子嵌入向量字典
    :return: 处理后的数据 X, y 和对应的 ID 列表
    """
    dir_path = os.path.dirname(binding_csv_path)
    dataset_type = os.path.basename(binding_csv_path).split('_')[1].split('.')[0]
    protein_h5_path = os.path.join(dir_path, f'../embeddings/per_protein_embeddings_{dataset_type}.h5')
    smile_h5_path = os.path.join(dir_path, f'../embeddings/per_smile_embeddings_{dataset_type}.h5')
    protein_dict, smile_dict = read_embeddings(protein_h5_path, smile_h5_path)
    from utils import preprocess_data as utils_preprocess_data
    return utils_preprocess_data(binding_csv_path, protein_dict, smile_dict)


def train_model(model_type, X, y, protein_id_list, smile_id_list, output_dir, split_type='random', use_pretrain=False, binding_csv_path=''):
    """
    训练模型函数
    :param model_type: 模型类型，包括 'LR', 'MLP', 'RF'
    :param X: 特征数据
    :param y: 标签数据
    :param protein_id_list: 蛋白质 ID 列表
    :param smile_id_list: 小分子 ID 列表
    :param output_dir: 输出结果的目录
    :param split_type: 数据划分方式，包括 'random', 'OR', 'VOC'
    :param use_pretrain: 是否使用预训练模型
    :param binding_csv_path: 绑定数据的 CSV 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    summary_records = []
    
    # 读取蛋白质和小分子嵌入数据
    dir_path = os.path.dirname(binding_csv_path)
    dataset_type = os.path.basename(binding_csv_path).split('_')[1].split('.')[0]
    protein_h5_path = os.path.join(dir_path, f'../embeddings/per_protein_embeddings_{dataset_type}.h5')
    smile_h5_path = os.path.join(dir_path, f'../embeddings/per_smile_embeddings_{dataset_type}.h5')
    
    with h5py.File(protein_h5_path, 'r') as f_protein:
        protein_dict = {k: f_protein[k][:] for k in f_protein.keys()}
    
    with h5py.File(smile_h5_path, 'r') as f_smile:
        smile_dict = {k: f_smile[k][:] for k in f_smile.keys()}

    if model_type == 'LR':
        # 线性回归模型
        fit_intercept = True
        hyperparams = {
            'fit_intercept': fit_intercept,
            'n_splits': n_splits,
            'seed': SEED
        }
        with open(os.path.join(output_dir, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(hyperparams, f)
        with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f'Fold {fold+1}/{n_splits}')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            prot_train = np.array(protein_id_list)[train_index]
            prot_test = np.array(protein_id_list)[test_index]
            smile_train = np.array(smile_id_list)[train_index]
            smile_test = np.array(smile_id_list)[test_index]
            y_test_gt = y[test_index]

            fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
            os.makedirs(fold_dir, exist_ok=True)

            with open(os.path.join(fold_dir, 'train_indices.pkl'), 'wb') as f:
                pickle.dump(train_index, f)
            with open(os.path.join(fold_dir, 'test_indices.pkl'), 'wb') as f:
                pickle.dump(test_index, f)

            # 保存 train.csv
            train_df = pd.DataFrame({
                'protein_id': prot_train,
                'smile_id': smile_train,
                'binding': y_train
            })
            train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

            # 保存 test.csv
            test_df = pd.DataFrame({
                'protein_id': prot_test,
                'smile_id': smile_test,
                'binding': y_test
            })
            test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

            # 定义 LR 模型
            model = LinearRegression(fit_intercept=fit_intercept)

            # 训练
            model.fit(X_train, y_train)

            # 保存模型
            with open(os.path.join(fold_dir, 'model.pth'), 'wb') as f:
                pickle.dump(model, f)

            # 保存 fold 内 hyperparameters
            fold_hyperparams = model.get_params()
            with open(os.path.join(fold_dir, 'hyperparameters.pkl'), 'wb') as f:
                pickle.dump(fold_hyperparams, f)
            with open(os.path.join(fold_dir, 'hyperparameters.json'), 'w') as f:
                json.dump(fold_hyperparams, f, indent=4)

            # 测试集预测
            y_pred = model.predict(X_test)

            # 保存 test_pred.csv
            test_pred_df = pd.DataFrame({
                'protein_id': prot_test,
                'smile_id': smile_test,
                'binding': y_test_gt,
                'pred_binding': y_pred
            })
            test_pred_df.to_csv(os.path.join(fold_dir, 'test_pred.csv'), index=False)

            # 计算 loss
            mse = mean_squared_error(y_test_gt, y_pred)
            pearson_r = pearsonr(y_test_gt, y_pred)[0]

            # 保存 summary_records
            summary_records.append({
                'fold': fold + 1,
                'pearson_r': pearson_r,
                'mse': mse
            })

            print(f'Fold {fold+1} finished. Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}\n')

    elif model_type == 'MLP':
        # MLP 模型
        epochs = 50
        batch_size = 512
        learning_rate = 1e-4
        hyperparams = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dim': 512,
            'input_dim': 1792,
            'n_splits': n_splits,
            'seed': SEED
        }
        with open(os.path.join(output_dir, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(hyperparams, f)
        with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)

        if split_type == 'random':
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            for fold, (train_index, test_index) in enumerate(kf.split(X_tensor)):
                print(f'Fold {fold+1}/{n_splits}')
                X_train, X_test = X_tensor[train_index], X_tensor[test_index]
                y_train, y_test = y_tensor[train_index], y_tensor[test_index]
                prot_train = np.array(protein_id_list)[train_index]
                prot_test = np.array(protein_id_list)[test_index]
                smile_train = np.array(smile_id_list)[train_index]
                smile_test = np.array(smile_id_list)[test_index]
                y_test_gt = y[test_index]

                fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)

                with open(os.path.join(fold_dir, 'train_indices.pkl'), 'wb') as f:
                    pickle.dump(train_index, f)
                with open(os.path.join(fold_dir, 'test_indices.pkl'), 'wb') as f:
                    pickle.dump(test_index, f)

                # 保存 train.csv
                train_df = pd.DataFrame({
                    'protein_id': prot_train,
                    'smile_id': smile_train,
                    'binding': y_train.squeeze().numpy()
                })
                train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

                # 保存 test.csv
                test_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': y_test.squeeze().numpy()
                })
                test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

                # 初始化模型
                model = MLP()
                if use_pretrain:
                    pretrained_model_path = './results_MLP_dock/model.pth'
                    if os.path.exists(pretrained_model_path):
                        model.load_state_dict(torch.load(pretrained_model_path))
                        print(f'Loaded pretrained model from {pretrained_model_path}')
                    else:
                        print(f'Warning: Pretrained model not found at {pretrained_model_path}, training from scratch')
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()

                writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs', f'fold_{fold+1}'))

                # 训练
                model.train()
                for epoch in range(epochs):
                    running_loss = 0.0
                    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    avg_loss = running_loss / len(train_loader)
                    print(f'Fold {fold+1}, Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}')
                    writer.add_scalar('Loss/train', avg_loss, epoch)

                writer.close()

                # 保存模型
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))

                # 保存 fold 内 hyperparameters
                with open(os.path.join(fold_dir, 'hyperparameters.pkl'), 'wb') as f:
                    pickle.dump(hyperparams, f)
                with open(os.path.join(fold_dir, 'hyperparameters.json'), 'w') as f:
                    json.dump(hyperparams, f, indent=4)

                # 最终 test 评估
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test).squeeze().cpu().numpy()

                # 保存 test_pred.csv
                test_pred_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': y_test_gt,
                    'pred_binding': y_pred
                })
                test_pred_df.to_csv(os.path.join(fold_dir, 'test_pred.csv'), index=False)

                # 计算指标
                pearson_r = pearsonr(y_test_gt, y_pred)[0]
                mse = mean_squared_error(y_test_gt, y_pred)
                summary_records.append({
                    'fold': fold + 1,
                    'pearson_r': pearson_r,
                    'mse': mse
                })
                print(f'Fold {fold+1} finished. Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}\n')
        elif split_type == 'OR':
            dir_path = os.path.dirname(binding_csv_path)
            seq_csv_path = os.path.join(dir_path, '../data/seq_mix.csv')
            seq_df = pd.read_csv(seq_csv_path, header=None)
            seq_df = seq_df[[1, 2]]
            seq_df.columns = ['protein_id', 'sequence']
            seq_df['protein_id'] = seq_df['protein_id'].astype(str).str.strip()
            protein_ids = seq_df['protein_id'].unique()
            print(f'Total unique ORs: {len(protein_ids)}')
            
            kf_or = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for fold, (train_or_idx, test_or_idx) in enumerate(kf_or.split(protein_ids)):
                print(f'\n==== Fold {fold+1}/{n_splits} based on OR categories ====')
                
                train_or_set = set(protein_ids[train_or_idx])
                test_or_set = set(protein_ids[test_or_idx])

                # 读取 binding.csv，处理列
                df = pd.read_csv(binding_csv_path, header=None)
                df = df[[2, 1, 3]]
                df.columns = ['protein_id', 'smile_id', 'binding']
                df['protein_id'] = df['protein_id'].astype(str).str.strip()
                df['smile_id'] = df['smile_id'].astype(str).str.strip()

                # 基于OR划分sample
                train_mask = df['protein_id'].isin(train_or_set)
                test_mask = df['protein_id'].isin(test_or_set)

                df_train = df[train_mask].reset_index(drop=True)
                df_test = df[test_mask].reset_index(drop=True)

                X_protein_train, X_smile_train, y_train = [], [], []
                prot_train, smile_train = [], []

                for idx, row in df_train.iterrows():
                    p_id, s_id, binding_value = row['protein_id'], row['smile_id'], row['binding']
                    if (p_id in protein_dict) and (s_id in smile_dict):
                        X_protein_train.append(protein_dict[p_id])
                        X_smile_train.append(smile_dict[s_id])
                        y_train.append(binding_value)
                        prot_train.append(p_id)
                        smile_train.append(s_id)

                X_protein_test, X_smile_test, y_test_gt = [], [], []
                prot_test, smile_test = [], []

                for idx, row in df_test.iterrows():
                    p_id, s_id, binding_value = row['protein_id'], row['smile_id'], row['binding']
                    if (p_id in protein_dict) and (s_id in smile_dict):
                        X_protein_test.append(protein_dict[p_id])
                        X_smile_test.append(smile_dict[s_id])
                        y_test_gt.append(binding_value)
                        prot_test.append(p_id)
                        smile_test.append(s_id)

                print(f'→ Fold {fold+1}: Train samples = {len(y_train)}, Test samples = {len(y_test_gt)}')

                # 转tensor
                X_train_tensor = torch.tensor(np.concatenate([np.stack(X_protein_train), np.stack(X_smile_train)], axis=1), dtype=torch.float32)
                y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

                X_test_tensor = torch.tensor(np.concatenate([np.stack(X_protein_test), np.stack(X_smile_test)], axis=1), dtype=torch.float32)
                y_test_tensor = torch.tensor(np.array(y_test_gt), dtype=torch.float32).unsqueeze(1)

                fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)

                # 保存train.csv
                train_df = pd.DataFrame({
                    'protein_id': prot_train,
                    'smile_id': smile_train,
                    'binding': np.array(y_train)
                })
                train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

                # 保存test.csv
                test_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': np.array(y_test_gt)
                })
                test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

                # 初始化模型
                model = MLP()
                if use_pretrain:
                    pretrained_model_path = './results_MLP_dock/model.pth'
                    if os.path.exists(pretrained_model_path):
                        model.load_state_dict(torch.load(pretrained_model_path))
                        print(f'Loaded pretrained model from {pretrained_model_path}')
                    else:
                        print(f'Warning: Pretrained model not found at {pretrained_model_path}, training from scratch')
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()

                writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs', f'fold_{fold+1}'))

                # 训练
                model.train()
                for epoch in range(epochs):
                    running_loss = 0.0
                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    avg_loss = running_loss / len(train_loader)
                    print(f'Fold {fold+1}, Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}')
                    writer.add_scalar('Loss/train', avg_loss, epoch)

                writer.close()

                # 保存模型
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))

                # 保存fold内hyperparameters
                with open(os.path.join(fold_dir, 'hyperparameters.pkl'), 'wb') as f:
                    pickle.dump(hyperparams, f)
                with open(os.path.join(fold_dir, 'hyperparameters.json'), 'w') as f:
                    json.dump(hyperparams, f, indent=4)

                # 最终test评估
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor).squeeze().cpu().numpy()

                # 保存test_pred.csv
                test_pred_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': y_test_gt,
                    'pred_binding': y_pred
                })
                test_pred_df.to_csv(os.path.join(fold_dir, 'test_pred.csv'), index=False)

                # 计算指标
                pearson_r = pearsonr(y_test_gt, y_pred)[0]
                mse = mean_squared_error(y_test_gt, y_pred)
                summary_records.append({
                    'fold': fold + 1,
                    'pearson_r': pearson_r,
                    'mse': mse
                })
                print(f'Fold {fold+1} finished. Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}\n')
        elif split_type == 'VOC':
            dir_path = os.path.dirname(binding_csv_path)
            voc_csv_path = os.path.join(dir_path, '../data/voc_mix.csv')
            
            # 读取binding.csv，处理列
            df = pd.read_csv(binding_csv_path, header=None)
            df = df[[2, 1, 3]]
            df.columns = ['protein_id', 'smile_id', 'binding']
            df['protein_id'] = df['protein_id'].astype(str).str.strip()
            df['smile_id'] = df['smile_id'].astype(str).str.strip()

            # 读取VOC类别，只保留最后两列
            voc_df = pd.read_csv(voc_csv_path, header=None)
            voc_df = voc_df.iloc[:, -2:]
            voc_df.columns = ['smile_id', 'voc_class']
            voc_df['smile_id'] = voc_df['smile_id'].astype(str).str.strip()
            voc_df['voc_class'] = voc_df['voc_class'].astype(str).str.strip()

            # 把VOC类别merge到inter_dmel dataframe
            df = df.merge(voc_df, on='smile_id', how='left')

            # 取VOC类别unique
            voc_classes = df['voc_class'].dropna().unique()
            print(f'Total unique VOC classes: {len(voc_classes)}')

            kf_voc = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for fold, (train_voc_idx, test_voc_idx) in enumerate(kf_voc.split(voc_classes)):
                print(f'\n==== Fold {fold+1}/{n_splits} based on VOC categories ====')
                
                train_voc_set = set(voc_classes[train_voc_idx])
                test_voc_set = set(voc_classes[test_voc_idx])

                # 基于VOC类别划分sample
                train_mask = df['voc_class'].isin(train_voc_set)
                test_mask = df['voc_class'].isin(test_voc_set)

                df_train = df[train_mask].reset_index(drop=True)
                df_test = df[test_mask].reset_index(drop=True)

                X_protein_train, X_smile_train, y_train = [], [], []
                prot_train, smile_train = [], []

                for idx, row in df_train.iterrows():
                    p_id, s_id, binding_value = row['protein_id'], row['smile_id'], row['binding']
                    if (p_id in protein_dict) and (s_id in smile_dict):
                        X_protein_train.append(protein_dict[p_id])
                        X_smile_train.append(smile_dict[s_id])
                        y_train.append(binding_value)
                        prot_train.append(p_id)
                        smile_train.append(s_id)

                X_protein_test, X_smile_test, y_test_gt = [], [], []
                prot_test, smile_test = [], []

                for idx, row in df_test.iterrows():
                    p_id, s_id, binding_value = row['protein_id'], row['smile_id'], row['binding']
                    if (p_id in protein_dict) and (s_id in smile_dict):
                        X_protein_test.append(protein_dict[p_id])
                        X_smile_test.append(smile_dict[s_id])
                        y_test_gt.append(binding_value)
                        prot_test.append(p_id)
                        smile_test.append(s_id)

                print(f'→ Fold {fold+1}: Train samples = {len(y_train)}, Test samples = {len(y_test_gt)}')

                # 转tensor
                X_train_tensor = torch.tensor(np.concatenate([np.stack(X_protein_train), np.stack(X_smile_train)], axis=1), dtype=torch.float32)
                y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

                X_test_tensor = torch.tensor(np.concatenate([np.stack(X_protein_test), np.stack(X_smile_test)], axis=1), dtype=torch.float32)
                y_test_tensor = torch.tensor(np.array(y_test_gt), dtype=torch.float32).unsqueeze(1)

                fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)

                # 保存train.csv
                train_df = pd.DataFrame({
                    'protein_id': prot_train,
                    'smile_id': smile_train,
                    'binding': np.array(y_train)
                })
                train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

                # 保存test.csv
                test_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': np.array(y_test_gt)
                })
                test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

                # 初始化模型
                model = MLP()
                if use_pretrain:
                    pretrained_model_path = './results_MLP_dock/model.pth'
                    if os.path.exists(pretrained_model_path):
                        model.load_state_dict(torch.load(pretrained_model_path))
                        print(f'Loaded pretrained model from {pretrained_model_path}')
                    else:
                        print(f'Warning: Pretrained model not found at {pretrained_model_path}, training from scratch')
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()

                writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs', f'fold_{fold+1}'))

                # 训练
                model.train()
                for epoch in range(epochs):
                    running_loss = 0.0
                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    avg_loss = running_loss / len(train_loader)
                    print(f'Fold {fold+1}, Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}')
                    writer.add_scalar('Loss/train', avg_loss, epoch)

                writer.close()

                # 保存模型
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))

                # 保存fold内hyperparameters
                with open(os.path.join(fold_dir, 'hyperparameters.pkl'), 'wb') as f:
                    pickle.dump(hyperparams, f)
                with open(os.path.join(fold_dir, 'hyperparameters.json'), 'w') as f:
                    json.dump(hyperparams, f, indent=4)

                # 最终test评估
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor).squeeze().cpu().numpy()

                # 保存test_pred.csv
                test_pred_df = pd.DataFrame({
                    'protein_id': prot_test,
                    'smile_id': smile_test,
                    'binding': y_test_gt,
                    'pred_binding': y_pred
                })
                test_pred_df.to_csv(os.path.join(fold_dir, 'test_pred.csv'), index=False)

                # 计算指标
                pearson_r = pearsonr(y_test_gt, y_pred)[0]
                mse = mean_squared_error(y_test_gt, y_pred)
                summary_records.append({
                    'fold': fold + 1,
                    'pearson_r': pearson_r,
                    'mse': mse
                })
                print(f'Fold {fold+1} finished. Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}\n')

    elif model_type == 'RF':
        # 随机森林模型
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=SEED)
        hyperparams = model.get_params()
        with open(os.path.join(output_dir, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(hyperparams, f)
        with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f'Fold {fold+1}/5')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            prot_train = np.array(protein_id_list)[train_index]
            prot_test = np.array(protein_id_list)[test_index]
            smile_train = np.array(smile_id_list)[train_index]
            smile_test = np.array(smile_id_list)[test_index]
            y_test_gt = y[test_index]

            fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
            os.makedirs(fold_dir, exist_ok=True)

            with open(os.path.join(fold_dir, 'train_indices.pkl'), 'wb') as f:
                pickle.dump(train_index, f)
            with open(os.path.join(fold_dir, 'test_indices.pkl'), 'wb') as f:
                pickle.dump(test_index, f)

            # 保存 train.csv
            train_df = pd.DataFrame({
                'protein_id': prot_train,
                'smile_id': smile_train,
                'binding': y_train
            })
            train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

            # 保存 test.csv
            test_df = pd.DataFrame({
                'protein_id': prot_test,
                'smile_id': smile_test,
                'binding': y_test
            })
            test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

            # 训练模型
            model.fit(X_train, y_train)

            # 保存模型
            with open(os.path.join(fold_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)

            # 保存 fold 内 hyperparameters
            fold_hyperparams = model.get_params()
            with open(os.path.join(fold_dir, 'hyperparameters.pkl'), 'wb') as f:
                pickle.dump(fold_hyperparams, f)
            with open(os.path.join(fold_dir, 'hyperparameters.json'), 'w') as f:
                json.dump(fold_hyperparams, f, indent=4)

            # 最终 test 评估
            y_pred = model.predict(X_test)

            # 保存 test_pred.csv
            test_pred_df = pd.DataFrame({
                'protein_id': prot_test,
                'smile_id': smile_test,
                'binding': y_test_gt,
                'pred_binding': y_pred
            })
            test_pred_df.to_csv(os.path.join(fold_dir, 'test_pred.csv'), index=False)

            # 计算指标
            pearson_r = pearsonr(y_test_gt, y_pred)[0]
            mse = mean_squared_error(y_test_gt, y_pred)

            summary_records.append({
                'fold': fold + 1,
                'pearson_r': pearson_r,
                'mse': mse
            })

            print(f'Fold {fold+1} finished. Final TEST → Pearson r = {pearson_r:.4f}, MSE = {mse:.4f}\n')

    # 汇总 summary 结果
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    print('All folds finished. Summary saved to summary.csv')


def main():
    # 用户输入新数据路径
    binding_csv_path = input('请输入新数据的 inter.csv 文件路径: ')
    
    # 用户选择模型类型
    print('请选择要训练的模型类型:')
    print('1. 线性回归 (LR)')
    print('2. 多层感知机 (MLP)')
    print('3. 随机森林 (RF)')
    model_choice = input('请输入对应的数字 (1/2/3): ')
    
    # 用户选择数据划分方式
    print('请选择数据划分方式:')
    print('1. 随机划分')
    print('2. 以 OR 为划分依据')
    print('3. 以 VOC 为划分依据')
    split_choice = input('请输入对应的数字 (1/2/3): ')
    
    # 用户选择是否预训练
    print('是否使用预训练模型?')
    print('1. 是')
    print('2. 否')
    pretrain_choice = input('请输入对应的数字 (1/2): ')
    
    model_mapping = {
        '1': ('LR', 'results_LR'),
        '2': ('MLP', 'results_MLP'),
        '3': ('RF', 'results_RF')
    }
    
    split_mapping = {
        '1': 'random',
        '2': 'OR',
        '3': 'VOC'
    }
    
    pretrain_mapping = {
        '1': True,
        '2': False
    }
    
    if model_choice not in model_mapping:
        print('无效的模型选择，程序退出。')
        return
    
    if split_choice not in split_mapping:
        print('无效的数据划分选择，程序退出。')
        return
    
    if pretrain_choice not in pretrain_mapping:
        print('无效的预训练选择，程序退出。')
        return
    
    model_type, base_output_dir = model_mapping[model_choice]
    split_type = split_mapping[split_choice]
    pretrain_status = pretrain_mapping[pretrain_choice]
    dataset_type = os.path.basename(binding_csv_path).split('_')[1].split('.')[0]
    output_dir = f'{base_output_dir}_{split_type}_' + ('with_pretrain' if pretrain_status else 'without_pretrain') + f'_{dataset_type}'
    
    # 预处理数据
    X, y, protein_id_list, smile_id_list = preprocess_data(binding_csv_path)
    
    # 训练模型
    train_model(model_type, X, y, protein_id_list, smile_id_list, output_dir, split_type, use_pretrain=pretrain_status, binding_csv_path=binding_csv_path)


if __name__ == '__main__':
    main()