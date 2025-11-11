# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import torch
import random
import os
import pickle
import json
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

# 固定随机种子
SEED = 1234

def set_random_seed():
    """
    设置随机种子以保证结果可复现。
    """
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


def read_embeddings(protein_h5_path, smile_h5_path):
    """
    读取蛋白质和小分子的嵌入向量。
    
    :param protein_h5_path: 蛋白质嵌入向量的H5文件路径
    :param smile_h5_path: 小分子嵌入向量的H5文件路径
    :return: 蛋白质和小分子的嵌入向量字典
    """
    if not os.path.exists(protein_h5_path):
        # 获取基础文件名
        base_name = os.path.basename(protein_h5_path)
        
        # 检查可能的替代路径
        alt_path = os.path.join(os.path.dirname(os.path.dirname(protein_h5_path)), 'embeddings', base_name)
        
        # 检查当前目录下的embeddings文件夹
        current_dir_embeddings = os.path.join(os.path.dirname(protein_h5_path), 'embeddings')
        if os.path.exists(current_dir_embeddings):
            alt_path = os.path.join(current_dir_embeddings, base_name)
        
        if os.path.exists(alt_path):
            raise FileNotFoundError(f'Protein embedding file not found at {protein_h5_path}. Did you mean {alt_path}?')
        else:
            # 列出embeddings目录下所有文件帮助调试
            embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(protein_h5_path)), 'embeddings')
            available_files = []
            if os.path.exists(embeddings_dir):
                available_files = [f for f in os.listdir(embeddings_dir) if f.startswith('per_protein_embeddings')]
            
            raise FileNotFoundError(f'''Protein embedding file not found at {protein_h5_path}.
Available protein embedding files in {embeddings_dir}: {available_files}
Please check:
1. The file exists in the embeddings directory
2. The file name matches the expected pattern''')
    if not os.path.exists(smile_h5_path):
        # 获取基础文件名
        base_name = os.path.basename(smile_h5_path)
        
        # 检查可能的替代路径
        alt_path = os.path.join(os.path.dirname(os.path.dirname(smile_h5_path)), 'embeddings', base_name)
        
        # 检查当前目录下的embeddings文件夹
        current_dir_embeddings = os.path.join(os.path.dirname(smile_h5_path), 'embeddings')
        if os.path.exists(current_dir_embeddings):
            alt_path = os.path.join(current_dir_embeddings, base_name)
        
        if os.path.exists(alt_path):
            raise FileNotFoundError(f'Smile embedding file not found at {smile_h5_path}. Did you mean {alt_path}?')
        else:
            # 列出embeddings目录下所有文件帮助调试
            embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(smile_h5_path)), 'embeddings')
            available_files = []
            if os.path.exists(embeddings_dir):
                available_files = [f for f in os.listdir(embeddings_dir) if f.startswith('per_smile_embeddings')]
            
            raise FileNotFoundError(f'''Smile embedding file not found at {smile_h5_path}.
Available smile embedding files in {embeddings_dir}: {available_files}
Please check:
1. The file exists in the embeddings directory
2. The file name matches the expected pattern''')
    
    with h5py.File(protein_h5_path, 'r') as f_protein:
        protein_dict = {k: f_protein[k][:] for k in f_protein.keys()}
    with h5py.File(smile_h5_path, 'r') as f_smile:
        smile_dict = {k: f_smile[k][:] for k in f_smile.keys()}
    return protein_dict, smile_dict


def preprocess_data(binding_csv_path, protein_dict, smile_dict):
    """
    数据预处理函数。
    
    :param binding_csv_path: 绑定数据的CSV文件路径
    :param protein_dict: 蛋白质嵌入向量字典
    :param smile_dict: 小分子嵌入向量字典
    :return: 处理后的数据X, y和对应的ID列表
    """
    df = pd.read_csv(binding_csv_path, header=None)
    if 'dock' in binding_csv_path:
        df = df[[0, 1, 2]]
    else:
        df = df[[2, 1, 3]]
    df.columns = ['protein_id', 'smile_id', 'binding']
    df['protein_id'] = df['protein_id'].astype(str).str.strip()
    df['smile_id'] = df['smile_id'].astype(str).str.strip()

    X_protein = []
    X_smile = []
    y = []
    protein_id_list = []
    smile_id_list = []

    missing_count = 0
    for idx, row in df.iterrows():
        p_id, s_id, binding_value = row['protein_id'], row['smile_id'], row['binding']
        if (p_id in protein_dict) and (s_id in smile_dict):
            X_protein.append(protein_dict[p_id])
            X_smile.append(smile_dict[s_id])
            y.append(binding_value)
            protein_id_list.append(p_id)
            smile_id_list.append(s_id)
        else:
            missing_count += 1

    print(f"Total valid samples: {len(y)}, Missing pairs skipped: {missing_count}")
    
    X_protein = np.stack(X_protein)
    X_smile = np.stack(X_smile)
    y = np.array(y)
    X = np.concatenate([X_protein, X_smile], axis=1)
    return X, y, protein_id_list, smile_id_list


def save_hyperparameters(output_dir, hyperparams):
    """
    保存超参数到文件。
    
    :param output_dir: 输出目录
    :param hyperparams: 超参数字典
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hyperparameters.pkl'), 'wb') as f:
        pickle.dump(hyperparams, f)
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)


def setup_fold_directory(output_dir, fold):
    """
    设置每个折的目录结构。
    
    :param output_dir: 输出目录
    :param fold: 当前折数
    :return: 当前折的目录路径
    """
    fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)
    return fold_dir


def save_data_to_csv(fold_dir, X, y, protein_id_list, smile_id_list, indices, data_type):
    """
    保存数据到CSV文件。
    
    :param fold_dir: 当前折的目录路径
    :param X: 特征数据
    :param y: 标签数据
    :param protein_id_list: 蛋白质ID列表
    :param smile_id_list: 小分子ID列表
    :param indices: 数据索引
    :param data_type: 数据类型，如'train'或'test'
    """
    prot = np.array(protein_id_list)[indices]
    smile = np.array(smile_id_list)[indices]
    data_df = pd.DataFrame({
        'protein_id': prot,
        'smile_id': smile,
        'binding': y[indices]
    })
    data_df.to_csv(os.path.join(fold_dir, f'{data_type}.csv'), index=False)


def setup_tensorboard_writer(output_dir, fold):
    """
    设置TensorBoard写入器。
    
    :param output_dir: 输出目录
    :param fold: 当前折数
    :return: TensorBoard写入器对象
    """
    return SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs', f'fold_{fold+1}'))

class MLP(torch.nn.Module):
    """
    定义多层感知机模型。
    """
    def __init__(self, input_dim=1792, hidden_dim=512):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_mlp(model, X_train, y_train, epochs, batch_size, learning_rate, writer, fold):
    """
    训练MLP模型。
    
    :param model: MLP模型对象
    :param X_train: 训练特征数据
    :param y_train: 训练标签数据
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :param writer: TensorBoard写入器对象
    :param fold: 当前折数
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
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