# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import os
import pickle
from utils import read_embeddings, MLP


def load_model(model_path):
    """
    加载训练好的模型
    
    :param model_path: 模型文件路径
    :return: 加载好的模型
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        # 尝试作为PyTorch模型加载
        model = MLP()
        # 添加weights_only=True以解决安全警告
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print(f"成功加载PyTorch模型: {model_path}")
        return model, 'pytorch'
    except:
        # 尝试作为sklearn模型加载
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"成功加载sklearn模型: {model_path}")
            return model, 'sklearn'
        except Exception as e:
            raise ValueError(f"无法加载模型文件，请检查文件格式: {e}")


def predict(model, X, model_type='pytorch'):
    """
    使用模型进行预测
    
    :param model: 加载好的模型
    :param X: 预处理后的特征数据
    :param model_type: 模型类型，'pytorch'或'sklearn'
    :return: 预测结果
    """
    if model_type == 'pytorch':
        # PyTorch模型预测
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X
            y_pred = model(X_tensor)
            return y_pred.numpy().squeeze()
    else:
        # sklearn模型预测
        return model.predict(X)


def get_user_input(prompt, default=None):
    """
    获取用户输入，支持默认值
    
    :param prompt: 提示信息
    :param default: 默认值
    :return: 用户输入的值或默认值
    """
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        user_input = input(prompt).strip()
        if user_input == '' and default is not None:
            return default
        elif user_input != '':
            return user_input
        else:
            print("输入不能为空，请重新输入。")


def preprocess_prediction_data(binding_csv_path, protein_dict, smile_dict):
    """
    专门用于预测数据的预处理函数
    
    :param binding_csv_path: 预测数据CSV文件路径（包含蛋白质ID和小分子ID）
    :param protein_dict: 蛋白质嵌入向量字典
    :param smile_dict: 小分子嵌入向量字典
    :return: 处理后的数据X，蛋白质ID列表，小分子ID列表
    """
    # 读取CSV文件
    df = pd.read_csv(binding_csv_path, header=None)
    num_columns = df.shape[1]
    
    # 检查列数并相应地处理
    if num_columns == 3:
        # 标准预测数据格式：[索引, 小分子ID, 蛋白质ID]
        df = df[[2, 1]]  # 选择蛋白质ID和小分子ID
        df.columns = ['protein_id', 'smile_id']
    elif num_columns >= 2:
        # 处理其他可能的格式，优先选择最后两列作为蛋白质ID和小分子ID
        print(f"注意：CSV文件有{num_columns}列，默认使用最后两列作为蛋白质ID和小分子ID")
        df = df.iloc[:, -2:]
        df.columns = ['protein_id', 'smile_id']
    else:
        raise ValueError(f"CSV文件格式不正确，至少需要2列数据（蛋白质ID和小分子ID），但实际有{num_columns}列")
    
    # 处理ID列
    df['protein_id'] = df['protein_id'].astype(str).str.strip()
    df['smile_id'] = df['smile_id'].astype(str).str.strip()

    # 准备数据
    X_protein = []
    X_smile = []
    protein_id_list = []
    smile_id_list = []

    missing_count = 0
    for idx, row in df.iterrows():
        p_id, s_id = row['protein_id'], row['smile_id']
        if (p_id in protein_dict) and (s_id in smile_dict):
            X_protein.append(protein_dict[p_id])
            X_smile.append(smile_dict[s_id])
            protein_id_list.append(p_id)
            smile_id_list.append(s_id)
        else:
            missing_count += 1

    print(f"有效的预测样本数: {len(protein_id_list)}, 跳过的缺失对数量: {missing_count}")
    
    # 检查是否有有效样本
    if len(X_protein) == 0:
        raise ValueError("没有找到有效的预测样本。可能的原因：\n" \
                         "1. 蛋白质或小分子ID在嵌入向量文件中不存在\n" \
                         "2. 输入文件中的ID格式与嵌入向量文件中的ID格式不匹配\n" \
                         "3. 嵌入向量文件可能损坏或不完整")
    
    # 转换为numpy数组
    X_protein = np.stack(X_protein)
    X_smile = np.stack(X_smile)
    X = np.concatenate([X_protein, X_smile], axis=1)
    
    return X, protein_id_list, smile_id_list


def validate_output_path(output_path, output_suffix=None):
    """
    验证输出路径，如果是目录则添加默认文件名
    
    :param output_path: 用户提供的输出路径
    :param output_suffix: 输出文件名的后缀部分（用于构建inter_xxx格式）
    :return: 有效的输出文件路径
    """
    # 检查路径是否为目录
    if os.path.isdir(output_path):
        # 如果是目录，添加默认文件名
        if output_suffix:
            default_filename = f"inter_{output_suffix}.csv"
        else:
            default_filename = "inter_predictions.csv"
        output_path = os.path.join(output_path, default_filename)
        print(f"提供的路径是目录，将使用文件名: {output_path}")
    
    # 检查文件名是否以inter_开头，如果不是且有后缀，则重命名
    elif output_suffix and not os.path.basename(output_path).startswith('inter_'):
        dir_name = os.path.dirname(output_path)
        new_filename = f"inter_{output_suffix}.csv"
        output_path = os.path.join(dir_name, new_filename)
        print(f"自动重命名为inter_xxx格式: {output_path}")
    
    # 检查文件扩展名
    if not output_path.endswith('.csv'):
        output_path += '.csv'
        print(f"添加CSV扩展名: {output_path}")
    
    return output_path


def main():
    print("蛋白质-小分子结合预测工具")
    print("="*50)
    print("本工具用于预测蛋白质和小分子之间的结合值")
    print("请输入预测所需的文件路径")
    print("="*50)
    
    try:
        # 获取用户输入
        model_path = get_user_input("请输入训练好的模型文件路径")
        prediction_csv = get_user_input("请输入预测数据CSV文件路径（包含蛋白质ID和小分子ID）")
        protein_embeddings = get_user_input("请输入蛋白质嵌入向量H5文件路径")
        smile_embeddings = get_user_input("请输入小分子嵌入向量H5文件路径")
        # 获取输出后缀名（用于inter_xxx格式）
        output_suffix = get_user_input("请输入输出文件名的后缀部分（将生成inter_xxx.csv格式文件）", "predictions")
        # 只获取输出目录，文件名会根据output_suffix自动生成
        output_dir = get_user_input("请输入预测结果输出目录", ".")
        
        # 设置随机种子
        np.random.seed(1234)
        
        print("\n开始执行预测...")
        
        # 1. 加载模型
        print(f"1. 正在加载模型: {model_path}...")
        model, model_type = load_model(model_path)
        
        # 2. 读取嵌入向量
        print(f"2. 正在读取蛋白质嵌入向量: {protein_embeddings}...")
        print(f"   正在读取小分子嵌入向量: {smile_embeddings}...")
        protein_dict, smile_dict = read_embeddings(protein_embeddings, smile_embeddings)
        
        # 3. 预处理预测数据
        print(f"3. 正在预处理预测数据: {prediction_csv}...")
        X, protein_id_list, smile_id_list = preprocess_prediction_data(prediction_csv, protein_dict, smile_dict)
        
        # 4. 进行预测
        print(f"4. 正在进行预测...")
        y_pred = predict(model, X, model_type)
        
        # 5. 保存预测结果
        # 验证输出路径并确保使用inter_xxx格式
        output_path = validate_output_path(output_dir, output_suffix)
        print(f"5. 正在保存预测结果到: {output_path}...")
        
        # 确保输出目录存在
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            print(f"  创建输出目录: {output_dir_path}")
        
        # 创建结果DataFrame - 只包含预测相关信息
        result_df = pd.DataFrame({
            'protein_id': protein_id_list,
            'smile_id': smile_id_list,
            'predicted_binding': y_pred
        })
        
        # 保存结果
        result_df.to_csv(output_path, index=False)
        
        print("\n预测完成！")
        print(f"预测结果已保存到: {output_path}")
        print(f"共预测了 {len(result_df)} 对蛋白质-小分子的结合值")
    except Exception as e:
        print(f"\n预测过程中发生错误: {str(e)}")
        print("请检查您提供的文件路径和文件格式是否正确，然后重试。")


if __name__ == '__main__':
    main()