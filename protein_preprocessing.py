import os
import torch
import torch.nn as nn
import pandas as pd
import h5py
import time
from transformers import T5EncoderModel, T5Tokenizer

input_path = '/home/gaocx/supplement_data_new/data/Aaeg.csv'

per_residue = True
per_residue_path = "/home/gaocx/supplement_data_new/embeddings/per_residue_embeddings_Aaeg.h5"

per_protein = True
per_protein_path = "/home/gaocx/supplement_data_new/embeddings/per_protein_embeddings_Aaeg.h5"

def get_T5_model():
    print('get model start')

    # 信任该类以便反序列化
    from transformers.models.t5.modeling_t5 import T5EncoderModel
    torch.serialization.add_safe_globals([T5EncoderModel])

    # 直接加载整个模型对象（你给的 .pth 就是完整模型）
    model = T5EncoderModel.from_pretrained('./protT5_local', local_files_only=True)

    # 加载 tokenizer（可以是你下载的 ./protT5_local）
    tokenizer = T5Tokenizer.from_pretrained('./protT5_local', local_files_only=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    print('Using device:', device)
    print('get model done')

    return model, tokenizer


def read_seqs(seq_path):
    df = pd.read_csv(seq_path, header=None, usecols=[1, 2])
    df = df.dropna().reset_index(drop=True)  # 去掉空行
    df[1] = df[1].astype(str).str.strip()    # 去掉名称前后空格
    df[2] = df[2].astype(str).str.strip()

    seq_dict = {}
    for name, seq in zip(df[1], df[2]):
        if not seq or seq.lower() == 'nan':  # 跳过空序列
            continue
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_dict[name] = seq

    print(f"有效序列数量: {len(seq_dict)} / 总行数: {len(df)}")
    return seq_dict

def get_embeddings(model, tokenizer, per_residue, per_protein, max_residues=3000, max_seq_len=1000, max_batch=7):
    results = {"residue_embs": {}, "protein_embs": {}}
    batch = []

    s_dic = read_seqs(input_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for idx, (change_name, seq) in enumerate(s_dic.items(), 1):
        seq_len = len(seq)
        seq_spaced = ' '.join(list(seq))
        batch.append((change_name, seq_spaced, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or idx == len(s_dic) or seq_len > max_seq_len:
            seq_ids, seqs, seq_lens = zip(*batch)
            batch.clear()

            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            with torch.no_grad():
                embedding_repr = model(input_ids, attention_mask=attention_mask)

            for batch_idx, identifier in enumerate(seq_ids):
                v_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :v_len]

                if per_residue:
                    results["residue_embs"][identifier] = emb.cpu().numpy().squeeze()
                if per_protein:
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.cpu().numpy().squeeze()

    return results

def save_embeddings(emb_dict, out_path):
    with h5py.File(out_path, "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

def main():
    os.makedirs('./output', exist_ok=True)
    model, tokenizer = get_T5_model()
    results = get_embeddings(model, tokenizer, per_residue=True, per_protein=True)

    if per_residue:
        save_embeddings(results["residue_embs"], per_residue_path)
    if per_protein:
        save_embeddings(results["protein_embs"], per_protein_path)

if __name__ == "__main__":
    main()
