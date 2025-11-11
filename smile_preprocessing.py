import os
import torch
import pandas as pd
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
import h5py


model_path = "./Smodel" 
weights_path = "./Smodel/Smodel_1M.pth"
csv_path = "/home/gaocx/supplement_data_new/data/voc_14499.csv"
per_atom_path = "/home/gaocx/supplement_data_new/embeddings/per_atom_embeddings_14499.h5"
per_smile_path = "/home/gaocx/supplement_data_new/embeddings/per_smile_embeddings_14499.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from transformers import AutoModel, AutoTokenizer

def get_model():
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return model, tokenizer



def read_smiles(seq_path):
    df = pd.read_csv(seq_path, header=None, usecols=[1, 2])
    change_names = df[1].tolist()
    smiles = df[2].tolist()
    return dict(zip(change_names, smiles))


def get_embeddings(model, tokenizer, per_atom=True, per_smile=True):
    data = read_smiles(csv_path)
    results = {"atom_embs": dict(), "smile_embs": dict()}

    for name, smile in tqdm(data.items()):
        inputs = tokenizer(smile, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)

        if per_atom:
            atom_vec = hidden_states.squeeze(0).cpu().numpy()  # shape: (seq_len, hidden_dim)
            results["atom_embs"][name] = atom_vec

        if per_smile:
            smile_vec = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()  # shape: (hidden_dim,)
            results["smile_embs"][name] = smile_vec

    return results


def save_embeddings(emb_dict, out_path):
    with h5py.File(out_path, "w") as hf:
        for name, emb in emb_dict.items():
            hf.create_dataset(str(name), data=emb)


def main():
    model, tokenizer = get_model()
    results = get_embeddings(model, tokenizer, per_atom=True, per_smile=True)

    if results.get("atom_embs"):
        save_embeddings(results["atom_embs"], per_atom_path)
    if results.get("smile_embs"):
        save_embeddings(results["smile_embs"], per_smile_path)


if __name__ == "__main__":
    main()
