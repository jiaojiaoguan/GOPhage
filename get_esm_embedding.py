#coding=utf-8
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

import os


def get_data(fasta_file):
    protein_sequence = []
    all_id=[]
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        all_id.append(str(record.id))
        protein_sequence.append(sequence)

    return protein_sequence,all_id


class VirDataset(Dataset):
    def __init__(self, fasta_file):

        self.texts, self.id= get_data(fasta_file)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index],self.id[index]


def get_loader(fasta_file,
        batch_size=64,
        num_workers=2):
    dataset = VirDataset(fasta_file)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers)
    return loader

def embedding_proteins_ESM2(plm_model_name):
    print("Preparing the data for PhaGO model ...")
    print("Embedding the protein sequence using ESM2 "+ plm_model_name +" ...")
    fasta_file = "test_protein.fa"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_loader(fasta_file,
                             batch_size=1,
                             num_workers=1)

    #esm2-12 model

    if plm_model_name=="esm2-12":

        tokenizer = AutoTokenizer.from_pretrained("Esm_model/facebook/esm2_t12_35M_UR50D/")
        model = EsmModel.from_pretrained("Esm_model/facebook/esm2_t12_35M_UR50D/")
        result_path = "esm12_per_residual_embedding/"
        os.makedirs(result_path)


    elif plm_model_name=="esm2-33":

        # esm2-33 model
        tokenizer = AutoTokenizer.from_pretrained("Esm_model/facebook/esm2_t33_650M_UR50D")
        model = EsmModel.from_pretrained("Esm_model/facebook/esm2_t33_650M_UR50D")
        result_path = "esm33_per_residual_embedding/"
        os.makedirs(result_path)

    else:
        print("There is an error in your input model name!")
        exit(0)

    if torch.cuda.device_count() > 1:
        print(f'Use {torch.cuda.device_count()} GPUs!\n')
        model = nn.DataParallel(model)

    model.to(device)


    with torch.no_grad():
        for texts, proteins_names in test_loader:
            inputs = tokenizer(texts, return_tensors="pt",
                               padding="max_length", max_length=1024, truncation=True)
            name = proteins_names[0]
            inputs = inputs.to(device)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states.to("cpu")

            file_path_embedding = result_path + name + "_embedding.pkl"

            torch.save({'data': last_hidden_states}, file_path_embedding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Run esm model to get the protein embedding""")

    parser.add_argument('--plm', help='name of PLM model (esm2-12 or esm2-33)', type=str, default='esm2-12')

    inputs = parser.parse_args()
    plm_model = inputs.plm

    embedding_proteins_ESM2(plm_model_name=plm_model)
