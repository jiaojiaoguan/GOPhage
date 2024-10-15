#coding=utf-8
import argparse
import os
import shutil
import torch

def preparing_context_protein_embedding(plm_model_name,contig_sentence):

    # intergating the proteins in the same contigs
    print("intergating the proteins embedding in the same contigs ...")
    if plm_model_name == "esm2-12":
        protein_embedding_path="esm12_per_residual_embedding/"

        sequence_embedding_path = "sequence_embedding_esm12/"
        if os.path.exists(sequence_embedding_path):
            shutil.rmtree(sequence_embedding_path)

        os.makedirs(sequence_embedding_path)

    elif plm_model_name == "esm2-33":
        protein_embedding_path = "esm33_per_residual_embedding/"
        sequence_embedding_path = "sequence_embedding_esm33/"

        if os.path.exists(sequence_embedding_path):
            shutil.rmtree(sequence_embedding_path)

        os.makedirs(sequence_embedding_path)

    else:
        print("Error: the model name is not correct!")
        exit(0)



    # read the contig sentence
    file1 = open(contig_sentence)
    for lines in file1:
        line = lines.strip().split(",")
        sequence_name=line[0]
        proteins = []
        for p in line[1:]:
            if p!="":
                proteins.append(p)

        all_embedding = []
        if len(proteins)==1: # if the proteins's length is 1, then we do not need to concatnate them
            continue
        else:
            for p in proteins:
                embedding = torch.load(
                    protein_embedding_path + p + "_embedding.pkl",
                    map_location=torch.device("cpu"))

                all_embedding.append(embedding["data"])

            sequence_embedding = torch.cat(all_embedding, dim=0)

            file_path = sequence_embedding_path + sequence_name + "_embedding.pt"

            torch.save({"data": sequence_embedding}, file_path)

def get_protein_names(contig_sentence):
    print("Preparing the protein names ...")


    protein_name_path="protein_names/"
    if os.path.exists(protein_name_path):
        shutil.rmtree(protein_name_path)

    os.makedirs(protein_name_path)

    file1 = open(contig_sentence)
    for lines in file1:
        line = lines.strip().split(",")
        sequence_name = line[0]
        proteins = []
        for p in line[1:]:
            if p != "":
                proteins.append(p)


        file_path_protein_name=protein_name_path+sequence_name+"_names.pt"
        torch.save({"proteins_name":proteins},file_path_protein_name)

    print("get sequence protein names successfully!")

def get_sequence_location(model_name,contig_sentence):
    print("preparing the location of the embedding and protein names ...")

    if model_name=="esm2-12":
        sequence_embedding_path="sequence_embedding_esm12/"
        protein_embedding_path = "esm12_per_residual_embedding/"
        file2 = open("results/test_location_esm12.csv", "w")

    elif model_name == "esm2-33":
        sequence_embedding_path = "sequence_embedding_esm33/"
        protein_embedding_path = "esm33_per_residual_embedding/"
        file2 = open("results/test_location_esm33.csv", "w")
    else:
        print("Error: the model name is not correct!")
        exit(0)

    file1 = open(contig_sentence)



    for lines in file1:
        line = lines.strip().split(",")
        sequence_name = line[0]
        proteins=[]
        for p in line[1:]:
            if p!="":
                proteins.append(p)
        length=len(proteins)


        file_path_embedding = sequence_embedding_path +sequence_name + "_embedding.pt"
        file_path_protein_name = "protein_names/"+ sequence_name + "_names.pt"

        if length == 1:

            protein_name = proteins[0]

            file_path_embedding = protein_embedding_path  + protein_name + "_embedding.pkl"

        file2.write(
            file_path_embedding + "," + file_path_protein_name +  "\n")

    file2.close()
