# coding=utf-8

import pandas as pd
import numpy as np
import argparse
import subprocess
import pickle
import torch
from Bio import SeqIO


def translate_contigs_into_proteins(input_fasta):
    # Prodigal translation
    prodigal_cmd = f'prodigal -i {input_fasta} -a test_protein.fa -f gff -p meta'
    print(f"Running prodigal ...")
    _ = subprocess.check_call(prodigal_cmd, shell=True)

    print("Encoding the phage genome sentence ...")

    # get the contig sequence file
    file1 = open("test_contig_sentence.csv", "w")
    dict_contig_proteins = {}

    for records in SeqIO.parse(f'test_protein.fa', format="fasta"):
        protein_name = str(records.id)
        protein_name_split = protein_name.split("_")
        contig_name = protein_name_split[0]
        for p in protein_name_split[1:-1]:
            contig_name = contig_name + "_" + p

        if contig_name not in dict_contig_proteins.keys():
            dict_contig_proteins[contig_name] = []
        dict_contig_proteins[contig_name].append(protein_name)

    for k, v in dict_contig_proteins.items():
        file1.write(k + ",")
        for v1 in v[:-1]:
            file1.write(v1 + ",")
        file1.write(v[-1] + "\n")
    file1.close()


def run_diamond_blatp_alignment(input_protein_fasta):
    bp_database = "./DataBase/BP_database"
    cc_database = "./DataBase/CC_database"
    mf_database = "./DataBase/MF_database"

    cmd1 = "diamond blastp -d " + bp_database + " -q " + input_protein_fasta + " -o test_against_BP_database.txt" + " -p 8 --sensitive "
    cmd2 = "diamond blastp -d " + cc_database + " -q " + input_protein_fasta + " -o test_against_CC_database.txt" + " -p 8 --sensitive "
    cmd3 = "diamond blastp -d " + mf_database + " -q " + input_protein_fasta + " -o test_against_MF_database.txt" + " -p 8 --sensitive "

    print(f"Running Diamond Blastp with BP database")
    _ = subprocess.check_call(cmd1, shell=True)

    print(f"Running Diamond Blastp with CC database")
    _ = subprocess.check_call(cmd2, shell=True)

    print(f"Running Diamond Blastp with MF database")
    _ = subprocess.check_call(cmd3, shell=True)


def read_test_protein():
    test_protein_names = []
    for records in SeqIO.parse("test_protein.fa", "fasta"):
        name = str(records.id)
        test_protein_names.append(name)
    return test_protein_names


def get_diamondscore(ont):
    print("Runing the DiamondBlastp method to get the preiction score ...")
    n_terms = {"BP": 126, "MF": 165, "CC": 23}
    number = n_terms[ont]

    file2 = open("./Protein_annotation/" + ont + "_known_proteins.csv")
    next(file2)

    dict_train_protein_go = {}
    for lines in file2:
        line = lines.strip().split(",")
        train_protein = line[0]
        label = line[2].split(";")
        final_label = set()
        for l in label:
            if l != "":
                final_label.add(l)
        dict_train_protein_go[train_protein] = final_label

    diamond_scores = {}
    input_diamond_file = "test_against_" + ont + "_database.txt"
    with open(input_diamond_file) as f:
        for line in f:
            it = line.strip().split("\t")
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[-1])


    test_protein_names = read_test_protein()

    blast_preds = []

    for test_protein in test_protein_names:
        annots = {}
        prot_id = test_protein

        if prot_id in diamond_scores:
            # sim_prots is the train proteins which are similar with test protein
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                # Get GO of the tran protein
                allgos |= dict_train_protein_go[p_id]
                total_score += score

            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in dict_train_protein_go[p_id]:
                        s += score
                sim[j] = s / total_score

            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)

    terms_file = "./Term_label/" + ont + "_term.pkl"
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    dict_go_id = {v: i for i, v in enumerate(terms)}

    all_test_blast_preds_score = []

    for pre in blast_preds:
        pre_score = [0] * number

        for go, score in pre.items():
            index = int(dict_go_id[go])
            pre_score[index] = score
        all_test_blast_preds_score.append(pre_score)

    test_results = {}
    test_results["diamondblastp_prediction"] = list(all_test_blast_preds_score)
    test_results["protein_name"] = test_protein_names

    with open("./" + ont + '_test_diamondblastp_results.pkl', 'wb') as handle:
        pickle.dump(test_results, handle)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""PhaGO is for phage protein annotation""")

    parser.add_argument('--contigs', help='FASTA file of contigs', default='test.fasta')

    parser.add_argument('--proteins', help='FASTA file of proteins', type=str, default=None)
    parser.add_argument('--sentences', help='The contigs sentence including the ordered proteins. Please named it as "test_contig_sentence.csv" and seperate each column with comma.', type=str,
                        default=None)

    inputs = parser.parse_args()

    contig_fasta = inputs.contigs
    protein_fasta = inputs.proteins
    contig_sentence = inputs.sentences

    if contig_fasta != None:
        translate_contigs_into_proteins(input_fasta=contig_fasta)
        run_diamond_blatp_alignment(input_protein_fasta="test_protein.fa")

    elif protein_fasta != None and contig_sentence != None:
        run_diamond_blatp_alignment(input_protein_fasta=protein_fasta)

    else:
        print("There is an error in your input!")
        exit(0)

    get_diamondscore(ont="BP")
    get_diamondscore(ont="CC")
    get_diamondscore(ont="MF")

