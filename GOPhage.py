#coding=utf-8

from dataloder import get_combine_loader_cut_annot
import pandas as pd
import torch as th
import numpy as np
from torch import nn
import pickle
import torch
from model import PhaGO_model
import argparse
import subprocess
from Bio import SeqIO
from get_esm_embedding import embedding_proteins_ESM2
from prepare_gophage_input import preparing_context_protein_embedding, get_protein_names, get_sequence_location
import os
import time

def translate_contigs_into_proteins(input_fasta):
    # Prodigal translation
    output_protein= input_fasta.split(".")[0]
    output_protein= output_protein+".fa"

    prodigal_cmd = f'prodigal -i {input_fasta} -a {output_protein} -f gff -p meta'
    print(f"Running prodigal ...")
    _ = subprocess.check_call(prodigal_cmd, shell=True)

    print("Encoding the phage genome sentence ...")

    # get the contig sequence file

    contig_sentence = "results/test_contig_sentence.csv"
    file1 = open(contig_sentence, "w")
    dict_contig_proteins = {}

    for records in SeqIO.parse(output_protein, format="fasta"):
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

    return output_protein, contig_sentence


def check_number_protein(contig_sentence,ont):

    dict_ontology_length = {"CC": 55, "BP": 17, "MF": 59}
    max_length = dict_ontology_length[ont]
    fiel1 = open(contig_sentence)
    new_contig_sentence_name = "results/new_" + ont + "_" + contig_sentence

    file2 = open(new_contig_sentence_name, "w")
    for lines in fiel1:
        line = lines.strip().split(",")
        contig_name = line[0]
        proteins_all = line[1:]

        final_proteins = []
        for p in proteins_all:
            if p != "":
                final_proteins.append(p)

        protein_number = len(final_proteins)

        if protein_number > max_length:
            subsentences = [proteins_all[i:i + max_length] for i in range(0, len(proteins_all), max_length)]

            for index in range(len(subsentences)):
                file2.write(contig_name + "_" + str(index) + ",")
                for j in subsentences[index]:
                    file2.write(j + ",")
                file2.write("\n")
        else:
            file2.write(lines)

    file2.close()

    return new_contig_sentence_name

def run_diamond_blatp_alignment(input_protein_fasta,ont):
    database = f"./DataBase/{ont}_database"

    cmd1 = "diamond blastp -d " + database + " -q " + input_protein_fasta + " -o results/test_against_"+ont+"_database.txt" + " -p 8 --sensitive "


    print(f"Running Diamond Blastp with database")
    _ = subprocess.check_call(cmd1, shell=True)


def read_test_protein(protein_fasta):
    test_protein_names = []
    for records in SeqIO.parse(protein_fasta, "fasta"):
        name = str(records.id)
        test_protein_names.append(name)
    return test_protein_names


def get_diamondscore(ont,protein_fasta):
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
    input_diamond_file = f"results/test_against_{ont}_database.txt"
    with open(input_diamond_file) as f:
        for line in f:
            it = line.strip().split("\t")
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[-1])


    test_protein_names = read_test_protein(protein_fasta)

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

    with open("results/" + ont + '_test_diamondblastp_results.pkl', 'wb') as handle:
        pickle.dump(test_results, handle)


def run_phaGO_model(plm_model_name,ont,batch_size):

    if plm_model_name== "esm2-12":
        nhead = 12
        d_model = 480

        if ont=="CC":
            dim_feedforward=128
        else:
            dim_feedforward = 480

        model_name = "./PhaGO_model/"+ont+"_PhaGO_base_model.th"
        out_put_file= "results/" + ont + '_phago_base_results.pkl'
        data_file_input = f"results/test_location_esm12.csv"

    elif plm_model_name=="esm2-33":
        nhead = 16
        d_model = 1280
        if ont=="CC":
            dim_feedforward=320
        else:
            dim_feedforward = 1280

        model_name = "./PhaGO_model/" + ont + "_PhaGO_large_model.th"
        out_put_file = "results//" + ont + '_phago_large_results.pkl'
        data_file_input = "results/test_location_esm33.csv"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_terms = {"BP":126,"MF":165,"CC":23}
    dict_ontology_length = {"CC": 55, "BP": 17, "MF": 59}

    length = dict_ontology_length[ont]


    test_loader = get_combine_loader_cut_annot(
        data_file=data_file_input,
        batch_size=batch_size,
        num_workers=2,
        ont=ont,
        dict_ontology_length=dict_ontology_length,
        plm_model = plm_model_name

    )

    net = PhaGO_model(nhead=nhead,
                       dim_feedforward=dim_feedforward,
                       num_layers=1,
                       dropout=0.5,
                       num_labels=n_terms[ont],
                       d_model=d_model,
                       vocab_size=length)


    net.load_state_dict(torch.load(model_name, map_location=device))



    if torch.cuda.device_count() > 1:
        print(f'Use {torch.cuda.device_count()} GPUs!\n')
        net = nn.DataParallel(net)

    net.to(device)

    net.eval()

    test_preds=[]
    test_all_proteins=[]
    with th.no_grad():
        for batch_idx, (embedding,test_proteins) in enumerate(test_loader):
            test_proteins = np.array(test_proteins)
            test_proteins = np.transpose(test_proteins)

            test_extract_protein = [protein for index, protein in enumerate(test_proteins.flatten()) if
                     protein != 'none']

            for i in test_extract_protein:
                test_all_proteins.append(i)

            test_batch_features = embedding.to(device)
            test_logits = net(test_batch_features)

            for i, row in enumerate(test_proteins):
                for protein in row:
                    if protein != 'none':
                        index = list(row).index(protein)
                        test_preds.append(test_logits[i][index].tolist())

        test_results = {}
        test_results["prediction"] = test_preds
        test_results["protein_name"] = test_all_proteins


        with open(out_put_file, 'wb') as handle:
            pickle.dump(test_results, handle)

def combine_diamondblasp_phaGO(plm_model_name,ont):

    if plm_model_name=="esm2-12":
        dict_onyology_alpha = { "BP": 1.0, "CC": 0.83,"MF": 0.91}
        output_results = "results//"+ont+ "_phago_base_plus_predictions.pkl"
        phago_prediction_results = "results//" + ont + '_phago_base_results.pkl'

    elif plm_model_name=="esm2-33":
        dict_onyology_alpha = {"BP": 0.9, "CC": 0.62, "MF": 0.82}
        output_results = "results//"+ont+ "_phago_large_plus_predictions.pkl"
        phago_prediction_results = "results/" + ont + '_phago_large_results.pkl'
    else:
        print("Error, please the correct plm model name!")
        exit(0)

    alpha_parameter = dict_onyology_alpha[ont]

    # diamond_score_prediction_results.
    diamond_blastp_result_file = "results//" + ont + '_test_diamondblastp_results.pkl'
    test_df = pd.read_pickle(diamond_blastp_result_file)
    diamond_blastp_preds = test_df["diamondblastp_prediction"]
    diamond_protein_name = test_df["protein_name"]

    #load the phago prediction results.

    phago_df = pd.read_pickle(phago_prediction_results)
    phago_proteins = phago_df["protein_name"]
    phago_prediction = phago_df["prediction"]

    context_dl_protein_prediction = {}

    for i in range(len(phago_proteins)):
        context_dl_protein_prediction[phago_proteins[i]] = phago_prediction[i]

    diamondscore_protein_preidction = {}

    for index in range(len(diamond_blastp_preds)):
        dia_pred = diamond_blastp_preds[index]
        pro = diamond_protein_name[index]

        prediction_diamondscore = list(dia_pred)
        preds = [int(i) for i in prediction_diamondscore]
        all_zeros = all(x == 0 for x in preds)
        if all_zeros:
            continue
        else:
            diamondscore_protein_preidction[pro] = prediction_diamondscore

    all_phagoplus = []

    proteins = []

    for k, v in context_dl_protein_prediction.items():
        proteins.append(k)

        if k in diamondscore_protein_preidction.keys():
            diamondscore = diamondscore_protein_preidction[k]
            diamondscore = [i * alpha_parameter for i in diamondscore]
            phago_score = [i * (1 - alpha_parameter) for i in v]
            phagoplus = [diamondscore[i] + phago_score[i] for i in range(len(diamondscore))]
        else:
            phagoplus = v

        all_phagoplus.append(phagoplus)

    data = {}

    data["proteins"] = proteins

    data["preds"] = all_phagoplus

    df = pd.DataFrame(data)

    df.to_pickle(output_results, protocol=4)

def output_the_prediction_results(plm_model_name, ont):
    #load the phago plus results
    # get cutoff for each go term

    if plm_model_name=="esm2-12":

        phagoplus_prediction_results = "results//" + ont + "_phago_base_plus_predictions.pkl"
        file1 = open("results/"+ont + "_GOPhage_base_plus_prediction_labels.csv", "w")
        file2=open("./PhaGO_model/esm12_"+ont+"_label_threshold.csv")
        next(file2)

    elif plm_model_name=="esm2-33":

        phagoplus_prediction_results = "results//" + ont + "_phago_large_plus_predictions.pkl"
        file1 = open("results/"+ont + "_GOPhage_large_plus_prediction_labels.csv", "w")
        file2 = open("./PhaGO_model/esm33_" + ont + "_label_threshold.csv")
        next(file2)
    else:
        print("Error, please the correct plm model name!")
        exit(0)




    test_df = pd.read_pickle(phagoplus_prediction_results)
    phago_plus_preds = test_df["preds"]
    phago_plus_protein = test_df["proteins"]


    terms_file = "./Term_label/" + ont + '_term.pkl'

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {i: v for i, v in enumerate(terms)}

    dict_label_threshold={}
    for lines in file2:
        line = lines.strip().split(",")
        dict_label_threshold[line[0]]=float(line[1])




    file1.write("Proteins,GO Term,Scores\n")

    for j in range(len(phago_plus_protein)):
        p= phago_plus_protein[j]
        for indice, score in enumerate( phago_plus_preds[j]):
            go_term = terms_dict[indice]
            cutoff= dict_label_threshold[go_term]
            if score>cutoff:
                file1.write(p+","+go_term+","+str(score)+"\n")
    file1.close()



if __name__ == '__main__':

    start_time= time.time()

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="""GOphage is a tool for phage protein annotation""")

    # user input
    parser.add_argument('--contigs', help='DNA FASTA file of contigs, you can only input contigs and ignore the --proteins and --sentences', default=None)

    parser.add_argument('--proteins', help='FASTA file of proteins, if you input the proteins, you also need to tell the proteins sentences', type=str, default=None)
    parser.add_argument('--sentences',
                        help='The contigs sentence including the ordered proteins. Please seperate each column with comma.',
                        type=str,
                        default=None)


    parser.add_argument('--plm', help='name of PLM model (esm2-12 or esm2-33)', type=str, default='esm2-12')
    parser.add_argument('--ont', help='The ontology including BP, CC and MF', type=str, default='CC')
    parser.add_argument('--batch_size', help="batch size", type=int, default=8)


    inputs = parser.parse_args()

    contig_fasta = inputs.contigs
    protein_fasta = inputs.proteins
    contig_sentence = inputs.sentences

    plm_model = inputs.plm
    ont = inputs.ont
    batch_size = inputs.batch_size

    if not os.path.exists("results/"):
        os.makedirs("results/")

    #Preprocess the contigs and generate the contig sentences.

    if contig_fasta != None:
        #check existing of the translated proteins
        protein_fasta = contig_fasta.split(".")[0]
        protein_fasta = protein_fasta + ".fa"

        if os.path.isfile(protein_fasta):
            print("The contig have been translated")
        else:
            protein_fasta,contig_sentence = translate_contigs_into_proteins(input_fasta=contig_fasta)

        run_diamond_blatp_alignment(input_protein_fasta=protein_fasta,ont=ont)


    elif protein_fasta != None and contig_sentence != None:
        run_diamond_blatp_alignment(input_protein_fasta=protein_fasta,ont=ont)
    else:
        print("There is an error in your input!")
        exit(0)


    # Align withe database and get the diamond score results.
    get_diamondscore(ont=ont, protein_fasta=protein_fasta)

    # Input protein into ESM model and get the embedding
    embedding_proteins_ESM2(plm_model_name=plm_model, fasta_file=protein_fasta)

    # Preparing the input files for GOPhage model including the protein names and the sequence embedding.
    contig_sentence = check_number_protein(contig_sentence=contig_sentence, ont=ont)

    preparing_context_protein_embedding(plm_model_name=plm_model,contig_sentence=contig_sentence)
    get_protein_names(contig_sentence=contig_sentence)
    get_sequence_location(model_name=plm_model,contig_sentence=contig_sentence)

    # run our model
    run_phaGO_model(plm_model_name=plm_model,ont=ont,batch_size= batch_size)

    # combine with the diamondscore
    combine_diamondblasp_phaGO(plm_model_name=plm_model,ont=ont)

    # output the final results
    output_the_prediction_results(ont=ont,plm_model_name=plm_model)
    end_time = time.time()
    spend_time = (end_time - start_time)/60
    print(f"Running time: {spend_time:.2f} min")
