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



def run_phaGO_model(plm_model_name,ont,batch_size):

    if plm_model_name== "esm2-12":
        nhead = 12
        d_model = 480

        if ont=="CC":
            dim_feedforward=128
        else:
            dim_feedforward = 480

        model_name = "./PhaGO_model/"+ont+"_PhaGO_base_model.th"
        out_put_file= "./" + ont + '_phago_base_results.pkl'
        data_file_input = "test_location_esm12.csv"

    elif plm_model_name=="esm2-33":
        nhead = 16
        d_model = 1280
        if ont=="CC":
            dim_feedforward=320
        else:
            dim_feedforward = 1280

        model_name = "./PhaGO_model/" + ont + "_PhaGO_large_model.th"
        out_put_file = "./" + ont + '_phago_large_results.pkl'
        data_file_input = "test_location_esm33.csv"


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
        output_results = "./"+ont+ "_phago_base_plus_predictions.pkl"
        phago_prediction_results = "./" + ont + '_phago_base_results.pkl'

    elif plm_model_name=="esm2-33":
        dict_onyology_alpha = {"BP": 0.9, "CC": 0.62, "MF": 0.82}
        output_results = "./"+ont+ "_phago_large_plus_predictions.pkl"
        phago_prediction_results = "./" + ont + '_phago_large_results.pkl'
    else:
        print("Error, please the correct plm model name!")
        exit(0)

    alpha_parameter = dict_onyology_alpha[ont]

    # diamond_score_prediction_results.
    diamond_blastp_result_file = "./" + ont + '_test_diamondblastp_results.pkl'
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

        phagoplus_prediction_results = "./" + ont + "_phago_base_plus_predictions.pkl"
        file1 = open(ont + "_GOPhage_base_plus_prediction_labels.csv", "w")
        file2=open("./PhaGO_model/esm12_"+ont+"_label_threshold.csv")
        next(file2)

    elif plm_model_name=="esm2-33":

        phagoplus_prediction_results = "./" + ont + "_phago_large_plus_predictions.pkl"
        file1 = open(ont + "_GOPhage_large_plus_prediction_labels.csv", "w")
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
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description="""PhaGO is for phage protein annotation""")


    parser.add_argument('--plm', help='name of PLM model (esm2-12 or esm2-33)', type=str, default='esm2-33')
    parser.add_argument('--ont', help='The ontology including BP, CC and MF', type=str, default='CC')
    parser.add_argument('--batch_size', help="batch size", type=int, default=2)


    inputs = parser.parse_args()

    plm_model = inputs.plm
    ont = inputs.ont
    batch_size = inputs.batch_size


    # call the function
    run_phaGO_model(plm_model_name=plm_model,ont=ont,batch_size= batch_size)
    combine_diamondblasp_phaGO(plm_model_name=plm_model,ont=ont)
    output_the_prediction_results(ont=ont,plm_model_name=plm_model)
