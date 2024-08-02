![Alt text]("./phago.png")

# Overview

PhaGO is a learning-based model, that can be used for annotation of phage proteins based on the Gene ontology terms. The major improvement in PhaGO can be attributed to utilizing the properties of phages and the foundation model. The Transformer model is used to learn the relationship of the genomic context proteins.

In addition, we integrate PhaGO with the DiamondBlastp to further improve the performance. You can choose to run PhaGO+ which has two versions based on the ESM2-12 and ESM2-33.

## Quick install
Note: we suggest you install all the packages using Conda (both Miniconda and Anaconda are ok).

After cloning this repository, you can use Anaconda to install the ‘phago.yaml’. This will install all packages you need with GPU mode (make sure you have installed Cuda on your system to use the GPU version).


## Prepare the data and environment
Due to the limited size of the GitHub, we zip the data. You can download the database and model from Google Drive or Baidu Netdisk(百度网盘). You can follow the steps below to use PhaGO.

### 1. Download the code.
      git clone https://github.com/jiaojiaoguan/PhaGO.git
   
### 2. Install the conda environment.

      cd PhaGO/
      conda env create -f phago.yaml -n phago
      conda activate phago
   
### 3. Download the database and model.
  #### from the Google Drive:
  https://drive.google.com/drive/folders/14IQ75pMW9FK0H4mwleGEAo6_M7vOJeG5?usp=sharing
  
  #### from Baidu NetDisk(百度网盘):
  链接：https://pan.baidu.com/s/1UafDBBdNyGE4oIf8ZF0Ulg 
  提取码：phag
  
  Note: You need to put the "Database", "ESM_model", "PhaGO_model", "Protein_annotation" and "Term_label" folders in "PhaGO/".
  
### 4. Run PhaGO+ model.

#### Step1. Preprocess the contigs and generate the contig sentences.

      python preprocess.py 
                        --contigs inputs contig fasta file
                        --proteins input the protein fasta file
                        --sentences input the contig sentences file
                     
In this step, you have two types of input. First, you can input the contigs and we translate them into proteins. Second, you can directly input the protein fasta file and the contigs sentences file named "test_contig_sentence.csv" and separate each column with a comma.

For example, you have two contigs named contig_1 and contig_2. The proteins for contig_1 are p1 and p2, and the proteins for contig_2 are p3. The "test_contig_sentence.csv" is like this:

      contig_1, p1, p2
      contig_2, p3

After inputting the files, the step will do the alignment and output the prediction from DiamondBlastp.
    
##### Example.

      python preprocess.py --contigs test.fasta
      or 
      python preprocess.py --proteins test_proteins.fasta --sentences test_contig_sentence.csv
    
#### Step2. Generate the protein embedding using ESM2.

      python get_esm_embedding.py 
                     --plm The name of PLM model (esm2-12 or esm2-33)
                     
#### Example.

      python get_esm_embedding.py --plm esm2-12 

    
#### Step3. Preparing the input files for PhaGO model including the protein names and the sequence embedding.

      python prepare_PhaGO_input.py 
                    --plm The name of PLM model (esm2-12 or esm2-33)
                    
#### Example.

      python prepare_PhaGO_input.py --plm esm2-12
  
#### Step4. Run PhaGO model and output the final prediction combined with DiamondScore.

      python PhaGO.py 
                       --plm The name of PLM model (esm2-12 or esm2-33)
                       --ont The ontology including BP, CC and MF
                       --batch_size The batch size for the input
                       --cutoff  Set the cutoff for output the prediction score. 
                    
#### Example.
      python PhaGO.py --plm esm2-12 --ont BP --cutoff 0.2

### Output

If you use the esm2-12 model, the prediction will be written in BP_phago_base_plus_prediction_labels.csv.
If you use the esm2-33 model, the prediction will be written in BP_phago_large_plus_prediction_labels.csv.
The CSV file has three columns: Proteins, GO term, and score.
   

### Contact 
If you have any questions, please email us: jiaojguan2-c@my.cityu.edu.hk
