
![icon](https://github.com/jiaojiaoguan/GOPhage/blob/main/gophage.png)
# Overview

GOPhage is a learning-based model, that can be used for annotation of phage proteins based on the Gene ontology terms. The major improvement in GOPhage can be attributed to utilizing the properties of phages and the foundation model. The Transformer model is used to learn the relationship of the genomic context proteins.

In addition, we integrate GOPhage with the DiamondBlastp to further improve the performance. You can choose to run GOPhage+ which has two versions based on the ESM2-12 and ESM2-33.

## Quick install
Note: we suggest you install all the packages using Conda (both Miniconda and Anaconda are ok).

After cloning this repository, you can use Anaconda to install the ‘GOPhage.yaml’. This will install all packages you need with GPU mode (make sure you have installed Cuda on your system to use the GPU version).


## Prepare the data and environment
Due to the limited size of the GitHub, we zip the data. You can download the database and model from Google Drive or Baidu Netdisk(百度网盘). You can follow the steps below to use GOPhage.

### 1. Download the code.
      git clone https://github.com/jiaojiaoguan/GOPhage.git
   
### 2. Install the conda environment.

      cd GOPhage/
      conda env create -f gophage.yaml -n gophage
      conda activate gophage
   
### 3. Download the database and model.
  #### from the Google Drive:
  https://drive.google.com/drive/folders/14IQ75pMW9FK0H4mwleGEAo6_M7vOJeG5?usp=sharing
  
  #### from Baidu NetDisk(百度网盘):
  链接：https://pan.baidu.com/s/1UafDBBdNyGE4oIf8ZF0Ulg 
  提取码：phag
  
  Note: You need to put the "Database", "ESM_model", "GOPhage_model", "Protein_annotation" and "Term_label" folders in "GOPhage/".
  
### 4. Run GOPhage+ model.
      python GOPhage.py 
                  --contigs [DNA FASTA file of contigs, you can only input contigs and ignore the --proteins and --sentences]
                  --plm [The name of PLM model (esm2-12 or esm2-33)]
                  --ont [The ontology including BP, CC, and MF]
                  --batch_size [The batch size for the input]
                  --mid_dir [The directory for saved results]
                  --threshold [The GO that satisfies the threshold will be output. default: 0.1]
                    
#### Example.
      python GOPhage.py --contigs test_conitg.fasta --ont BP --plm esm2-12 --mid_dir BP_results


### Output

If you use the esm2-12 model, the prediction will be written in BP_GOPhage_base_plus_prediction_labels_summary.csv.
If you use the esm2-33 model, the prediction will be written in BP_GOPhage_large_plus_prediction_labels_summary.csv.
The CSV file has two columns: Proteins, GO term
   

### Contact 
If you have any questions, please email us: jiaojguan2-c@my.cityu.edu.hk
