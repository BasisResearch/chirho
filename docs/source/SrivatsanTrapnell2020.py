import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix

# As raised in the issue here https://github.com/sanderlab/scPerturb/issues/7
# there is an issue w/ processing the sciplex dataset. In this script, we fix the bugs
# in the preprocessing script here https://github.com/sanderlab/scPerturb/blob/master/dataset_processing/SrivatsanTrapnell2020.py
# for the sciplex3 dataset. We also remove the column renaming and dropping of columns from the script
# in scPerturb.

# Download and unzip 3 files from here:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378
# (1) GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz
# (2) GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz
# (3) GSM4150378_sciPlex3_pData.txt.gz

var = pd.read_csv(
    "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt",
    sep=" ",
)
var.columns = ["gene_id", "gene_name"]

var = var.set_index("gene_name")

obs2 = pd.read_csv("GSM4150378_sciPlex3_pData.txt", sep=" ")
counts = [x for x in files if "UMI.count.matrix" in x][0]

UMI_counts = pd.read_csv(
    "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix", sep="\t", header=None
)

X = csr_matrix(
    (UMI_counts[2], (UMI_counts[1] - 1, UMI_counts[0] - 1)), shape=(len(obs2), len(var))
)  # this may crash your kernel

UMI_counts = None  # save on memory
adata = sc.AnnData(X, obs=obs2, var=var)

# reform var
adata.var.index = adata.var.index.rename("gene_symbol")
adata.var["ensembl_gene_id"] = [x.split(".")[0] for x in adata.var["gene_id"]]
adata.obs["dose_unit"] = "nM"
adata.obs["celltype"] = [
    "alveolar basal epithelial cells"
    if line == "A549"
    else "mammary epithelial cells"
    if line == "MCF7"
    else "lymphoblasts"
    if line == "K562"
    else "None"
    for line in adata.obs.cell_type
]
adata.obs["disease"] = [
    "lung adenocarcinoma"
    if line == "A549"
    else "breast adenocarcinoma"
    if line == "MCF7"
    else "chronic myelogenous leukemia"
    if line == "K562"
    else "None"
    for line in adata.obs.cell_type
]
adata.write("SrivatsanTrapnell2020_sciplex3_debugged.h5")
