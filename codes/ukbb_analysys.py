import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Complex trait enrichment of top variant-gene pairs prioritized by EMSv2 in different tissues
tissues = ['Whole_Blood', 'Muscle_Skeletal', 'Liver', 'Brain_Cerebellum','Prostate', 'Spleen', 'Skin_Sun_Exposed_Lower_leg', 'Artery_Coronary',
       'Esophagus_Muscularis', 'Esophagus_Gastroesophageal_Junction','Artery_Tibial', 'Heart_Atrial_Appendage', 'Nerve_Tibial',
       'Heart_Left_Ventricle', 'Adrenal_Gland', 'Adipose_Visceral_Omentum','Pancreas', 'Lung', 'Pituitary',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Colon_Transverse','Adipose_Subcutaneous', 'Esophagus_Mucosa', 'Brain_Cortex', 'Thyroid',
       'Stomach', 'Breast_Mammary_Tissue', 'Colon_Sigmoid','Skin_Not_Sun_Exposed_Suprapubic', 'Testis', 'Artery_Aorta',
       'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24','Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere',
       'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus','Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1',
       'Brain_Substantia_nigra', 'Cells_Cultured_fibroblasts','Cells_EBV-transformed_lymphocytes', 'Kidney_Cortex',
       'Minor_Salivary_Gland', 'Ovary', 'Small_Intestine_Terminal_Ileum','Uterus', 'Vagina']

traits = ["Cardiovascular","Hematopoietic","Hepatic","Immunological","Lipids","Metabolic","Neurological","Other","Psychological","Renal","Skeletal"]
df_ukb_trait_enrichment = pd.DataFrame()
for trait in tqdm(traits):
    df_ukb_trait = pd.read_csv(f"./enformer_project/dataset/Output/UKBB_{trait}_EMSv2_PC100_tss_PIP01over.csv.gz", delimiter=",",index_col=0) 
    ukb_trait_top = []
    for tissue in tissues:
        df_ukb_trait = df_ukb_trait.sort_values(by=tissue, ascending=False)
        threshold = df_ukb_trait[tissue].quantile(0.99) 
        ukb_trait_top.append(df_ukb_trait[df_ukb_trait[tissue] >= threshold]["pip"].values) 
    df_ukb_trait_top = pd.DataFrame(ukb_trait_top).T
    df_ukb_trait_top.columns=tissues
    df_enr = df_ukb_trait_top.mean(axis=0,numeric_only=True)/df_ukb_trait["pip"].mean() 
    df_err = (df_ukb_trait_top.std() / np.sqrt(len(df_ukb_trait_top))) / df_ukb_trait["pip"].mean() 
    df_ukb_trait_enrichment[trait + "_enr"] = df_enr
    df_ukb_trait_enrichment[trait + "_err"] = df_err

df_ukb_trait_enrichment.to_csv(f"./enformer_project/analysis/UKBB/PIP_enrichment/UKBB_trait_enrichment_in_different_tissue_top1percent.csv",sep=',')

df_ukb_trait_enrichment = pd.read_csv(f"./enformer_project/analysis/UKBB/PIP_enrichment/UKBB_trait_enrichment_in_different_tissue_top1percent.csv", delimiter=",",index_col=0) 
num = len(df_ukb_trait_enrichment)
traits = ["Cardiovascular","Hematopoietic","Hepatic","Immunological","Lipids","Metabolic","Neurological","Other","Psychological","Renal","Skeletal"]

for trait in tqdm(traits):
    df_ukb_trait_enrichment = df_ukb_trait_enrichment.sort_values(trait+"_enr",ascending=True)
    plt.figure(figsize=(13, 4), dpi = 500)
    for i in range(num):
        color = plt.cm.viridis(i / num)  
        plt.errorbar(i, df_ukb_trait_enrichment[trait+"_enr"][i], df_ukb_trait_enrichment[trait+"_err"][i], fmt="o", color=color, markeredgecolor='black', markeredgewidth=0.5)
    plt.title(f'{trait}',fontsize=22)
    plt.xlabel('Tissue',fontsize=18)
    plt.xticks(np.arange(num), df_ukb_trait_enrichment.index,rotation=90)
    plt.ylabel('UKBB trait PIP enrichment',fontsize=18)
    plt.savefig(f'./enformer_project/analysis/UKBB/PIP_enrichment/UKBB_trait_enrichment_in_different_tissue_{trait}.pdf', bbox_inches='tight')


# examples of putative causal eQTL prioritized by EMSv2
# EMSv2 Whole Blood - UKBB - MPRA variant
def putative_causal_eqtl(df,df_tss,ensg_id):
    df1 = df["id"].str.split('_', expand=True).iloc[:,:2]
    df1.columns=['chr','locus']
    df1 =  pd.concat([df1,df],axis=1)
    df1["locus"]=df1["locus"].astype(int)

    df_tss["ensg_id"] = df_tss["ensg_id"].str.split('.', expand=True)[0]
    tss = df_tss["tss_position"][df_tss["ensg_id"] == ensg_id] 
    chrom = df_tss["chr"][df_tss["ensg_id"] == ensg_id].to_string(index=False)

    df_plot = df1[(df1["chr"]==chrom) & (df1["locus"] <= int(tss)+1e6) & (df1["locus"] >= int(tss)-1e6)]
    df_plot["tss_dist"]  = (df_plot["locus"] - int(tss))
    return df_plot 

#1 UKBB読み込み
df_ukb_all = pd.read_csv("./enformer_project/dataset/UKBB/UKBB_94traits_pip_hg38.tsv.gz", delimiter="\t",index_col=None,
                 usecols=["variant_hg38","rsid","trait","method","pip","pip_bin","categ"]) 
df_ukb_all = df_ukb_all[df_ukb_all["method"]=="SUSIE"]
df_ukb_all["variant_hg38"] = df_ukb_all["variant_hg38"].str.replace(':', '_')
df_ukb_all.rename(columns={'variant_hg38': 'id'},inplace=True)

#2 MPRA読み込み
df_mpra = pd.read_csv(f"./unique_xy/Input/validation_data/K562_pvals.tsv", delimiter="\t", keep_default_na=False) 
df_mpra.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_mpra.dropna(inplace=True)

#3 UKBBとMPRA統合
df = pd.merge(df_ukb_all, df_mpra, how="left",on="id") 
df_tss = pd.read_csv("./enformer_project/tss_distance_annot/gencode.v26.GRCh38.genes.tssposition.tsv", sep='\t')
df_ems = pd.read_csv(f"./enformer_project/dataset/Output/UKBB_94traits_MPRA_K562_EMSv2_Whole_Blood_EMSv2score_PC100_tss_scaling.csv.gz", delimiter=",",
                     index_col=None,usecols=["id","ensg_id","Whole_Blood_EMSv2"]) 
df_enf = pd.read_csv("./enformer_project/dataset/Input/UKBB_94traits_MPRA_K562_EMSv2_Whole_Blood_All_scores.csv.gz", delimiter=",",index_col=0)

ensg_id = "ENSG00000253252" #ENSG00000253252:RP11-10A14.6.

df_plot = putative_causal_eqtl(df,df_tss,ensg_id)
df_plot_mpra = df_plot.loc[:, ("id","tss_dist","emvar")].dropna(subset="emvar")
df_plot_ems =pd.merge(df_plot[df_plot["categ"]=="Hematopoietic"],df_ems[df_ems["ensg_id"]==ensg_id],how="left",on = "id")

ems_top_idx = df_plot["pip"][df_plot['emvar'] == 'Tier 1'].idxmax() 
var_id = df_plot.loc[df_plot['emvar'] == 'Tier 1', 'id'].loc[ems_top_idx]
rs_id =df_plot.loc[df_plot['emvar'] == 'Tier 1', 'rsid'].loc[ems_top_idx]
enf_top_f = df_enf[df_enf["id"]==var_id].iloc[:,5:].abs().idxmax(axis=1).to_string(index=False)
df_plot = pd.merge(df_plot,df_enf[["id",enf_top_f]],how="left",on="id")

#plot
f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, gridspec_kw=dict(height_ratios=[3,3,4,1]),figsize=(6, 6),dpi=300)
colors1 = np.where(df_plot["categ"]=="Hematopoietic", 'orange', 'blue') # Tier1を赤，それ以外は黒．
colors2 = np.where(df_plot_mpra["emvar"] == "Tier 1", 'red', np.where(df_plot_mpra["emvar"] == "Tier 2", 'orange', 'grey'))

gene_name = df_tss["gene_name"][df_tss["ensg_id"] == ensg_id].to_string(index=False)
x_line = df_plot.loc[df_plot['id']==var_id, 'tss_dist'].values[0]

ax1.scatter(df_plot["tss_dist"][colors1 == 'orange'], df_plot["pip"][colors1 == 'orange'], color="xkcd:orange", label="Hematopoietic", zorder=2)
ax1.scatter(df_plot["tss_dist"][colors1 == 'blue'], df_plot["pip"][colors1 == 'blue'], color="blue", label="Other trait categories", zorder=1)
ax2.scatter(df_plot_ems["tss_dist"], df_plot_ems["Whole_Blood_EMSv2"], color="green", label="pip")
ax3.scatter(df_plot["tss_dist"][df_plot["categ"]=="Hematopoietic"], df_plot[enf_top_f][df_plot["categ"]=="Hematopoietic"], color="black")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'grey'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'grey'], c='grey', label="None")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'orange'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'orange'], c='xkcd:orange', label="Tier2")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'red'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'red'], c='red', label="Tier1") #Tier1赤が先頭に来るように．

ax1.set_ylabel("PIP$_{UKBB}$",rotation=0, ha='right', va='center', multialignment='center')
ax2.set_ylabel("EMS\n(Whole Blood)",rotation=0, ha='right', va='center', multialignment='center')
ax3.set_ylabel("Enformer score\n(DNASE : K562)",rotation=0, ha='right', va='center', multialignment='center')
ax4.set_ylabel("MPRA (K562)",rotation=0, ha='right', va='center')
ax4.set_xlabel(fr"Distance to TSS of ${gene_name}$")

ax1.set_ylim([0,1.05])
ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize=12)
ax4.set_ylim([-0.5, 0.5])
ax4.set_yticks([]) 
ax4.set_xlim([-1e6, 1e6])

ax1.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax2.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax3.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax4.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax1.text(x_line, 1.2, f"{rs_id}", ha='center', va='center') 

plt.subplots_adjust(hspace=0.3)

f.savefig(f'./enformer_project/analysis/UKBB/putative_causal_eQTL/Hematopoietic_K562/UKBB_MPRA_causal_eQTL_{gene_name}.pdf', bbox_inches='tight')


# EMSv2 Liver - UKBB - MPRA variant
#1 UKBB読み込み
df_ukb_all = pd.read_csv("./enformer_project/dataset/UKBB/UKBB_94traits_pip_hg38.tsv.gz", delimiter="\t",index_col=None,
                 usecols=["variant_hg38","rsid","trait","method","pip","pip_bin","categ"]) 
df_ukb_all = df_ukb_all[df_ukb_all["method"]=="SUSIE"] 
df_ukb_all["variant_hg38"] = df_ukb_all["variant_hg38"].str.replace(':', '_')
df_ukb_all.rename(columns={'variant_hg38': 'id'},inplace=True)

#2 MPRA読み込み
df_mpra = pd.read_csv(f"./unique_xy/Input/validation_data/HepG2_pvals.tsv", delimiter="\t", keep_default_na=False) 
df_mpra.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_mpra.dropna(inplace=True)

#3 UKBBとMPRA統合
# All variant-geneの場合
df = pd.merge(df_ukb_all, df_mpra, how="left",on="id") 
df_tss = pd.read_csv("./enformer_project/tss_distance_annot/gencode.v26.GRCh38.genes.tssposition.tsv", sep='\t')
df_ems = pd.read_csv(f"./enformer_project/dataset/Output/UKBB_94traits_MPRA_HepG2_EMSv2_Liver_EMSv2score_PC100_tss_scaling.csv.gz", delimiter=",",
                     index_col=None,usecols=["id","ensg_id","Liver_EMSv2"]) 
df_enf = pd.read_csv("./enformer_project/dataset/Input/UKBB_94traits_MPRA_HepG2_EMSv2_Liver_All_scores.csv.gz", delimiter=",",index_col=0)

ensg_id = "ENSG00000111321" 

df_plot = putative_causal_eqtl(df,df_tss,ensg_id)
df_plot_mpra = df_plot.loc[:, ("id","tss_dist","emvar")].dropna(subset="emvar") 
df_plot_ems =pd.merge(df_plot[df_plot["categ"]=="Hepatic"],df_ems[df_ems["ensg_id"]==ensg_id],how="left",on = "id")

ems_top_idx = df_plot["pip"][df_plot['emvar'] == 'Tier 1'].idxmax() 
var_id = df_plot.loc[df_plot['emvar'] == 'Tier 1', 'id'].loc[ems_top_idx]
rs_id =df_plot.loc[df_plot['emvar'] == 'Tier 1', 'rsid'].loc[ems_top_idx]
enf_top_f = df_enf[df_enf["id"]==var_id].iloc[:,5:].abs().idxmax(axis=1).to_string(index=False)
df_plot = pd.merge(df_plot,df_enf[["id",enf_top_f]],how="left",on="id")

#plot
f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, gridspec_kw=dict(height_ratios=[3,3,4,1]),figsize=(6, 6),dpi=300)
colors1 = np.where(df_plot["categ"]=="Hepatic", 'orange', 'blue') # Tier1を赤，それ以外は黒．
colors2 = np.where(df_plot_mpra["emvar"] == "Tier 1", 'red', np.where(df_plot_mpra["emvar"] == "Tier 2", 'orange', 'grey'))

gene_name = df_tss["gene_name"][df_tss["ensg_id"] == ensg_id].to_string(index=False)
x_line = df_plot.loc[df_plot['id']==var_id, 'tss_dist'].values[0]

ax1.scatter(df_plot["tss_dist"][colors1 == 'orange'], df_plot["pip"][colors1 == 'orange'], color="xkcd:orange", label="Hepatic", zorder=2)
ax1.scatter(df_plot["tss_dist"][colors1 == 'blue'], df_plot["pip"][colors1 == 'blue'], color="blue", label="Other trait categories", zorder=1)
ax2.scatter(df_plot_ems["tss_dist"], df_plot_ems["Liver_EMSv2"], color="green", label="pip")
ax3.scatter(df_plot["tss_dist"][df_plot["categ"]=="Hepatic"], df_plot[enf_top_f][df_plot["categ"]=="Hepatic"], color="black")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'grey'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'grey'], c='grey', label="None")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'orange'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'orange'], c='xkcd:orange', label="Tier2")
ax4.scatter(df_plot_mpra["tss_dist"][colors2 == 'red'], np.zeros_like(df_plot_mpra["tss_dist"])[colors2 == 'red'], c='red', label="Tier1") #Tier1赤が先頭に来るように．

ax1.set_ylabel("PIP$_{UKBB}$",rotation=0, ha='right', va='center', multialignment='center')
ax2.set_ylabel("EMS\n(Liver)",rotation=0, ha='right', va='center', multialignment='center')
ax3.set_ylabel("Enformer score\n(CAGE : CD133+)",rotation=0, ha='right', va='center', multialignment='center')
ax4.set_ylabel("MPRA(HepG2)",rotation=0, ha='right', va='center')
ax4.set_xlabel(fr"Distance to TSS of ${gene_name}$")
ax1.set_ylim([0,1.05])
ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0),fontsize=12)
ax4.set_ylim([-0.5, 0.5])
ax4.set_yticks([]) 
ax4.set_xlim([0, 20000])

ax1.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax2.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax3.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax4.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax1.text(x_line, 1.2, f"{rs_id}", ha='center', va='center') 
plt.subplots_adjust(hspace=0.3)

f.savefig(f'./enformer_project/analysis/UKBB/putative_causal_eQTL/Hepatic_HepG2/UKBB_MPRA_causal_eQTL_{gene_name}.pdf', bbox_inches='tight')

