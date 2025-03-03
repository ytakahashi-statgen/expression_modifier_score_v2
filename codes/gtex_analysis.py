import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns



# EMSv2 outperforms other methods in causal eQTL prioritization
def get_auroc(tissue, y_pred):
    df = pd.DataFrame({"real":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
    df = df.loc[~df.real.isna(),:]
    df = df.loc[~df.pred.isna(),:]
    return (metrics.roc_auc_score(df.real, df.pred))
def get_auprc(tissue, y_pred):
    df = pd.DataFrame({"real":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
    df = df.loc[~df.real.isna(),:]
    df = df.loc[~df.pred.isna(),:]
    return (metrics.average_precision_score(df.real, df.pred))

tissues = ['Whole_Blood', 'Muscle_Skeletal', 'Liver', 'Brain_Cerebellum','Prostate', 'Spleen', 'Skin_Sun_Exposed_Lower_leg', 'Artery_Coronary',
       'Esophagus_Muscularis', 'Esophagus_Gastroesophageal_Junction','Artery_Tibial', 'Heart_Atrial_Appendage', 'Nerve_Tibial',
       'Heart_Left_Ventricle', 'Adrenal_Gland', 'Adipose_Visceral_Omentum','Pancreas', 'Lung', 'Pituitary',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Colon_Transverse','Adipose_Subcutaneous', 'Esophagus_Mucosa', 'Brain_Cortex', 'Thyroid',
       'Stomach', 'Breast_Mammary_Tissue', 'Colon_Sigmoid','Skin_Not_Sun_Exposed_Suprapubic', 'Testis', 'Artery_Aorta',
       'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24','Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere',
       'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus','Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1',
       'Brain_Substantia_nigra', 'Cells_Cultured_fibroblasts','Cells_EBV-transformed_lymphocytes', 'Kidney_Cortex',
       'Minor_Salivary_Gland', 'Ovary', 'Small_Intestine_Terminal_Ileum','Uterus', 'Vagina']

#1 EMSv2
df_emsv2= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv2_PC100_tss_finalize.csv.gz", delimiter=",",index_col=0)
aurocs = pd.Series(tissues).apply(lambda x: get_auroc(x, df_emsv2))
auprcs = pd.Series(tissues).apply(lambda x: get_auprc(x, df_emsv2))
out = pd.DataFrame({"aurocs":aurocs, "auprcs":auprcs})
out.index = tissues
out = out.T
out.to_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_EMSv2_finalized.csv",sep=",")

#2 EMSv1
df_emsv1 = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv1.csv.gz", delimiter=",",index_col=0) 
aurocs = pd.Series(tissues).apply(lambda x: get_auroc(x, df_emsv1))
auprcs = pd.Series(tissues).apply(lambda x: get_auprc(x, df_emsv1))
out = pd.DataFrame({"aurocs":aurocs, "auprcs":auprcs})
out.index = tissues
out = out.T
out.to_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_EMSv1_finalized.csv",sep=",")

#3 Distance to TSS
df_logistic_tss= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Logistic_Tss_distance.csv.gz", delimiter=",",index_col=0) 
aurocs = pd.Series(tissues).apply(lambda x: get_auroc(x, df_logistic_tss))
auprcs = pd.Series(tissues).apply(lambda x: get_auprc(x, df_logistic_tss))
out = pd.DataFrame({"aurocs":aurocs, "auprcs":auprcs})
out.index = tissues
out = out.T
out.to_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_Logistic_Tss_distance.csv",sep=",")

#4 Sei
df_sei = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Sei.csv.gz", delimiter=",",index_col=0) 
aurocs = pd.Series(tissues).apply(lambda x: get_auroc(x, df_sei))
auprcs = pd.Series(tissues).apply(lambda x: get_auprc(x, df_sei))
out = pd.DataFrame({"aurocs":aurocs, "auprcs":auprcs})
out.index = tissues
out = out.T
out.to_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_Sei.csv",sep=",")


scores_sei =pd.read_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_Sei.csv",sep=",",index_col=0).T
scores_tss =pd.read_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_Logistic_Tss_distance.csv",sep=",",index_col=0).T
scores_emsv1 =pd.read_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_EMSv1_finalized.csv",sep=",",index_col=0).T
scores_emsv2 =pd.read_csv(f"./enformer_project/analysis/GTEx/AUC/GTEx_AUC_EMSv2_finalized.csv",sep=",",index_col=0).T

# EMSv2基準に並べる。
tissues_sort_auroc= ['Testis', 'Brain_Cerebellum', 'Cells_EBV-transformed_lymphocytes',
       'Thyroid', 'Brain_Caudate_basal_ganglia', 'Whole_Blood',
       'Cells_Cultured_fibroblasts', 'Muscle_Skeletal',
       'Brain_Substantia_nigra', 'Brain_Spinal_cord_cervical_c-1',
       'Brain_Cerebellar_Hemisphere', 'Artery_Tibial', 'Nerve_Tibial',
       'Skin_Sun_Exposed_Lower_leg', 'Brain_Putamen_basal_ganglia', 'Vagina',
       'Pituitary', 'Skin_Not_Sun_Exposed_Suprapubic', 'Brain_Cortex',
       'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Brain_Frontal_Cortex_BA9',
       'Brain_Hippocampus', 'Pancreas', 'Esophagus_Mucosa',
       'Breast_Mammary_Tissue', 'Prostate', 'Ovary', 'Liver',
       'Heart_Atrial_Appendage', 'Minor_Salivary_Gland',
       'Esophagus_Muscularis', 'Brain_Hypothalamus', 'Artery_Aorta',
       'Heart_Left_Ventricle', 'Small_Intestine_Terminal_Ileum', 'Spleen',
       'Brain_Anterior_cingulate_cortex_BA24',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Colon_Transverse', 'Lung',
       'Adipose_Subcutaneous', 'Stomach', 'Brain_Amygdala',
       'Esophagus_Gastroesophageal_Junction', 'Colon_Sigmoid',
       'Artery_Coronary', 'Kidney_Cortex', 'Uterus']

tissues_sort_auprc= ['Brain_Substantia_nigra', 'Brain_Anterior_cingulate_cortex_BA24',
       'Brain_Spinal_cord_cervical_c-1', 'Brain_Amygdala', 'Kidney_Cortex',
       'Cells_EBV-transformed_lymphocytes', 'Brain_Hippocampus',
       'Minor_Salivary_Gland', 'Vagina', 'Brain_Putamen_basal_ganglia',
       'Brain_Hypothalamus', 'Uterus', 'Liver', 'Brain_Frontal_Cortex_BA9',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Caudate_basal_ganglia',
       'Testis', 'Brain_Cerebellum', 'Brain_Cortex',
       'Brain_Cerebellar_Hemisphere', 'Ovary', 'Pituitary', 'Adrenal_Gland',
       'Pancreas', 'Whole_Blood', 'Prostate', 'Spleen',
       'Small_Intestine_Terminal_Ileum', 'Heart_Left_Ventricle',
       'Artery_Coronary', 'Muscle_Skeletal', 'Esophagus_Mucosa',
       'Cells_Cultured_fibroblasts', 'Skin_Not_Sun_Exposed_Suprapubic',
       'Colon_Sigmoid', 'Stomach', 'Breast_Mammary_Tissue',
       'Adipose_Visceral_Omentum', 'Heart_Atrial_Appendage', 'Artery_Aorta',
       'Esophagus_Gastroesophageal_Junction', 'Skin_Sun_Exposed_Lower_leg',
       'Thyroid', 'Lung', 'Nerve_Tibial', 'Adipose_Subcutaneous',
       'Colon_Transverse', 'Artery_Tibial', 'Esophagus_Muscularis']

# AUROCをEMSv2基準にソート。
num = len(tissues_sort_auroc)
plt.figure(figsize=(13, 5), dpi = 500)
plt.scatter(np.arange(num)-0.075, scores_sei.loc[tissues_sort_auroc]["aurocs"],marker="D",color="tab:blue",label="Sei", edgecolor='black', s=50)
plt.scatter(np.arange(num)-0.025, scores_tss.loc[tissues_sort_auroc]["aurocs"],marker="s",color="tab:orange",label="TSS", edgecolor='black', s=50)
plt.scatter(np.arange(num)+0.025, scores_emsv1.loc[tissues_sort_auroc]["aurocs"],marker="^",color="tab:olive",label="EMSv1", edgecolor='black', s=50)
plt.scatter(np.arange(num)+0.075, scores_emsv2.loc[tissues_sort_auroc]["aurocs"],marker="o",color="tab:green",label="EMSv2", edgecolor='black', s=50)

plt.ylabel('AUROC',fontsize=18)
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 2, 1, 0]  
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=20)
plt.xticks(np.arange(num), tissues_sort_auroc, rotation=90)
plt.xlabel("Tissue", fontsize=18)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/GTEx_AUROC.pdf', bbox_inches='tight')

# AUPRCをEMSv2基準にソート。
num = len(tissues_sort_auprc)
plt.figure(figsize=(13, 5), dpi = 500)
plt.scatter(np.arange(num)-0.075, scores_sei.loc[tissues_sort_auprc]["auprcs"],marker="D",color="tab:blue",label="Sei", edgecolor='black', s=50)
plt.scatter(np.arange(num)-0.025, scores_tss.loc[tissues_sort_auprc]["auprcs"],marker="s",color="tab:orange",label="TSS", edgecolor='black', s=50)
plt.scatter(np.arange(num)+0.025, scores_emsv1.loc[tissues_sort_auprc]["auprcs"],marker="^",color="tab:olive",label="EMSv1", edgecolor='black', s=50)
plt.scatter(np.arange(num)+0.075, scores_emsv2.loc[tissues_sort_auprc]["auprcs"],marker="o",color="tab:green",label="EMSv2", edgecolor='black', s=50)

plt.ylabel('AUPRC',fontsize=18)
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 2, 1, 0]  
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0, fontsize=20)
plt.xticks(np.arange(num), tissues_sort_auprc,rotation=90)
plt.xlabel("Tissue", fontsize=18)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/GTEx_AUPRC.pdf', bbox_inches='tight')


# TSS bin
def auc_tss_bin(df,tissues):
    # Tss_distanceのbin設定
    column_bins = ["[0, 1,000]","(1,000, 3,000]","(3,000, 10,000]","(10,000, 1,000,000]"]
    bins = [0, np.log(1000+1),np.log(3000+1), np.log(10000+1), np.log(1000000+1)]
    
    def roc_group(y_pred):
        df = pd.DataFrame({"label":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
        df = df.loc[~df["label"].isna(),:]
        return metrics.roc_auc_score(df["label"], df["pred"])
    def prc_group(y_pred):
        df = pd.DataFrame({"label":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
        df = df.loc[~df["label"].isna(),:]
        return metrics.average_precision_score(df["label"], df["pred"])

    df_roc = pd.DataFrame()
    df_prc = pd.DataFrame()
    for tissue in tissues:
      tmp1 = pd.cut(df["tss_distance"], bins=bins, include_lowest = True, right=True)
      tmp2 = df.groupby(tmp1).apply(roc_group)
      tmp3 = df.groupby(tmp1).apply(prc_group)
      df_roc = pd.concat([df_roc,tmp2],axis=1)
      df_prc = pd.concat([df_prc,tmp3],axis=1)
    df_roc.columns=tissues
    df_prc.columns=tissues

    # AUROC,AUPRC
    df_roc = df_roc.T
    df_prc = df_prc.T
    df_roc.columns=column_bins
    df_prc.columns=column_bins
    df = pd.DataFrame()
    df_score = pd.DataFrame()
    for column_bin in column_bins:
      df["AUROC"] = df_roc.loc[:,column_bin]
      df["AUPRC"] = df_prc.loc[:,column_bin]
      df["Tss_bin"] = column_bin
      df_score = pd.concat([df_score,df],axis=0) 
        
    return df_score

# TSS bin
df_tss = pd.read_csv("./unique_xy/Input/yx_test_pcs9.csv.gz", delimiter=",",index_col=None,usecols=["variant_id","gene_id","tss_distance"]) 

#1 EMSv2
df_emsv2= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv2_PC100_tss_finalize.csv.gz", delimiter=",",index_col=0)
df_emsv2 = pd.merge(df_emsv2,df_tss,how="left",on=["variant_id","gene_id"])
df_emsv2_auc = auc_tss_bin(df_emsv2,tissues)
df_emsv2_auc["method"] = "EMSv2"

#2 EMSv1
df_emsv1 = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv1.csv.gz", delimiter=",",index_col=0) 
df_emsv1 = pd.merge(df_emsv1,df_tss,how="left",on=["variant_id","gene_id"])
df_emsv1_auc = auc_tss_bin(df_emsv1,tissues)
df_emsv1_auc["method"] = "EMSv1"

#3 Distance to TSS
df_logistic_tss= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Logistic_Tss_distance.csv.gz", delimiter=",",index_col=0) 
df_logistic_tss = pd.merge(df_logistic_tss,df_tss,how="left",on=["variant_id","gene_id"])
df_logistic_tss_auc = auc_tss_bin(df_logistic_tss,tissues)
df_logistic_tss_auc["method"] = "Tss_distance"

#4 Sei
df_sei = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Sei.csv.gz", delimiter=",",index_col=0) 
df_sei = pd.merge(df_sei,df_tss,how="left",on=["variant_id","gene_id"])
df_sei_auc = auc_tss_bin(df_sei,tissues)
df_sei_auc["method"] = "Sei"

scores = pd.concat([df_emsv2_auc,df_emsv1_auc,df_logistic_tss_auc,df_sei_auc],axis=0)

plt.figure(figsize=(12, 6), dpi = 500)
order =["Sei","Tss_distance", "EMSv1","EMSv2"]
sns.swarmplot(data=scores,x="Tss_bin",y="AUROC",hue="method",hue_order=order,palette=["tab:blue", "tab:orange","tab:olive", "tab:green"], dodge=True, size=4)
# plt.xticks(rotation=90)
plt.legend(handles=[
                    plt.Line2D([], [], marker='None', color='tab:green', linewidth=5,label="EMSv2"),
                    plt.Line2D([], [], marker='None', color='tab:olive', linewidth=5,label="EMSv1"),
                    plt.Line2D([], [], marker='None', color='tab:orange', linewidth=5, label="TSS"),
                    plt.Line2D([], [], marker='None', color='tab:blue', linewidth=5, label="Sei")],
           bbox_to_anchor=(0.85, 0.26), loc='upper left', borderaxespad=0)
plt.ylabel('AUROC',fontsize=20)
plt.xlabel('Variant TSS distance',fontsize=20)
plt.ylim(0.2, 1)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/Tss_bin/GTEx_auroc_Tss_bin_finalized.pdf', bbox_inches='tight')

plt.figure(figsize=(12, 6), dpi = 500)
sns.swarmplot(data=scores,x="Tss_bin",y="AUPRC",hue="method",hue_order=order,palette=["tab:blue", "tab:orange","tab:olive", "tab:green"], dodge=True, size=4)
# plt.xticks(rotation=90)
plt.legend(handles=[
                    plt.Line2D([], [], marker='None', color='tab:green', linewidth=5,label="EMSv2"),
                    plt.Line2D([], [], marker='None', color='tab:olive', linewidth=5,label="EMSv1"),
                    plt.Line2D([], [], marker='None', color='tab:orange', linewidth=5, label="TSS"),
                    plt.Line2D([], [], marker='None', color='tab:blue', linewidth=5, label="Sei")],
           bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0)
plt.ylabel('AUPRC',fontsize=20)
plt.xlabel('Variant TSS distance',fontsize=20)
plt.ylim(0, 0.4)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/Tss_bin/GTEx_auprc_Tss_bin_finalized.pdf', bbox_inches='tight')


# Gene TPM bin
def auc_tpm_bin(df,tissues):
    # tpmのbin設定
    column_bins =  ["Low","Intermediate","High"]
    bins = [0, 2.045,12.468, 5208.114]
    
    def roc_group(y_pred):
        df = pd.DataFrame({"label":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
        df = df.loc[~df["label"].isna(),:]
        return metrics.roc_auc_score(df["label"], df["pred"])
    def prc_group(y_pred):
        df = pd.DataFrame({"label":y_pred[tissue] ,"pred":y_pred[tissue+".1"]})
        df = df.loc[~df["label"].isna(),:]
        return metrics.average_precision_score(df["label"], df["pred"])

    df_roc = pd.DataFrame()
    df_prc = pd.DataFrame()
    for tissue in tissues:
      tmp1 = pd.cut(df["Mean_tpm"], bins=bins, include_lowest = True, right=True)
      tmp2 = df.groupby(tmp1).apply(roc_group)
      tmp3 = df.groupby(tmp1).apply(prc_group)
      df_roc = pd.concat([df_roc,tmp2],axis=1)
      df_prc = pd.concat([df_prc,tmp3],axis=1)
    df_roc.columns=tissues
    df_prc.columns=tissues

    # AUROC,AUPRC
    df_roc = df_roc.T
    df_prc = df_prc.T
    df_roc.columns=column_bins
    df_prc.columns=column_bins
    df = pd.DataFrame()
    df_score = pd.DataFrame()
    for column_bin in column_bins:
      df["AUROC"] = df_roc.loc[:,column_bin]
      df["AUPRC"] = df_prc.loc[:,column_bin]
      df["TPM_bin"] = column_bin
      df_score = pd.concat([df_score,df],axis=0) 
        
    return df_score

gene_tpm = pd.read_csv("./unique_xy/Input/parsed_gene_features_including_tpm.tsv.gz", delimiter="\t")
gene_tpm.rename(columns = {"Unnamed: 0":"gene_id"}, inplace=True)
gene_tpm = gene_tpm.iloc[:,:55]
gene_tpm["Mean_tpm"] = gene_tpm.iloc[:,1:].mean(axis=1)

#1 EMSv2
df_emsv2= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv2_PC100_tss_finalize.csv.gz", delimiter=",",index_col=0)
df_emsv2["gene_id"] = df_emsv2["gene_id"].str.split(".", expand=True)[0]
df_emsv2 = pd.merge(df_emsv2,gene_tpm.loc[:,["gene_id","Mean_tpm"]], how="left",on="gene_id")
df_emsv2_tpm = auc_tpm_bin(df_emsv2,tissues)
df_emsv2_tpm["method"] = "EMSv2"

#2 EMSv1
df_emsv1 = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_EMSv1.csv.gz", delimiter=",",index_col=0) 
df_emsv1["gene_id"] = df_emsv1["gene_id"].str.split(".", expand=True)[0]
df_emsv1 = pd.merge(df_emsv1,gene_tpm.loc[:,["gene_id","Mean_tpm"]], how="left",on="gene_id")
df_emsv1_tpm = auc_tpm_bin(df_emsv1,tissues)
df_emsv1_tpm["method"] = "EMSv1"

#3 Distance to TSS
df_logistic_tss= pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Logistic_Tss_distance.csv.gz", delimiter=",",index_col=0) 
df_logistic_tss["gene_id"] = df_logistic_tss["gene_id"].str.split(".", expand=True)[0]
df_logistic_tss = pd.merge(df_logistic_tss,gene_tpm.loc[:,["gene_id","Mean_tpm"]], how="left",on="gene_id")
df_logistic_tss_tpm = auc_tpm_bin(df_logistic_tss,tissues)
df_logistic_tss_tpm["method"] = "Tss_distance"

#4 Sei
df_sei = pd.read_csv("./unique_xy/Output/keras/multi_task/GTEx_Sei.csv.gz", delimiter=",",index_col=0) 
df_sei["gene_id"] = df_sei["gene_id"].str.split(".", expand=True)[0]
df_sei = pd.merge(df_sei,gene_tpm.loc[:,["gene_id","Mean_tpm"]], how="left",on="gene_id")
df_sei_tpm = auc_tpm_bin(df_sei,tissues)
df_sei_tpm["method"] = "Sei"

scores_tpm = pd.concat([df_emsv2_tpm,df_emsv1_tpm,df_logistic_tss_tpm,df_sei_tpm],axis=0)

plt.figure(figsize=(12, 6), dpi = 300)
order =["Sei","Tss_distance", "EMSv1","EMSv2"]
sns.swarmplot(data=scores_tpm,x="TPM_bin",y="AUROC",hue="method",hue_order=order,palette=["tab:blue", "tab:orange","tab:olive", "tab:green"], dodge=True, size=4)
# plt.xticks(rotation=90)
plt.legend(handles=[
                    plt.Line2D([], [], marker='None', color='tab:green', linewidth=5,label="EMSv2"),
                    plt.Line2D([], [], marker='None', color='tab:olive', linewidth=5,label="EMSv1"),
                    plt.Line2D([], [], marker='None', color='tab:orange', linewidth=5, label="TSS"),
                    plt.Line2D([], [], marker='None', color='tab:blue', linewidth=5, label="Sei")],
           bbox_to_anchor=(0.85, 0.26), loc='upper left', borderaxespad=0)
plt.ylabel('AUROC')
plt.xlabel('Gene TPM')
plt.ylim(0.5, 1)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/TPM_bin/GTEx_auroc_TPM_bin_finalized.pdf', bbox_inches='tight')

plt.figure(figsize=(12, 5), dpi = 300)
sns.swarmplot(data=scores_tpm,x="TPM_bin",y="AUPRC",hue="method",hue_order=order,palette=["tab:blue", "tab:orange","tab:olive", "tab:green"], dodge=True, size=4)
# plt.xticks(rotation=90)
plt.legend(handles=[
                    plt.Line2D([], [], marker='None', color='tab:green', linewidth=5,label="EMSv2"),
                    plt.Line2D([], [], marker='None', color='tab:olive', linewidth=5,label="EMSv1"),
                    plt.Line2D([], [], marker='None', color='tab:orange', linewidth=5, label="TSS"),
                    plt.Line2D([], [], marker='None', color='tab:blue', linewidth=5, label="Sei")],
           bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0)
plt.ylabel('AUPRC')
plt.xlabel('Gene TPM')
plt.ylim(0, 0.3)
plt.savefig(f'./enformer_project/analysis/GTEx/AUC/TPM_bin/GTEx_auprc_TPM_bin_finalized.pdf', bbox_inches='tight')

