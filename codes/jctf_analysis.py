import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import precision_recall_curve


# Performance evaluation of EMSv2 in a Japanese population
#1 EMSv2
df_emsv2_pos= pd.read_csv("./enformer_project/dataset/Output/Taskforce_EMSv2_PC100_tss_pos.csv.gz", delimiter=",",index_col=0) 
df_emsv2_neg= pd.read_csv("./enformer_project/dataset/Output/Taskforce_EMSv2_PC100_tss_neg.csv.gz", delimiter=",",index_col=0) 
df_emsv2_pos["label"] = 1
df_emsv2_neg["label"] = 0
df_emsv2 = pd.concat([df_emsv2_pos,df_emsv2_neg],axis=0)

#2 Distance to TSS
df_tss_pos= pd.read_csv("./enformer_project/dataset/Output/Taskforce_Logistic_Tss_distance_pos.csv.gz", delimiter=",",index_col=0) 
df_tss_neg= pd.read_csv("./enformer_project/dataset/Output/Taskforce_Logistic_Tss_distance_neg.csv.gz", delimiter=",",index_col=0) 
df_tss_pos["label"] = 1
df_tss_neg["label"] = 0
df_tss = pd.concat([df_tss_pos,df_tss_neg],axis=0)

#3 Sei
df_sei_pos= pd.read_csv("./enformer_project/dataset/Output/Sei_sequence_class_scores/sorted.taskforce_pos.sequence_class_scores.tsv", delimiter="\t") 
df_sei_neg= pd.read_csv("./enformer_project/dataset/Output/Sei_sequence_class_scores/sorted.taskforce_neg.sequence_class_scores.tsv", delimiter="\t") 
df_sei_pos["label"] = 1
df_sei_neg["label"] = 0
df_sei = pd.concat([df_sei_pos,df_sei_neg],axis=0)

# AUROC
plt.figure(dpi=400)
fpr, tpr, threshold = metrics.roc_curve(df_emsv2["label"], df_emsv2["Whole_Blood"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:green', label = 'EMSv2, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df_tss["label"], df_tss["Whole_Blood"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:orange', label = 'TSS, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df_sei["label"], df_sei["seqclass_max_absdiff"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:blue', label = 'Sei, AUC = %0.4f' % roc_auc)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig(f'./enformer_project/analysis/Japan_COVID-19_Task_Force/AUC/Taskforce_AUROC.pdf', bbox_inches='tight')

# AUPRC
plt.figure(dpi=400)
precision, recall, thresholds = precision_recall_curve(df_emsv2["label"], df_emsv2["Whole_Blood"])
prc_auc = metrics.average_precision_score(df_emsv2["label"], df_emsv2["Whole_Blood"])
plt.plot(recall, precision, color='tab:green', label = 'EMSv2, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df_tss["label"], df_tss["Whole_Blood"])
prc_auc = metrics.average_precision_score(df_tss["label"], df_tss["Whole_Blood"])
plt.plot(recall, precision, color='tab:orange', label = 'TSS, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df_sei["label"], df_sei["seqclass_max_absdiff"])
prc_auc = metrics.average_precision_score(df_sei["label"], df_sei["seqclass_max_absdiff"])
plt.plot(recall, precision, color='tab:blue', label= 'Sei, AUC = %0.4f' % prc_auc)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper left', bbox_to_anchor=(-0.015, 0.27))
plt.savefig(f'./enformer_project/analysis/Japan_COVID-19_Task_Force/AUC/Taskforce_AUPRC.pdf', bbox_inches='tight')



# Functionally-informed fine-mapping with EMSv2 as a prior
# Taskforce all variant-gene
pip_unif = "pip_susie" 
pip_ems = "pip_updated_v2"

df_tf = pd.read_csv("./enformer_project/dataset/Input/taskforce_n465_susiepip_updated_wbprior_forshare_v2_20230313.tsv.gz", delimiter="\t",index_col=None) 
df_tf["id"] = df_tf["id"].str.replace(':', '_') 
raw_mpra = pd.read_csv(f"./unique_xy/Input/validation_data/K562_pvals.tsv", delimiter="\t") 
raw_mpra.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df =pd.merge(df_tf, raw_mpra, how="left",on="id") 
df = df.dropna(subset="ensg_id")
df = df.dropna(subset=pip_ems) # dropna

x_bins = np.arange(0, 1.1, 0.1)
x_bins[0]=-1
x_bins[-1]=1.1
y_bins = np.arange(0, 1.1, 0.1)
y_bins[0]=-1
y_bins[-1]=1.1
index = ["<0.1", "(0.1, 0.2]", "(0.2, 0.3]", "(0.3, 0.4]", "(0.4, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]", "0.9<"]
columns = ["<0.1", "(0.1, 0.2]", "(0.2, 0.3]", "(0.3, 0.4]", "(0.4, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]", "0.9<"]

data =pd.DataFrame(np.histogram2d(df[pip_ems],df[pip_unif], bins=[x_bins, y_bins])[0],
            index=index, columns=columns)
data = data.astype('int')

# Heatmap
plt.figure(figsize=(12,12), dpi = 500)
symlog_norm  = SymLogNorm(linthresh=50,vmin=-1, vmax=40000) 
log_norm = LogNorm(vmin=1, vmax=3000) 
sns.heatmap((data.T), cmap="viridis", cbar_kws={'label': 'count', "shrink": 0.8}, norm=symlog_norm, annot=True, fmt='d',square=True, linecolor="white", linewidths=.1)
plt.xlabel("PIP using EMS as a prior (PIP$_{EMS}$)", fontsize=20)
plt.ylabel("PIP using uniform prior (PIP$_{unif}$)", fontsize=20)
plt.yticks(rotation=20, fontsize=16)
plt.xticks(rotation=20, fontsize=16)
plt.savefig(f'./enformer_project/analysis/Japan_COVID-19_Task_Force/Taskforce_{pip_ems}_{pip_unif}_heatmap.pdf', bbox_inches='tight')
