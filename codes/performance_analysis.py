import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


# Prediction accuracy of each feature
tissues = ['Whole_Blood', 'Muscle_Skeletal', 'Liver', 'Brain_Cerebellum','Prostate', 'Spleen', 'Skin_Sun_Exposed_Lower_leg', 'Artery_Coronary',
       'Esophagus_Muscularis', 'Esophagus_Gastroesophageal_Junction','Artery_Tibial', 'Heart_Atrial_Appendage', 'Nerve_Tibial',
       'Heart_Left_Ventricle', 'Adrenal_Gland', 'Adipose_Visceral_Omentum','Pancreas', 'Lung', 'Pituitary',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Colon_Transverse','Adipose_Subcutaneous', 'Esophagus_Mucosa', 'Brain_Cortex', 'Thyroid',
       'Stomach', 'Breast_Mammary_Tissue', 'Colon_Sigmoid','Skin_Not_Sun_Exposed_Suprapubic', 'Testis', 'Artery_Aorta',
       'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24','Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere',
       'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus','Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1',
       'Brain_Substantia_nigra', 'Cells_Cultured_fibroblasts','Cells_EBV-transformed_lymphocytes', 'Kidney_Cortex',
       'Minor_Salivary_Gland', 'Ovary', 'Small_Intestine_Terminal_Ileum','Uterus', 'Vagina']

scores = pd.DataFrame(columns = [])
for k in tissues:
    for j in glob.glob(f"./unique_xy/Output/keras/feature_selection/one_feature/*{k}*"):
        tmp1 = pd.read_csv(j, sep=',',index_col=0,header=0).T
        scores[k]= tmp1["aurocs"]
scores.sort_index(axis=1,ascending=True, inplace=True)
plt.figure(figsize = (90,25), dpi = 500)
sns.heatmap(scores.T, cmap='RdBu', vmax=1, vmin=0,center=0.5, square=True, xticklabels=True, yticklabels=True)
plt.title('a', loc='left', fontsize=100, fontweight='bold')

plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/feature_selection/GTEx_logistic_regression_feature_selection_AUROC.pdf', bbox_inches='tight')


for k in tissues:
    for j in glob.glob(f"./unique_xy/Output/keras/feature_selection/one_feature/*{k}*"):
        tmp1 = pd.read_csv(j, sep=',',index_col=0,header=0).T
        scores[k]= tmp1["auprcs"]
scores.sort_index(axis=1,ascending=True, inplace=True)
plt.figure(figsize = (90,25), dpi = 500)

sns.heatmap(scores.T, vmin=0, square=True, xticklabels=True, yticklabels=True)
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/feature_selection/GTEx_logistic_regression_feature_selection_AUPRC.pdf', bbox_inches='tight')



# Comparison of prediction accuracy by combination of features
tissue = "Whole_Blood" 
df = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_EMSv2_feature_selection_{tissue}.csv", delimiter=",") 

pos_n = sum(df["label"]==1)
neg_n = sum(df["label"]==0)
df_pos = df[df["label"]==1]
df_neg = df[df["label"]==0]
df_neg = df[df["label"]==0].sample(pos_n, random_state=42)
df = pd.concat([df_pos,df_neg]) 

#draw roc:
plt.figure(dpi = 500)
fpr, tpr, threshold = metrics.roc_curve(df["label"], df["all"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:brown', label = 'All features, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df["label"], df["enf_tss_tpm"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:purple', label = "EnformerPCs+TSS+Gene TPM, AUC = %0.4f" % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df["label"], df["enf_tss"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:red', label = 'EnformerPCs+TSS, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df["label"], df["tss"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:green', label = 'TSS, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df["label"], df["enf"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:orange', label = 'EnformerPCs, AUC = %0.4f' % roc_auc)

fpr, tpr, threshold = metrics.roc_curve(df["label"], df["tpm"])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='tab:blue', label = 'Gene TPM, AUC = %0.4f' % roc_auc)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.figure(dpi = 500)
precision, recall, thresholds = precision_recall_curve(df["label"], df["all"])
prc_auc = metrics.average_precision_score(df["label"], df["all"])
plt.plot(recall, precision, color='tab:brown', label = 'All features, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df["label"], df["enf_tss_tpm"])
prc_auc = metrics.average_precision_score(df["label"], df["enf_tss_tpm"])
plt.plot(recall, precision, color='tab:purple', label = "EnformerPCs+TSS+Gene TPM, AUC = %0.4f" % prc_auc)

precision, recall, thresholds = precision_recall_curve(df["label"], df["enf_tss"])
prc_auc = metrics.average_precision_score(df["label"], df["enf_tss"])
plt.plot(recall, precision, color='tab:red', label = 'EnformerPCs+TSS, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df["label"], df["tss"])
prc_auc = metrics.average_precision_score(df["label"], df["tss"])
plt.plot(recall, precision, color='tab:green', label = 'TSS, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df["label"], df["enf"])
prc_auc = metrics.average_precision_score(df["label"], df["enf"])
plt.plot(recall, precision, color='tab:orange', label = 'EnformerPCs, AUC = %0.4f' % prc_auc)

precision, recall, thresholds = precision_recall_curve(df["label"], df["tpm"])
prc_auc = metrics.average_precision_score(df["label"], df["tpm"])
plt.plot(recall, precision,color='tab:blue', label = 'Gene TPM, AUC = %0.4f' % prc_auc)

plt.xlim([0, 1])
plt.ylim([0, 0.35])
plt.xlabel('recall',fontsize=18)
plt.ylabel('precision',fontsize=18)
plt.legend()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)



df_auc_all = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_All_features.csv", delimiter=",",index_col=0).T
df_auc_enf_tss_tpm = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_Enformer_Tss_TPM.csv", delimiter=",",index_col=0).T 
df_auc_enf_tss = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_Enformer_Tss.csv", delimiter=",",index_col=0).T 
df_auc_tss = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_Tss_distance.csv", delimiter=",",index_col=0).T 
df_auc_enf = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_Enformer100PCs.csv", delimiter=",",index_col=0).T 
df_auc_tpm = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/feature_selection/GTEx_AUC_EMSv2_feature_selection_Gene_TPM.csv", delimiter=",",index_col=0).T 

df_auc_all["method"] = "All features"
df_auc_enf_tss_tpm["method"] = "EnformerPCs+Tss+Gene TPM"
df_auc_enf_tss["method"] = "EnformerPCs+Tss"
df_auc_tss["method"] = "Tss distance"
df_auc_enf["method"] = "EnformerPCs"
df_auc_tpm["method"] = "Gene TPM"

df_auc = pd.concat([df_auc_all, df_auc_enf_tss_tpm, df_auc_enf_tss,df_auc_tss, df_auc_enf, df_auc_tpm],axis=0)
df_auc.reset_index(inplace= True)
df_auc.columns = ["tissue","auroc","auprc","method"]
var_lst = ["auroc","auprc"]
df_auc[var_lst] = df_auc[var_lst].astype(float) 

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
order = ["Gene TPM", "EnformerPCs", "Tss distance", "EnformerPCs+Tss", "EnformerPCs+Tss+Gene TPM", "All features"]
palette = dict(zip(order, colors))  
xlabel_order = ["Gene TPM", "EnformerPCs", "Distance to TSS", "EnformerPCs+TSS", "EnformerPCs+TSS+Gene TPM", "All features"]

plt.figure(figsize=(9, 4), dpi = 500)
sns.violinplot(data=df_auc,x="method",y="auroc",order=order,palette=palette)
plt.xticks(rotation=70)
plt.ylim([0.5, 1])
plt.ylabel("AUROC")
plt.xlabel("Features")
plt.xticks(ticks=range(len(xlabel_order)), labels=xlabel_order, rotation=70)
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/feature_selection/GTEx_EMSv2_feature_selection_AUROC.png', bbox_inches='tight')

plt.figure(figsize=(9, 4), dpi = 500)
sns.violinplot(data=df_auc,x="method",y="auprc",cut=0,order=order,palette=palette)
plt.xticks(rotation=70)
plt.ylabel("AUPRC")
plt.xlabel("Features")
plt.xticks(ticks=range(len(xlabel_order)), labels=xlabel_order, rotation=70)
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/feature_selection/GTEx_EMSv2_feature_selection_AUPRC.png', bbox_inches='tight')



# Comparison of prediction accuracy across various PIP threshold values set for identifying causal variant
df_auc = pd.DataFrame()
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=500) 

for threshold in thresholds:
    df = pd.read_csv(f"./enformer_project/dataset/Output/EMSv2_development/pip_threshold/GTEx_AUC_EMSv2_pip_threshold_thres{threshold}.csv", sep=',',index_col=0).T
    df["pip_threshold"] = str(thres) + "<"
    df_auc = pd.concat([df_auc,df],axis=0)
df_auc.reset_index(inplace= True)
df_auc.columns = ["tissue","auroc","auprc","pip_threshold"]
var_lst = ["auroc","auprc"]
df_auc[var_lst] = df_auc[var_lst].astype(float) 
sns.violinplot(data=df_auc,x="pip_threshold",y="auprc", palette="pastel",cut=0, ax=axs[0])
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/GTEx_EMSv2_pip_threshold_AUPRC.png', bbox_inches='tight')

sns.violinplot(data=df_auc,x="pip_threshold",y="auroc", palette="pastel", ax=axs[1])
plt.ylim([0.8,1])
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/GTEx_EMSv2_pip_threshold_AUROC.png', bbox_inches='tight')

sns.violinplot(data=df_auc, x="pip_threshold", y="auprc", palette="pastel", cut=0, ax=axs[0])
axs[0].set_xlabel('PIP threshold in Loss function') 
axs[0].set_ylabel('AUPRC') 
axs[0].set_title('a', fontsize=30, fontweight='bold', loc='left')  

sns.violinplot(data=df_auc, x="pip_threshold", y="auroc", palette="pastel", ax=axs[1])
axs[1].set_xlabel('PIP threshold in Loss function') 
axs[1].set_ylabel('AUROC')  
axs[1].set_title('b', fontsize=30, fontweight='bold', loc='left')  

plt.subplots_adjust(hspace=0.4)  
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/GTEx_EMSv2_pip_threshold_AUC.png', bbox_inches='tight')



# Performance evaluation of multi-task learning in brain tissues
df_auc_all = pd.read_csv("./enformer_project/dataset/Output/EMSv2_development/brain_group/GTEx_AUC_EMSv2_PC100_tss.csv", sep=',',index_col=0).filter(like='Brain').T
df_auc_group = pd.read_csv("./enformer_project/dataset/Output/EMSv2_development/brain_group/GTEx_AUC_EMSv2_brain_group.csv", sep=',',index_col=0).filter(like='Brain').T
df_auc_single = pd.read_csv("./enformer_project/dataset/Output/EMSv2_development/brain_group/GTEx_AUC_EMSv2_brain_group_single_task.csv", sep=',',index_col=0).filter(like='Brain').T

tissues_sort_pos= ['Nerve_Tibial', 'Thyroid', 'Skin_Sun_Exposed_Lower_leg',
       'Cells_Cultured_fibroblasts', 'Artery_Tibial', 'Adipose_Subcutaneous',
       'Testis', 'Whole_Blood', 'Skin_Not_Sun_Exposed_Suprapubic',
       'Esophagus_Muscularis', 'Esophagus_Mucosa', 'Lung', 'Muscle_Skeletal',
       'Adipose_Visceral_Omentum', 'Artery_Aorta', 'Heart_Atrial_Appendage',
       'Colon_Transverse', 'Breast_Mammary_Tissue',
       'Esophagus_Gastroesophageal_Junction', 'Brain_Cerebellum', 'Pancreas',
       'Spleen', 'Colon_Sigmoid', 'Heart_Left_Ventricle', 'Stomach',
       'Brain_Cerebellar_Hemisphere', 'Brain_Cortex', 'Pituitary',
       'Adrenal_Gland', 'Brain_Caudate_basal_ganglia',
       'Brain_Nucleus_accumbens_basal_ganglia', 'Prostate',
       'Brain_Frontal_Cortex_BA9', 'Artery_Coronary', 'Ovary', 'Liver',
       'Brain_Putamen_basal_ganglia', 'Small_Intestine_Terminal_Ileum',
       'Cells_EBV-transformed_lymphocytes', 'Minor_Salivary_Gland',
       'Brain_Hypothalamus', 'Brain_Hippocampus',
       'Brain_Spinal_cord_cervical_c-1',
       'Brain_Anterior_cingulate_cortex_BA24', 'Uterus', 'Brain_Amygdala',
       'Vagina', 'Brain_Substantia_nigra', 'Kidney_Cortex']

brain_tissues_sort_pos = [s for s in tissues_sort_pos if 'Brain' in s]

df_auc_all = df_auc_all.loc[brain_tissues_sort_pos]
df_auc_group = df_auc_group.loc[brain_tissues_sort_pos]
df_auc_single = df_auc_single.loc[brain_tissues_sort_pos]

num = len(brain_tissues_sort_pos)
plt.figure(figsize=(9, 4), dpi = 400)
plt.scatter(np.arange(num)+0.1, df_auc_all["aurocs"],marker="o",color="tab:orange",label="Multi-task 49 Tissues", edgecolor='black', s=50)
plt.scatter(np.arange(num), df_auc_group["aurocs"],marker="^",color="tab:green",label="Multi-task Brain Tissues", edgecolor='black', s=50)
plt.scatter(np.arange(num)-0.1, df_auc_single["aurocs"],marker="s",color="tab:blue",label="Single-task Brain Tissues", edgecolor='black', s=40)
plt.ylabel('AUROC')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Training method')
plt.xticks(np.arange(num), brain_tissues_sort_pos,rotation=90)
plt.xlabel("Tissue (sorted by num. positives)")
plt.title("GTEx")
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/brain_group/GTEx_EMSv2_brain_group_AUROC.png', bbox_inches='tight')

plt.figure(figsize=(9, 4), dpi = 400)
plt.scatter(np.arange(num)+0.1, df_auc_all["auprcs"],marker="o",color="tab:orange",label="Multi-task 49 Tissues", edgecolor='black', s=50)
plt.scatter(np.arange(num), df_auc_group["auprcs"],marker="^",color="tab:green",label="Multi-task Brain Tissues", edgecolor='black', s=50)
plt.scatter(np.arange(num)-0.1, df_auc_single["auprcs"],marker="s",color="tab:blue",label="Single-task Brain Tissues", edgecolor='black', s=40)
plt.ylabel('AUPRC')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Training method')
plt.xticks(np.arange(num), brain_tissues_sort_pos,rotation=90)
plt.xlabel("Tissue (sorted by num. positives)")
plt.title("GTEx")
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/brain_group/GTEx_EMSv2_brain_group_AUPRC.png', bbox_inches='tight')



# Correlation between each feature
df_yx_train = pd.read_csv("./unique_xy/Input/yx_train_pcs_gene.csv.gz", delimiter=",") 
plt.figure(figsize = (50,35), dpi = 500)
sns.heatmap(df_yx_train.iloc[:,51:].corr(method="pearson"), cmap='RdBu', vmax=1, vmin=-1, center=0, square=True, xticklabels=True, yticklabels=True)
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/feature_selection/GTEx_feature_correlation.pdf', bbox_inches='tight')


# Hyperparameter tuning in EMSv2
int_units=[32,64,128,256,512,1024]
hid_units=[16,32,64,128,256,512]
dropout_rate=[0.2,0.5]

aurocs=[]
auprcs=[]
archit=[]
for i in int_units: 
    for j in hid_units:
      for k in dropout_rate:
          for m in glob.glob(f"./unique_xy/Output/keras/multi_task/learning_curve/PC100_tss/PC100multi_scores_Custom{i}_{j}_{k}.csv"):
            tmp = pd.read_csv(m, sep=',',index_col=0,header=None).T
            tmp["aurocs"]=tmp["aurocs"].astype(float)
            tmp["auprcs"]=tmp["auprcs"].astype(float)
            aurocs.append(tmp["aurocs"].mean())
            auprcs.append(tmp["auprcs"].mean())
            archit.append(f"int{i}_drop{k}_hid{j}_drop{k}_out49")

scores = pd.DataFrame({"Architecture": archit, "AUROC": aurocs,"AUPRC": auprcs})

plt.figure(figsize=(7, 7), dpi = 500)
x = scores["AUROC"].values
y = scores["AUPRC"].values
xmax = np.max(x)
x_max_index = np.where(x == xmax)[0][0]
ymax = np.max(y)
y_max_index = np.where(y == ymax)[0][0]

plt.scatter(scores["AUROC"], scores["AUPRC"], s=50)
plt.xlabel("AUROC")
plt.ylabel("AUPRC")

plt.axvline(x=xmax, linestyle="--", linewidth=0.5, color="black")
plt.axhline(y=scores["AUPRC"][scores["AUROC"] ==xmax].values, linestyle="--", linewidth=0.5, color="black")
plt.scatter(x[x_max_index], y[x_max_index], s=200, c='red', edgecolors='black')
plt.savefig(f'./enformer_project/analysis/GTEx/EMSv2_development/model_architecture/EMSv2_architecture_PC100_tss.png')

