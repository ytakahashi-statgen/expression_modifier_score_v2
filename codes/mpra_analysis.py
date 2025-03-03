import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Comparison of EMSv2 and Sei enrichment using massively parallel reporter assay results
def mpra_enrichment(df,score,bins):
    tmp1 = pd.cut(df[score][df["emvar"]=="Tier 1"], bins=bins, include_lowest = True, right=True).value_counts()
    tmp1 = pd.DataFrame(tmp1).sort_index()

    tmp2 = pd.cut(df[score][df["emvar"]=="Tier 2"], bins=bins, include_lowest = True, right=True).value_counts()
    tmp2 = pd.DataFrame(tmp2).sort_index()

    tmp3 = pd.cut(df[score][df["emvar"]=="None"], bins=bins, include_lowest = True, right=True).value_counts()
    tmp3 = pd.DataFrame(tmp3).sort_index()

    tmp4 = pd.DataFrame()
    tmp4["n1(Tier 1)"] = tmp1
    tmp4["n2(Tier 2)"] = tmp2
    tmp4["n3(None)"] = tmp3
    tmp4["num"] = tmp4.sum(axis=1)

    tmp4["n1_enr"] = tmp4["n1(Tier 1)"]/ tmp4["num"]
    tmp4["n2_enr"] = tmp4["n2(Tier 2)"]/ tmp4["num"]
    tmp4["n3_enr"] = tmp4["n3(None)"]/ tmp4["num"]

    tmp4["n1_err"] = np.sqrt(tmp4["n1_enr"]*(1-tmp4["n1_enr"]) / tmp4["num"])
    tmp4["n2_err"] = np.sqrt(tmp4["n2_enr"]*(1-tmp4["n2_enr"]) / tmp4["num"])
    tmp4["n3_err"] = np.sqrt(tmp4["n3_enr"]*(1-tmp4["n3_enr"]) / tmp4["num"])

    tmp4["enrichment(Tier1)"] = tmp4["n1_enr"] / (tmp4["n1(Tier 1)"].sum() / tmp4["num"].sum())
    tmp4["enrichment(Tier2)"] = tmp4["n2_enr"] / (tmp4["n2(Tier 2)"].sum() / tmp4["num"].sum())
    tmp4["enrichment(None)"] = tmp4["n3_enr"] / (tmp4["n3(None)"].sum() / tmp4["num"].sum())

    tmp4["Standard_error(Tier1)"] = tmp4["n1_err"] / (tmp4["n1(Tier 1)"].sum() / tmp4["num"].sum())
    tmp4["Standard_error(Tier2)"] = tmp4["n2_err"] / (tmp4["n2(Tier 2)"].sum() / tmp4["num"].sum())
    tmp4["Standard_error(None)"] = tmp4["n3_err"] / (tmp4["n3(None)"].sum() / tmp4["num"].sum())
    
    return tmp4

# K562, EMSv2 Whole Blood
df_ems = pd.read_csv("./enformer_project/dataset/Output/MPRA_K562_EMSv2_scaling_tss200bp.csv.gz", delimiter=",",index_col=0)
df_k562 = pd.read_csv(f"./unique_xy/Input/validation_data/K562_pvals.tsv", delimiter="\t",keep_default_na=False) 
df_k562.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_k562.drop_duplicates(subset="id",keep='first',inplace=True)
df_ems_k562 = pd.merge(df_ems, df_k562, how="left",on="id")

num=4
sr_cut, bins = pd.qcut(df_ems_k562["Whole_Blood_EMSv2"], num, retbins=True) 
bins[0]=0
df_ems_k562_enr = mpra_enrichment(df_ems_k562,"Whole_Blood_EMSv2",bins)

df_sei = pd.read_csv(f"./unique_xy/Input/validation_data/Sei/sorted.mpra_K562.sequence_class_scores.tsv", delimiter="\t") 
df_sei.rename(columns={'name': 'id'},inplace=True)

# MPRA K562 pvals 読み込み
df_k562 = pd.read_csv(f"./unique_xy/Input/validation_data/K562_pvals.tsv", delimiter="\t",keep_default_na=False) 
df_k562.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_k562.drop_duplicates(subset="id",keep='first',inplace=True)
df_sei_k562 = pd.merge(df_sei, df_k562, how="left",on="id")

# score bins設定
sr_cut, bins = pd.qcut(df_sei_k562["seqclass_max_absdiff"], num, retbins=True) 

df_sei_k562_enr = mpra_enrichment(df_sei_k562,"seqclass_max_absdiff",bins)

# plot
plt.figure(figsize=(8, 4), dpi = 400)
plt.errorbar(np.arange(num)+0.25, df_ems_k562_enr["enrichment(Tier1)"], df_ems_k562_enr["Standard_error(Tier1)"], fmt="o",color="xkcd:red", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)+0.2, df_sei_k562_enr["enrichment(Tier1)"], df_sei_k562_enr["Standard_error(Tier1)"], fmt="^",color="xkcd:red", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)+0.05, df_ems_k562_enr["enrichment(Tier2)"], df_ems_k562_enr["Standard_error(Tier2)"], fmt="o",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num), df_sei_k562_enr["enrichment(Tier2)"], df_sei_k562_enr["Standard_error(Tier2)"], fmt="^",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)-0.2, df_ems_k562_enr["enrichment(None)"], df_ems_k562_enr["Standard_error(None)"], fmt="o",color="grey", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)-0.25, df_sei_k562_enr["enrichment(None)"], df_sei_k562_enr["Standard_error(None)"], fmt="^",color="grey", markeredgecolor='black', markeredgewidth=0.5)
plt.xticks([], labels=[])
plt.xlabel('Score(quartile)')
plt.legend(["Tier1(EMSv2)", "Tier1(Sei)","Tier2(EMSv2)","Tier2(Sei)", "None(EMSv2)","None(Sei)"],loc='upper left', bbox_to_anchor=(0, 1))
plt.ylabel('Enrichment')
plt.savefig(f'./enformer_project/analysis/MPRA_enrichment/K562/MPRA_K562_enrichment.pdf', bbox_inches='tight')

# HepG2, EMSv2 Liver
df_ems = pd.read_csv("./enformer_project/dataset/Output/MPRA_HepG2_EMSv2_scaling_tss200bp.csv.gz", delimiter=",",index_col=0)
df_hepg2 = pd.read_csv(f"./unique_xy/Input/validation_data/HepG2_pvals.tsv", delimiter="\t",keep_default_na=False) 
df_hepg2.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_hepg2.drop_duplicates(subset="id",keep='first',inplace=True)
df_ems_hepg2 = pd.merge(df_ems, df_hepg2, how="left",on="id")

num=4
sr_cut, bins = pd.qcut(df_ems_hepg2["Liver_EMSv2"], num, retbins=True) 
bins[0]=0
df_ems_hepg2_enr = mpra_enrichment(df_ems_hepg2,"Liver_EMSv2",bins)

df_sei = pd.read_csv(f"./unique_xy/Input/validation_data/Sei/sorted.mpra_HepG2.sequence_class_scores.tsv", delimiter="\t") 
df_sei.rename(columns={'name': 'id'},inplace=True)

# MPRA K562 pvals 読み込み
df_hepg2 = pd.read_csv(f"./unique_xy/Input/validation_data/HepG2_pvals.tsv", delimiter="\t",keep_default_na=False) 
df_hepg2.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df_hepg2.drop_duplicates(subset="id",keep='first',inplace=True)
df_sei_hepg2 = pd.merge(df_sei, df_hepg2, how="left",on="id")

# score bins設定
sr_cut, bins = pd.qcut(df_sei_hepg2["seqclass_max_absdiff"], num, retbins=True) 

df_sei_hepg2_enr = mpra_enrichment(df_sei_hepg2,"seqclass_max_absdiff",bins)

# plot
plt.figure(figsize=(8, 4), dpi = 400)
plt.errorbar(np.arange(num)+0.25, df_ems_hepg2_enr["enrichment(Tier1)"], df_ems_hepg2_enr["Standard_error(Tier1)"], fmt="o",color="xkcd:red", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)+0.2, df_sei_hepg2_enr["enrichment(Tier1)"], df_sei_hepg2_enr["Standard_error(Tier1)"], fmt="^",color="xkcd:red", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)+0.05, df_ems_hepg2_enr["enrichment(Tier2)"], df_ems_hepg2_enr["Standard_error(Tier2)"], fmt="o",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num), df_sei_hepg2_enr["enrichment(Tier2)"], df_sei_hepg2_enr["Standard_error(Tier2)"], fmt="^",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)-0.2, df_ems_hepg2_enr["enrichment(None)"], df_ems_hepg2_enr["Standard_error(None)"], fmt="o",color="grey", markeredgecolor='black', markeredgewidth=0.5)
plt.errorbar(np.arange(num)-0.25, df_sei_hepg2_enr["enrichment(None)"], df_sei_hepg2_enr["Standard_error(None)"], fmt="^",color="grey", markeredgecolor='black', markeredgewidth=0.5)
plt.xticks([], labels=[])
plt.xlabel('Score(quartile)')
plt.legend(["Tier1(EMSv2)", "Tier1(Sei)","Tier2(EMSv2)","Tier2(Sei)", "None(EMSv2)","None(Sei)"],loc='upper left', bbox_to_anchor=(0, 1))
plt.ylabel('Enrichment')
plt.savefig(f'./enformer_project/analysis/MPRA_enrichment/HepG2/MPRA_HepG2_enrichment.pdf', bbox_inches='tight')


# enrichment of Tier 1 and Tier 2 MPRA hits
cell = "K562" 
pip_unif = "pip_susie" 
pip_ems = "pip_updated_v2"

df_tf = pd.read_csv("./enformer_project/dataset/Input/taskforce_n465_susiepip_updated_wbprior_forshare_v2_20230313.tsv.gz", delimiter="\t",index_col=None) 
df_tf["id"] = df_tf["id"].str.replace(':', '_') 
raw_mpra = pd.read_csv(f"./unique_xy/Input/validation_data/{cell}_pvals.tsv", delimiter="\t",keep_default_na=False) 
raw_mpra.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df =pd.merge(raw_mpra, df_tf, how="left",on="id")
df = df.dropna(subset="ensg_id")
df = df.dropna(subset=pip_unif) # dropna
df = df.sort_values(["id",pip_unif],ascending=False)

data_df = []
for i in range(1,10):
    df_up = df[df[pip_ems] - df[pip_unif]>i*0.1].sort_values(pip_ems)
    data_df.append(df_up["emvar"].value_counts())
    
    df_inter = df[(df[pip_ems] - df[pip_unif]<=i*0.1) & (df[pip_unif] - df[pip_ems]<=i*0.1)].sort_values(pip_ems)
    data_df.append(df_inter["emvar"].value_counts())
    
    df_down= df[df[pip_unif] - df[pip_ems]>i*0.1].sort_values(pip_ems)
    data_df.append(df_down["emvar"].value_counts())

index = ["PIPems - PIPunif > 0.1","-0.1 <= PIPems - PIPunif <= 0.1","PIPems - PIPunif < -0.1",
         "PIPems - PIPunif > 0.2","-0.2 <= PIPems - PIPunif <= 0.2","PIPems - PIPunif < -0.2",
         "PIPems - PIPunif > 0.3","-0.3 <= PIPems - PIPunif <= 0.3","PIPems - PIPunif < -0.3",
         "PIPems - PIPunif > 0.4","-0.4 <= PIPems - PIPunif <= 0.4","PIPems - PIPunif < -0.4",
         "PIPems - PIPunif > 0.5","-0.5 <= PIPems - PIPunif <= 0.5","PIPems - PIPunif < -0.5",
         "PIPems - PIPunif > 0.6","-0.6 <= PIPems - PIPunif <= 0.6","PIPems - PIPunif < -0.6",
         "PIPems - PIPunif > 0.7","-0.7 <= PIPems - PIPunif <= 0.7","PIPems - PIPunif < -0.7",
         "PIPems - PIPunif > 0.8","-0.8 <= PIPems - PIPunif <= 0.8","PIPems - PIPunif < -0.8",
         "PIPems - PIPunif > 0.9","-0.9 <= PIPems - PIPunif <= 0.9","PIPems - PIPunif < -0.9"]
tmp1 = pd.DataFrame(data_df,index=index)
tmp1.columns = ["None_n","Tier2_n","Tier1_n"]
tmp1 = tmp1.fillna(0)
tmp1 = tmp1.astype("int")

data_df = []
for i in range(1,10):
    df_up = df[df[pip_ems] - df[pip_unif]>i*0.1].sort_values(pip_ems)
    data_df.append(df_up["emvar"].value_counts(normalize=True)*100)
    
    df_inter = df[(df[pip_ems] - df[pip_unif]<=i*0.1) & (df[pip_unif] - df[pip_ems]<=i*0.1)].sort_values(pip_ems)
    data_df.append(df_inter["emvar"].value_counts(normalize=True)*100)
    
    df_down= df[df[pip_unif] - df[pip_ems]>i*0.1].sort_values(pip_ems)
    data_df.append(df_down["emvar"].value_counts(normalize=True)*100)

tmp2 = pd.DataFrame(data_df,index=index)
tmp2.columns = ["None_%","Tier2_%","Tier1_%"]
tmp2 = tmp2.fillna(0)

tmp = pd.concat([tmp1,tmp2],axis=1)

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
for i in range(1,6):
    df_up = df[df[pip_ems] - df[pip_unif]>i*0.1].sort_values(pip_ems)
    data1.append(df_up["emvar"].value_counts())
    data2.append(df_up["emvar"].value_counts(normalize=True)*100)

    df_inter = df[(df[pip_ems] - df[pip_unif]<=i*0.1) & (df[pip_unif] - df[pip_ems]<=i*0.1)].sort_values(pip_ems)
    data3.append(df_inter["emvar"].value_counts())
    data4.append(df_inter["emvar"].value_counts(normalize=True)*100)
    
    df_down= df[df[pip_unif] - df[pip_ems]>i*0.1].sort_values(pip_ems)
    data5.append(df_down["emvar"].value_counts())
    data6.append(df_down["emvar"].value_counts(normalize=True)*100)
    
tmp1 = pd.DataFrame(data1)
tmp1 = tmp1.fillna(0)
tmp1 = tmp1.astype("int")

tmp2 = pd.DataFrame(data2)
tmp2 = tmp2.fillna(0)

tmp3 = pd.DataFrame(data3)
tmp3 = tmp3.fillna(0)
tmp3 = tmp3.astype("int")

tmp4 = pd.DataFrame(data4)
tmp4 = tmp4.fillna(0)

tmp5 = pd.DataFrame(data5)
tmp5 = tmp5.fillna(0)
tmp5 = tmp5.astype("int")

tmp6 = pd.DataFrame(data6)
tmp6 = tmp6.fillna(0)

tmp1 = tmp1.reset_index(drop=True)
tmp2 = tmp2.reset_index(drop=True)
tmp3 = tmp3.reset_index(drop=True)
tmp4 = tmp4.reset_index(drop=True)
tmp5 = tmp5.reset_index(drop=True)
tmp6 = tmp6.reset_index(drop=True)

df_ems_unif = pd.concat([tmp1,tmp2,tmp3,tmp4,tmp5,tmp6],axis=1)

f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, gridspec_kw=dict(height_ratios=[3,3,3]),figsize=(7, 5),dpi=400)
num=5
ax1.errorbar(np.arange(num)+0.2, df_ems_unif.iloc[:,5], fmt="o",color="tab:red", markeredgecolor='black', markeredgewidth=0.5)
ax1.errorbar(np.arange(num), df_ems_unif.iloc[:,11], fmt="^",color="tab:red", markeredgecolor='black', markeredgewidth=0.5)
ax1.errorbar(np.arange(num)-0.2, df_ems_unif.iloc[:,17], fmt="X",color="tab:red", markeredgecolor='black', markeredgewidth=0.5)
ax2.errorbar(np.arange(num)+0.2, df_ems_unif.iloc[:,4], fmt="o",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
ax2.errorbar(np.arange(num), df_ems_unif.iloc[:,10], fmt="^",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
ax2.errorbar(np.arange(num)-0.2, df_ems_unif.iloc[:,16], fmt="X",color="tab:orange", markeredgecolor='black', markeredgewidth=0.5)
ax3.errorbar(np.arange(num)+0.2, df_ems_unif.iloc[:,3], fmt="o",color="grey", markeredgecolor='black', markeredgewidth=0.5)
ax3.errorbar(np.arange(num), df_ems_unif.iloc[:,9], fmt="^",color="grey", markeredgecolor='black', markeredgewidth=0.5)
ax3.errorbar(np.arange(num)-0.2, df_ems_unif.iloc[:,15], fmt="X",color="grey", markeredgecolor='black', markeredgewidth=0.5)

ax1.set_ylabel("%",rotation=0, ha='right', va='center', multialignment='center')
ax2.set_ylabel("%",rotation=0, ha='right', va='center', multialignment='center')
ax3.set_ylabel("%",rotation=0, ha='right', va='center', multialignment='center')

ax1.set_ylim([-0.5,3.5])
ax2.set_ylim([-0,4])
ax3.set_ylim([92,100])
label1 = "PIP$_{EMS}$ - PIP$_{unif}$ > Threshold"
label2 = "PIP$_{EMS}$ - PIP$_{unif}$ <= ±Threshold"
label3 = "PIP$_{unif}$ - PIP$_{EMS}$ > Threshold"

plt.legend(handles=[plt.Line2D([], [],linestyle='None', marker='o', color='black', label=label1),
                     plt.Line2D([], [], linestyle='None',marker='^', color='black', label=label2),
                    plt.Line2D([], [], linestyle='None',marker='X', color='black', label=label3),
                    plt.Line2D([], [], marker='None', color='tab:red', linewidth=5,label="Tier1"),
                    plt.Line2D([], [], marker='None', color='tab:orange', linewidth=5, label="Tier2"),
                    plt.Line2D([], [], marker='None', color='grey', linewidth=5, label="None")], loc='upper left', bbox_to_anchor=(1, 3.5))
plt.xticks(np.arange(num), ["0.1", "0.2","0.3","0.4","0.5"])
ax3.set_xlabel("Threshold")

plt.savefig(f'./enformer_project/analysis/MPRA_enrichment/K562/PIP_enrhichment/MPRA_K562_{pip_ems}_{pip_unif}_%.pdf', bbox_inches='tight')



# Examples of putative causal eQTL prioritized by EMSv2
def putative_causal_eqtl(df,df_tss,ensg_id):
    df1 = df["id"].str.split('_', expand=True).iloc[:,:2]
    df1.columns=['chr','locus']
    df1 =  pd.concat([df1,df],axis=1)
    df1["locus"]=df1["locus"].astype(int)
    
    df_tss["ensg_id"] = df_tss["ensg_id"].str.split('.', expand=True)[0]
    tss = df_tss["tss_position"][df_tss["ensg_id"] == ensg_id]
    
    df_plot = df1[(df1["ensg_id"]==ensg_id) & (df1["locus"] <= int(tss)+1e6) & (df1["locus"] >= int(tss)-1e6)]
    df_plot["tss_dist"]  = (df_plot["locus"] - int(tss))

    return df_plot


df_tf = pd.read_csv("./enformer_project/dataset/Input/taskforce_n465_susiepip_updated_wbprior_forshare_v2_20230313.tsv.gz", delimiter="\t",index_col=None) 
# All variant-gene
pip_unif = "pip_susie"
pip_ems = "pip_updated_v2"

df_tf = pd.read_csv("./enformer_project/dataset/Input/taskforce_n465_susiepip_updated_wbprior_forshare_v2_20230313.tsv.gz", delimiter="\t",index_col=None) 
df_tf["id"] = df_tf["id"].str.replace(':', '_')
raw_mpra = pd.read_csv(f"./unique_xy/Input/validation_data/K562_pvals.tsv", delimiter="\t", keep_default_na=False) 
raw_mpra.columns=['id','pval', 'alpha_diff', 'idx', 'bh_value_001','bh_value_01','emvar']
df =pd.merge(df_tf, raw_mpra, how="left",on="id")
df = df.dropna(subset="ensg_id")
df = df.dropna(subset=pip_ems) # dropna
df["ensg_id"] = df["ensg_id"].str.split('.', expand=True)[0]

df_tss = pd.read_csv("./enformer_project/tss_distance_annot/gencode.v26.GRCh38.genes.tssposition.tsv", sep='\t')
df_enf = pd.read_csv("./enformer_project/dataset/Input/enformer_mpra_taskpip01_All_scores.csv.gz", delimiter=",",index_col=0)

ensg_id = "ENSG00000198879" #SFMBT2
df_plot = putative_causal_eqtl(df,df_tss,ensg_id)
df_mpra = df_plot.loc[:, ("id", "ensg_id","tss_dist","emvar")].dropna(subset="emvar")

ems_top_idx = df_plot[pip_ems][df_plot['emvar'] == 'Tier 1'].idxmax() 
var_id = df_plot.loc[df_plot['emvar'] == 'Tier 1', 'id'].loc[ems_top_idx]
enf_top_f = df_enf[df_enf["id"]==var_id].iloc[:,5:].abs().idxmax(axis=1).to_string(index=False)
df_plot = pd.merge(df_plot,df_enf[["id",enf_top_f]],how="left",on="id")

#plot
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharex=True, gridspec_kw=dict(height_ratios=[3,3,3,4,1]),figsize=(6, 6),dpi=300)
colors = np.where(df_mpra["emvar"] == "Tier 1", 'red', 'grey') # Tier1を赤，それ以外は黒．
gene_name = df_tss["gene_name"][df_tss["ensg_id"] == ensg_id].to_string(index=False)
x_line = df_plot.loc[df_plot['id']==var_id, 'tss_dist'].values[0]

ax1.scatter(df_plot["tss_dist"], df_plot[pip_unif], color="blue", label="pip")
ax2.scatter(df_plot["tss_dist"], df_plot["Whole_Blood"], color="green", label="EMS")
ax3.scatter(df_plot["tss_dist"], df_plot[pip_ems], color="orange")
ax4.scatter(df_plot["tss_dist"], df_plot[enf_top_f], color="black")
ax5.scatter(df_mpra["tss_dist"][colors == 'grey'], np.zeros_like(df_mpra["tss_dist"])[colors == 'grey'], c='grey')
ax5.scatter(df_mpra["tss_dist"][colors == 'red'], np.zeros_like(df_mpra["tss_dist"])[colors == 'red'], c='red') #Tier1赤が先頭に来るように．

ax1.set_ylabel("PIP$_{unif}$",rotation=0, ha='right', va='center', multialignment='center')
ax2.set_ylabel("EMS\n(Whole Blood)",rotation=0, ha='right', va='center', multialignment='center')
ax3.set_ylabel("PIP$_{EMS}$",rotation=0, ha='right', va='center', multialignment='center')
ax4.set_ylabel("Enformer score\n(CAGE : CD14+ monocytes)",rotation=0, ha='right', va='center', multialignment='center')
ax5.set_ylabel("MPRA(K562)",rotation=0, ha='right', va='center')
ax5.set_xlabel(fr"Distance to TSS of ${gene_name}$")

ax1.set_ylim([-0.02,1.05])
ax2.set_ylim([-0.00001,0.0003])
ax3.set_ylim([-0.02,1.05])
ax4.set_ylim([-100,2200])
ax5.set_ylim([-0.5, 0.5])
ax5.set_yticks([]) 
ax5.set_xlim([-235000, -225000])
ax1.text(x_line, 1.3, "rs61835060", ha='center', va='center') 
ax1.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax2.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax3.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax4.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
ax5.axvline(x=x_line, linestyle="--", linewidth=0.5, color="black")
plt.subplots_adjust(hspace=0.3)
f.savefig(f'./enformer_project/analysis/MPRA_enrichment/putative_causal_eQTL/MPRA_Taskforce_causal_eQTL_{gene_name}.pdf', bbox_inches='tight')

