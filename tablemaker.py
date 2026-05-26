from wartermarks.tablemaker import main_roc, make_table
import fire, sys


# RAR row in main table for no clusters + token prediction
# python tablemaker.py --files experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_tokpred_evalfirst2000_perturb=full.json,experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_tokpred_evalfirst2000_perturb=regen.json


SPEC_MAIN_TABLE_OURS = [
    "\midrule",
    "\multicolumn{10}{c}{LlamaGen (GPT-B)}\\\\",
    "\midrule",
    ("Ours (No Clustering)", None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",         "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    (" + Token Pred",        None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_tokpred_evalfirst2000_perturb=fullchallenge.json",         "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_tokpred_evalfirst2000_perturb=regen.json"),
    ("Ours (Clustering, k=64)", None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",        "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    ("+ CPN (k=64)",            None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_clpred_evalfirst2000_perturb=fullchallenge.json", "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-B_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_clpred_evalfirst2000_perturb=allregen.json"),
    "\midrule",
    "\multicolumn{10}{c}{LlamaGen (GPT-L)}\\\\",
    "\midrule",
    ("Ours (No Clustering)", None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",         "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    (" + Token Pred",        None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_tokpred_evalfirst2000_perturb=fullchallenge.json",         "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters0_penalty5_greenfrac=0.25_wmseedprefix=4/roc_summary_v3_tokpred_evalfirst2000_perturb=regen.json"),
    ("Ours (Clustering, k=64)", None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",        "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    ("+ CPN (k=64)",            None, "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_clpred_evalfirst2000_perturb=fullchallenge.json", "/USERSPACE/lukovdg1/LlamaGen/experiments_v2.2/gen_wm_pp_v2.2_2000samples_GPT-L_c2i_wm_clusters64_penalty5_greenfrac=0.25_wmseedprefix=7/roc_summary_v3_clpred_evalfirst2000_perturb=allregen.json"),
    "\midrule",
    "\multicolumn{10}{c}{RAR-XL}\\\\",
    "\midrule",
    ("Ours (No Clustering)", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/cleanfid-256.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",         "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    (" + Token Pred",        "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1/gen_wm_v1_50000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/cleanfid-256.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_tokpred_evalfirst2000_perturb=fullchallenge.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_0clusters_greenfrac0.25_penalty5_prefix2/roc_summary_v3_tokpred_evalfirst2000_perturb=regen.json"),
    ("Ours (Clustering, k=64)", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1/gen_wm_v1_50000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/cleanfid-256.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/roc_summary_v3_evalfirst2000_perturb=fullchallenge.json",        "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/roc_summary_v3_evalfirst2000_perturb=allregen.json"),
    ("+ CPN (k=64)",            "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1/gen_wm_v1_50000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/cleanfid-256.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/roc_summary_v3_clpred_evalfirst2000_perturb=fullchallenge.json", "/USERSPACE/lukovdg1/1d-tokenizer/experiments_v1_prefixes/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.25_penalty5_prefix3/roc_summary_v3_clpred_evalfirst2000_perturb=allregen.json")
]

def _make_table(
        tablespec:str=SPEC_MAIN_TABLE_OURS,
         tprfield="tpr@fpr=1%",
         aucfield="roc_auc",
         fprfield="actual_fpr",
         tablecolumns="none|jpeg_ratio|gaussian_blur_r|gaussian_std_fixed|sp_prob_fixed|random_drop_ratio|AVG(brightness_factor,contrast_factor,hue_factor,saturation_factor)|AVG(ctrl_sd21.60,ae_sd15,ae_flux1)",
    ):
    table = make_table(
        tablespec=tablespec,
        tprfield=tprfield,
        aucfield=aucfield,
        fprfield=fprfield,
        tablecolumns=tablecolumns,
    )
    print(table)

if __name__ == "__main__":
    fire.Fire(_make_table)