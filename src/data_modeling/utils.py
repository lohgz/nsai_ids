import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def make_output_dirs(base_path: str, split_tag: str, run: int):
    """
    Creates the following under data/output/model/{split_tag}_run_{run}/:
      - tables
      - ckpts
      - plts
    Returns their paths.
    """
    root = os.path.join(base_path, "data", "output", "model",
                        f"{split_tag}_run_{run}")
    tables_dir = os.path.join(root, "tables")
    ckpt_dir   = os.path.join(root, "ckpts")
    plts_dir   = os.path.join(root, "plts")
    for d in (tables_dir, ckpt_dir, plts_dir):
        os.makedirs(d, exist_ok=True)
    return tables_dir, ckpt_dir, plts_dir

def merge_predictions(
    df_meta: pd.DataFrame,
    all_preds:           list,
    all_labels:          list
) -> pd.DataFrame:
    """
    Given a metadata DataFrame and two lists/arrays of preds & labels,
    assemble the combined mapping_df.
    """
    df = df_meta.copy().reset_index(drop=True)
    df['predicted_label'] = all_preds
    df['actual_label']    = all_labels
    # rename for clarity
    df = df.rename(columns={'id.orig_h':'src_ip', 'id.resp_h':'dst_ip'})
    # reorder columns
    front = ['uid','src_ip','dst_ip','predicted_label','actual_label']
    rest  = [c for c in df.columns if c not in front]
    return df[front + rest]

def save_prediction_tables(
    mapping_df: pd.DataFrame,
    split_tag: str,
    run:       int,
    tables_dir:str
):
    """
    Saves mapping_df, false-negatives, false-positives CSVs.
    """
    fn_df = mapping_df.loc[
        (mapping_df.predicted_label==0)&(mapping_df.actual_label==1)
    ]
    fp_df = mapping_df.loc[
        (mapping_df.predicted_label==1)&(mapping_df.actual_label==0)
    ]

    mapping_df.to_csv(
        os.path.join(
            tables_dir,
            f"predictions_with_all_features{split_tag}_run_{run}.csv"
        ), index=False
    )
    fn_df.to_csv(
        os.path.join(
            tables_dir,
            f"false_negatives{split_tag}_run_{run}.csv"
        ), index=False
    )
    fp_df.to_csv(
        os.path.join(
            tables_dir,
            f"false_positives{split_tag}_run_{run}.csv"
        ), index=False
    )

def save_model_checkpoint(
    model:     torch.nn.Module,
    split_tag: str,
    run:       int,
    ckpt_dir:  str
):
    model_name = type(model).__name__
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{model_name}{split_tag}_run_{run}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

def plot_and_save_confusion_matrix(
    all_preds: list,
    all_labels:list,
    split_tag: str,
    run:       int,
    plts_dir:  str
):
    """
    Computes confusion matrix, plots it, and saves the figure.
    """
    label_map = {0: 'benign', 1: 'attack'}
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=[label_map[0],label_map[1]],
        yticklabels=[label_map[0],label_map[1]],
        cmap='Blues'
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    fname = f"confusion_matrix{split_tag}_run_{run}.png"
    outpath = os.path.join(plts_dir, fname)
    plt.savefig(outpath)
    plt.close()
    return outpath

def save_all_results(
    base_path:    str,
    df_test_meta: pd.DataFrame,
    all_preds:    list,
    all_labels:   list,
    split_tag:    str,
    run:          int,
    model:        torch.nn.Module
) -> dict:
    """
    Orchestrates saving of:
      • prediction tables (mapping + FN/FP) -> tables/
      • model checkpoint -> ckpts/
      • confusion‐matrix plot -> plts/
    under data/output/model/{split_tag}_run_{run}/.
    """
    tables_dir, ckpt_dir, plts_dir = make_output_dirs(base_path, split_tag, run)

    mapping_df = merge_predictions(df_test_meta, all_preds, all_labels)
    save_prediction_tables(mapping_df, split_tag, run, tables_dir)
    ckpt_path = save_model_checkpoint(model, split_tag, run, ckpt_dir)
    cm_path   = plot_and_save_confusion_matrix(
        all_preds, all_labels, split_tag, run, plts_dir
    )
    return {
        "tables_dir": tables_dir,
        "ckpt_path":  ckpt_path,
        "cm_path":    cm_path
    }