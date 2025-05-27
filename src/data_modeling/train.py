# grid_search.py

import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# append project root
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_prep.data_module import load_split_data, prepare_tensors_and_datasets
from models.cnn import CNN_Classifier
from data_modeling.utils import save_all_results


def train_and_collect(model, loader, device, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct    = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1)==yb).sum().item()
    return


def evaluate_and_collect(model, loader, device):
    model.eval()
    tp = fp = tn = fn = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds  = logits.argmax(1)
            tp += ((preds==1)&(yb==1)).sum().item()
            fp += ((preds==1)&(yb==0)).sum().item()
            tn += ((preds==0)&(yb==0)).sum().item()
            fn += ((preds==0)&(yb==1)).sum().item()
            all_preds .append(preds)
            all_labels.append(yb)
    # metrics
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    # flatten
    all_preds  = torch.cat(all_preds,  dim=0).cpu().numpy().astype(int)
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
    return acc, prec, rec, all_preds, all_labels


def main():
    # config
    base_path    = os.getcwd()
    split_tag    = "_split_2"
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs   = 10

    # hyperparam grid
    param_grid = {
        "learning_rate": [0.001, 0.0005],
        "weight_decay":  [0.0001, 0.001],
        "batch_size":    [256, 512],
    }
    
    # loss function
    criterion = nn.CrossEntropyLoss()

    print(f"[INFO] device={device} split={split_tag}")

    # load and split data
    df_train, df_test = load_split_data(split_tag, base_path)
    train_ds, test_ds, df_train_meta, df_test_meta = prepare_tensors_and_datasets(
        df_train, df_test
    )
    # feature dimension
    X_train_tensor, y_train_tensor = train_ds.tensors
    num_features = X_train_tensor.shape[1]

    # grid search
    for run_idx, (lr, wd, bs) in enumerate(itertools.product(
            param_grid["learning_rate"],
            param_grid["weight_decay"],
            param_grid["batch_size"],
    )):
        cfg_name = f"run{run_idx}_lr{lr}_wd{wd}_bs{bs}"
        print(f"\n[GRID] {cfg_name}")

        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                  num_workers=4, pin_memory=True)

        # model / optimizer
        model     = CNN_Classifier(num_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # train
        train_and_collect(model, train_loader, device,
                          criterion, optimizer, num_epochs)

        # eval
        acc, prec, rec, all_preds, all_labels = evaluate_and_collect(
            model, test_loader, device
        )
        print(f"[RESULT] {cfg_name} -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

        # save via save_all_results
        results = save_all_results(
            base_path=base_path,
            df_test_meta=df_test_meta,
            all_preds=all_preds,
            all_labels=all_labels,
            split_tag=split_tag,
            run=run_idx,
            model=model
        )

        print(f"[SAVED] tables -> {results['tables_dir']}")
        print(f"[SAVED] checkpoint -> {results['ckpt_path']}")
        print(f"[SAVED] confusion matrix -> {results['cm_path']}")

        summary = []
        # save to csv
        summary.append({
            "run": run_idx,
            "cfg": cfg_name,
            "lr": lr,
            "wd": wd,
            "bs": bs,
            "acc": acc,
            "prec": prec,
            "rec": rec
        })

        # save summary df
        df_summary = pd.DataFrame(summary)
        grids_dir = os.path.join(
            base_path, "data", "output", "model",
            f"{split_tag}_run_{run_idx}", "grids"
        )
        os.makedirs(grids_dir, exist_ok=True)
        summary_csv = os.path.join(grids_dir, "grid_search_summary.csv")
        df_summary.to_csv(os.path.join(grids_dir, "grid_search_summary.csv"), index=False)
        print(f"\n[SAVED SUMMARY] {summary_csv}")
        print("[DONE]")


if __name__ == "__main__":
    main()
