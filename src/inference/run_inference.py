import os
import sys
import numpy as np
import numba
import torch
import pandas as pd
import networkx as nx
import pyreason as pr

# make project-root imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_prep.data_module     import load_split_data, prepare_tensors_and_datasets
from models.model_module       import build_model
from inference.helpers         import compute_weights, soft_high, soft_low, export_graph_for_neo4j

def main():
    # config
    base_path   = os.getcwd()
    split_tag   = "_split_1"    # set which split to run (e.g. split_2_run_Y)
    run         = 1             # choose which run (e.g. split_X_run_1)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = None          # limit inference for testing

    # print to console which device, split and run were selected
    print(f"[INFER] device={device}  split={split_tag}  run={run}")

    # load & prepare data
    df_train, df_test = load_split_data(split_tag, base_path)
    _, test_ds, _, df_test_meta = prepare_tensors_and_datasets( # no need extract train data
        df_train, df_test
    )
    X_test_tensor, _ = test_ds.tensors # ignore y_test_tensor

    # check if inference limit is set
    if num_samples is not None:
        df_meta = df_test_meta.iloc[:num_samples].reset_index(drop=True)
        X_feat  = X_test_tensor[:len(df_meta)]
    else:
        df_meta = df_test_meta
        X_feat  = X_test_tensor

    # build model that was selected (split + run)
    model = build_model(
        X_test_tensor.shape[1],
        split_tag, run, base_path, device
    )

    # set class names
    class_names = ["benign", "attack"]
    
    # PyReason interface options
    interface_options = pr.ModelInterfaceOptions(
        threshold=0.9, # only predictions above 0.9 are considered
        set_lower_bound=True, # set probability as lower bound of confidence interval
        set_upper_bound=False, # 1 by default
        snap_value=None
    )

    # initialize the LIC
    anomaly_detector = pr.LogicIntegratedClassifier(
        model,
        class_names,
        model_name="anomaly_detector",
        interface_options=interface_options
)

    # custom annotation function 
    @numba.njit
    def prod_confidence(annotations, weights):
        lower, upper = 1.0, 1.0
        idx = 0
        for clause in annotations:
            for atom in clause:
                w = 1.0 if weights is None else weights[idx] # if w=None product t-norm else weighted geometric mean
                lower *= atom.lower ** w
                upper *= atom.upper ** w
                idx += 1
        return lower, min(upper, 1.0)

    # add the ann-function to the engine
    pr.add_annotation_function(prod_confidence)

    # inference loop (collect facts + preds)
    results              = []
    all_classifier_facts = []
    for meta_row, feature_vector in zip(df_meta.itertuples(index=False), X_feat):
        flow_id, batch = getattr(meta_row, "uid", None), feature_vector.unsqueeze(0).to(device)
        _, probs, facts = anomaly_detector(batch, flow_id=flow_id) # ignore logits
        pred_idx = int(probs.argmax()) # compute which class with argmax

        results.append({
            "flow_id":         flow_id,
            "predicted_label": class_names[pred_idx],
            "probabilities":   probs.tolist()
        })
        all_classifier_facts.extend(facts)

    #
    df_meta["predicted_label"] = [r["predicted_label"] for r in results]

    # prepare numeric training data for weight calc with logistic regression 
    X_train_df      = df_train.drop(columns="attack")
    num_cols        = X_train_df.select_dtypes(include="number").columns.tolist()
    X_train_numeric = X_train_df[num_cols]
    y_train         = df_train["attack"].astype(int)

    # output directory
    infer_out = os.path.join(
        base_path,
        "data", "output", "inference",
        f"{split_tag}_run_{run}"
    )

    # create sub directories
    for sub in ("thresholds", "weights", "neo4j", "nodes", "edges"):
        os.makedirs(os.path.join(infer_out, sub), exist_ok=True)

    # add all PyReason rules
    def add_all_rules(w_a, w_c, w_e):
        # FP brute-force rule
        pr.add_rule(pr.Rule(
            "fp_bruteforce(f):prod_confidence<-"
            "attack_flow(f):[0,1],"
            "flow_RST_flag_count_low(f):[0,1],"
            "bwd_PSH_flag_count_low(f):[0,1],"
            "fwd_pkts_per_sec_low(f):[0,1]",
            "rule_fp_bruteforce"
        ))

        # FN rules
        pr.add_rule(pr.Rule(
            "short_tcp_bursts_a(f):prod_confidence<-"
            "flow_duration_low(f):[0,1],flow_RST_flag_count_high(f):[0,1],"
            "bwd_PSH_flag_count_high(f):[0,1],fwd_pkts_tot_low(f):[0,1],"
            "bwd_pkts_tot_low(f):[0,1],payload_bytes_per_second_high(f):[0,1],"
            "is_psh_present(f):[1,1]",
            "rule_short_tcp_bursts_a"
        ))
        pr.add_rule(pr.Rule(
            "short_tcp_bursts_a_w(f):prod_confidence<-"
            "flow_duration_low(f):[0,1],flow_RST_flag_count_high(f):[0,1],"
            "bwd_PSH_flag_count_high(f):[0,1],fwd_pkts_tot_low(f):[0,1],"
            "bwd_pkts_tot_low(f):[0,1],payload_bytes_per_second_high(f):[0,1],"
            "is_psh_present(f):[1,1]",
            "rule_short_tcp_bursts_a_w",
            weights=w_a
        ))

        pr.add_rule(pr.Rule(
            "short_tcp_bursts_c(f):prod_confidence<-"
            "flow_duration_low(f):[0,1],flow_RST_flag_count_high(f):[0,1],"
            "bwd_PSH_flag_count_high(f):[0,1],micro_0(f):[0,1],is_psh_present(f):[0,1]",
            "rule_short_tcp_bursts_c"
        ))
        pr.add_rule(pr.Rule(
            "short_tcp_bursts_c_w(f):prod_confidence<-"
            "flow_duration_low(f):[0,1],flow_RST_flag_count_high(f):[0,1],"
            "bwd_PSH_flag_count_high(f):[0,1],micro_0(f):[0,1],is_psh_present(f):[0,1]",
            "rule_short_tcp_bursts_c_w",
            weights=w_c
        ))

        # explain attack rule (no weights)
        pr.add_rule(pr.Rule(
            "explain_flagged_http_attack(f):prod_confidence<-"
            "is_http(f):[1,1],attack_flow(f):[0,1],"
            "flow_duration_low(f):[0,1],payload_bytes_per_second_high(f):[0,1],"
            "fwd_pkts_per_sec_high(f):[0,1],fwd_pkts_tot_low(f):[0,1],"
            "bwd_pkts_tot_low(f):[0,1]",
            "rule_explain_flagged_http_attack"
        ))
        
    # dynamic threshold + weight sweep + reasoning & traces
    for hi_pct, lo_pct in [(0.75,0.25),(0.80,0.20),(0.70,0.30)]:
        label = f"{int(hi_pct*100)}th"

        # get benign features only for threshold
        benign      = df_train[df_train["attack"]==0][num_cols]

        # prepare thresholds
        dynamic_thr = benign.quantile([lo_pct, hi_pct]).T
        dynamic_thr.columns = [f"{int(lo_pct*100)}th", f"{int(hi_pct*100)}th"]

        # replace any zero values with 1
        dynamic_thr[f"{int(lo_pct*100)}th"].replace(0.0, 1.0, inplace=True)
        dynamic_thr[f"{int(hi_pct*100)}th"].replace(0.0, 1.0, inplace=True)

        # cap backward PSH flag count at 5 (5 is already a very high count for a single flow, so we use that instead of the the dynamic)
        # this can be done for any feature threshold
        dynamic_thr.loc["bwd_PSH_flag_count", f"{int(hi_pct*100)}th"] = 5.0

        # save thresholds as csv
        thr_file = f"thresholds_{label}{split_tag}_run_{run}_test.csv"
        dynamic_thr.to_csv(os.path.join(infer_out, "thresholds", thr_file))
        print(f"-> Saved thresholds to {thr_file}")

        # get the upper (hi-percentile) threshold for feature f
        hi_thr_of = lambda f: dynamic_thr.loc[f, f"{int(hi_pct*100)}th"]

        # for flow_duration: use the lower threshold
        # for every other f: use the upper threshold
        lo_thr_of = lambda f: (
            dynamic_thr.loc[f, f"{int(lo_pct*100)}th"]
            if f == "flow_duration"
            else dynamic_thr.loc[f, f"{int(hi_pct*100)}th"]
        )

        # sweep k's
        for k in [5.0, 1.0, 10.0]:
            # compute weights
            w_c_ser, w_a_ser, w_ext_ser = compute_weights(
                X_train_numeric, y_train, hi_thr_of, lo_thr_of, k, which="all"  
            )
            # merge & save
            weights_df = pd.concat([w_c_ser, w_a_ser, w_ext_ser], axis=1) # adjust where necessary
            w_name     = f"weights_{label}_k{k:.1f}{split_tag}_run_{run}_test.csv"
            weights_df.to_csv(os.path.join(infer_out, "weights",  w_name))
            print(f"-> Saved {w_name}")

            # reset pr engine
            pr.reset()

            # add classifier facts (attack_flow:[l,1]/benign_flow:[l,1])
            for fact in all_classifier_facts:
                pr.add_fact(fact)

            # pr settings
            pr.settings.allow_ground_rules = True
            pr.settings.atom_trace        = True

            # add rules using these weights
            add_all_rules(w_a_ser.values, w_c_ser.values, w_ext_ser.values)

            # build & load graph
            G = nx.DiGraph()

            # add IP nodes
            for ip in pd.unique(df_meta[["id.orig_h","id.resp_h"]].values.ravel()):
                # skip ipv6
                if ":" in ip:
                    continue
                G.add_node(ip,
                           type="ip",
                           internal=int(ip.startswith("192.168.100.")),
                           external=int(not ip.startswith("192.168.100.")),
                           server=int(ip=="192.168.100.218"))

            # add flow nodes & edges with composite attributes
            for _, r in df_meta.iterrows(): # ignore index
                # set source and destintation and skip ipv6
                src, dst = r["id.orig_h"], r["id.resp_h"]
                if ":" in src or ":" in dst:
                    continue

                # flow id        
                uid = r["uid"]
                # numeric features
                features = {f: float(r[f]) for f in num_cols}

                # high/low feature lists
                highs = [f for f in num_cols if features[f] >= hi_thr_of(f)]
                lows  = [f for f in num_cols if features[f] < lo_thr_of(f)]

                # custom composite and binary flags
                is_psh = (features["fwd_PSH_flag_count"] > 0 and features["bwd_PSH_flag_count"] > 0)
                low_pkts = (
                    features["fwd_pkts_tot"] <= lo_thr_of("fwd_pkts_tot")
                    and features["bwd_pkts_tot"] <= lo_thr_of("bwd_pkts_tot")
                )
                micro0 = low_pkts and (features["payload_bytes_per_second"] >= hi_thr_of("payload_bytes_per_second"))
                is_ftp  = (r["service"] == "ftp-data")
                is_http = (r["service"] == "http")
                is_ssh  = (r["service"] == "ssh")
                dirn    = r["traffic_direction"]
                isc2s   = (dirn == "client->server")
                iss2c   = (dirn == "server->client")


                # static entries
                attrs = {
                    "type": "flow",
                    "flow": 1,
                    "classification": r["predicted_label"],
                    "high_feats": ",".join(highs),
                    "low_feats": ",".join(lows),
                }

                # merge in all numeric features
                attrs.update(features)

                # add the composite and binary entries
                attrs.update({
                    "is_psh_present":      int(is_psh),
                    "micro_0":             int(micro0),
                    "is_ftp":              int(is_ftp),
                    "is_http":             int(is_http),
                    "is_ssh":              int(is_ssh),
                    "is_client_to_server": int(isc2s),
                    "is_server_to_client": int(iss2c),
                })

                # attach to the graph
                G.add_node(uid, **attrs)
                G.add_edge(src, uid, sent=1)
                G.add_edge(uid, dst, received_by=1)
                G.add_edge(src, dst, communicated=1)

            # load pr graph
            pr.load_graph(G)

            # export to neo4j as csv for faster imports
            subdir = os.path.join(infer_out, "neo4j", f"neo4j_{label}_k{k:.1f}")
            export_graph_for_neo4j(G, subdir)
            print(f"Exported graph for {label}, k={k} -> {subdir}")

            # assert fuzzy + boolean facts from the built graph for reasoning
            for node, data in G.nodes(data=True):
                # skip if not flow
                if not data.get("flow"): continue

                # split the features
                highs = data["high_feats"].split(",") if data["high_feats"] else []
                lows  = data["low_feats"].split(",")  if data["low_feats"]  else []

                # calculate the soft thresholds for all highs and lows
                for f in highs:
                    score = soft_high(data[f], hi_thr_of(f), k)
                    pr.add_fact(pr.Fact(f"{f}_high({node}):[{score:.4f},1.0]","flow_feat"))
                for f in lows:
                    score = soft_low(data[f], lo_thr_of(f), k)
                    pr.add_fact(pr.Fact(f"{f}_low({node}):[{score:.4f},1.0]","flow_feat"))


                for flag,val in data.items():
                    if (flag.startswith("is_") or flag=="micro_0") and val:
                        pr.add_fact(pr.Fact(f"{flag}({node})","flow_flag"))

            # reason & export traces
            interp = pr.reason()
            df_nodes, df_edges = pr.get_rule_trace(interp)
            node_file = f"pr_nodes_{label}_k{k:.1f}{split_tag}_run_{run}.csv"
            edge_file = f"pr_edges_{label}_k{k:.1f}{split_tag}_run_{run}.csv"
            df_nodes.to_csv(os.path.join(infer_out, "nodes", node_file), index=False)
            df_edges.to_csv(os.path.join(infer_out, "edges", edge_file), index=False)
            print(f"-> saved traces for k={k}, threshold={label}")

if __name__ == "__main__":
    main()
