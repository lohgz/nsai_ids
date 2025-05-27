import os
import json
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def soft_high(v: float, t: float, k: float) -> float:
    """
    Compute a smooth “high‐threshold” activation: values below t map to ~0,
    values above t map to ~1, with transition sharpness controlled by k.
    
    Parameters
    ----------
    v : float
        The input value to evaluate.
    t : float
        The threshold around which the sigmoid transition is centered.
    k : float
        The steepness parameter: larger k -> sharper transition.
    
    Returns
    -------
    float
        A value in [0,1] indicating how far v exceeds t on a smooth sigmoid scale.
    """
    # avoid division by zero
    span = t if t > 0 else 1.0
    
    # normalized distance above threshold, steepened by k
    return 1.0 / (1.0 + np.exp(-k * ((v - t) / span)))


def soft_low(v, t, k):
    """
    Compute a smooth “low‐threshold” activation: values above t map to ~0,
    values below t map to ~1, with transition sharpness controlled by k.
    
    Parameters:
    ----------
    v : float
        The input value to evaluate.
    t : float
        The threshold around which the sigmoid transition is centered.
    k : float
        The steepness parameter: larger k -> sharper transition.
    
    Returns:
    ----------
    float
        A value in [0,1] indicating how far v falls below t on a smooth sigmoid scale.
    """
    # avoid division by zero when t == 0
    span = t if t > 0 else 1.0
    
    # normalized distance above threshold, steepened by k
    return 1.0 / (1.0 + np.exp(-k * ((t - v) / span)))

def compute_weights(
    X_numeric: pd.DataFrame,
    y: pd.Series,
    hi_thr_of: callable,
    lo_thr_of: callable,
    k: float,
    which: str = "all"
) -> pd.Series | tuple[pd.Series, pd.Series, pd.Series]:  # by default returns three Series (all weights below) otherwise return Series for selected weight
    """
    Compute fuzzy‐rule weights via logistic regression at selected k.
    """

    # short tcp burst weights (composite version)
    Xfc = pd.DataFrame({
        "flow_duration_low":        X_numeric["flow_duration"].apply(lambda v: soft_low(v, lo_thr_of("flow_duration"), k)),
        "flow_RST_flag_count_high": X_numeric["flow_RST_flag_count"].apply(lambda v: soft_high(v, hi_thr_of("flow_RST_flag_count"), k)),
        "bwd_PSH_flag_count_high":  X_numeric["bwd_PSH_flag_count"].apply(lambda v: soft_high(v, hi_thr_of("bwd_PSH_flag_count"), k)),
        "micro_0":                  ((X_numeric["fwd_pkts_tot"] <= lo_thr_of("fwd_pkts_tot")) &
                                     (X_numeric["bwd_pkts_tot"] <= lo_thr_of("bwd_pkts_tot")) &
                                     (X_numeric["payload_bytes_per_second"] >= hi_thr_of("payload_bytes_per_second"))).astype(float),
        "is_psh_present":           ((X_numeric["fwd_PSH_flag_count"] > 0) &
                                     (X_numeric["bwd_PSH_flag_count"] > 0)).astype(float)
    })
    clf_c = LogisticRegression(class_weight="balanced", solver="liblinear")
    clf_c.fit(Xfc, y)
    w_c = np.abs(clf_c.coef_[0]); w_c /= w_c.sum()
    w_c_ser = pd.Series(w_c, index=Xfc.columns, name="w_c")

    # short tcp burst weights (atomic version)
    Xfa = pd.DataFrame({
        "flow_duration_low":             X_numeric["flow_duration"].apply(lambda v: soft_low(v, lo_thr_of("flow_duration"), k)),
        "flow_RST_flag_count_high":      X_numeric["flow_RST_flag_count"].apply(lambda v: soft_high(v, hi_thr_of("flow_RST_flag_count"), k)),
        "bwd_PSH_flag_count_high":       X_numeric["bwd_PSH_flag_count"].apply(lambda v: soft_high(v, hi_thr_of("bwd_PSH_flag_count"), k)),
        "fwd_pkts_tot_low":              X_numeric["fwd_pkts_tot"].apply(lambda v: soft_low(v, lo_thr_of("fwd_pkts_tot"), k)),
        "bwd_pkts_tot_low":              X_numeric["bwd_pkts_tot"].apply(lambda v: soft_low(v, lo_thr_of("bwd_pkts_tot"), k)),
        "payload_bytes_per_second_high": X_numeric["payload_bytes_per_second"].apply(lambda v: soft_high(v, hi_thr_of("payload_bytes_per_second"), k)),
        "is_psh_present":                ((X_numeric["fwd_PSH_flag_count"] > 0) &
                                          (X_numeric["bwd_PSH_flag_count"] > 0)).astype(float)
    })
    clf_a = LogisticRegression(class_weight="balanced", solver="liblinear")
    clf_a.fit(Xfa, y)
    w_a = np.abs(clf_a.coef_[0]); w_a /= w_a.sum()
    w_a_ser = pd.Series(w_a, index=Xfa.columns, name="w_a")

    # test
    Xfe = pd.DataFrame({
        "flow_SYN_flag_count_low":  X_numeric["flow_SYN_flag_count"].apply(lambda v: soft_low(v, lo_thr_of("flow_SYN_flag_count"), k)),
        "flow_RST_flag_count_low":  X_numeric["flow_RST_flag_count"].apply(lambda v: soft_low(v, lo_thr_of("flow_RST_flag_count"), k)),
        "flow_FIN_flag_count_low":  X_numeric["flow_FIN_flag_count"].apply(lambda v: soft_low(v, lo_thr_of("flow_FIN_flag_count"), k)),
        "is_psh_present":           ((X_numeric["fwd_PSH_flag_count"] > 0) & (X_numeric["bwd_PSH_flag_count"] > 0)).astype(float),
        "bwd_init_window_size_high":X_numeric["bwd_init_window_size"].apply(lambda v: soft_high(v, hi_thr_of("bwd_init_window_size"), k)),
    })
    clf_e = LogisticRegression(class_weight="balanced", solver="liblinear")
    clf_e.fit(Xfe, y)
    w_e = np.abs(clf_e.coef_[0]); w_a /= w_a.sum()
    w_e_ser = pd.Series(w_e, index=Xfe.columns, name="w_ext")

    if which == "wc":
        return w_c_ser
    if which == "wa":
        return w_a_ser
    if which == "we":
        return w_e_ser
    # default: return all three
    return w_c_ser, w_a_ser, w_e_ser



def export_graph_for_neo4j(
    G: nx.DiGraph,
    out_dir: str
) :
    """
    NX Graph G whose nodes have a 'type' field
    ('ip' or 'flow') and edges annotated with either
      - sent=1  (IP -> Flow)
      - received_by=1  (Flow -> IP)
      - communicated=1 (IP <-> IP)
    Export three sets of CSVs for neo4j-admin import:
      - ip_nodes.csv
      - flow_nodes.csv
      - sent_edges.csv, received_by_edges.csv, communicated_edges.csv
    """

    # create directories
    os.makedirs(out_dir, exist_ok=True)

    # empty lists objects for loop
    ips, flows, rels = [], [], []

    # extract node types
    
    node_type = {n: d.get("type") for n, d in G.nodes(data=True)}
    # collect nodes
    for node, data in G.nodes(data=True):
        t = data.get("type")
        if t == "ip":
            row = {"address": node}
            for k, v in data.items():
                if k != "type":
                    row[k] = v
            ips.append(row)
        elif t == "flow":
            row = {"id": node}
            for k, v in data.items():
                if k == "type":
                    continue
                # json encoding
                row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            flows.append(row)

    # collect relationships
    for src, dst, data in G.edges(data=True):
        src_t = node_type.get(src)
        dst_t = node_type.get(dst)

        if data.get("sent") and src_t == "ip" and dst_t == "flow":
            rels.append({"type": "SENT", "start": src, "end": dst})
        elif data.get("received_by") and src_t == "flow" and dst_t == "ip":
            rels.append({"type": "RECEIVED_BY", "start": src, "end": dst})
        elif data.get("communicated") and src_t == "ip" and dst_t == "ip":
            rels.append({"type": "COMMUNICATED", "start": src, "end": dst})
        # else: ignore

    # build dfs
    df_ips   = pd.DataFrame(ips)
    df_flows = pd.DataFrame(flows)
    df_rels  = pd.DataFrame(rels)

    # export ip nodes
    ip_csv = os.path.join(out_dir, "ip_nodes.csv")
    df_ips = df_ips.rename(columns={"address": "address:ID(IP)"})
    df_ips.insert(1, ":LABEL", "IP")
    df_ips.to_csv(ip_csv, index=False, encoding="utf-8")
    print(f"Wrote IP nodes -> {ip_csv}")

    # export flow nodes
    flow_csv = os.path.join(out_dir, "flow_nodes.csv")
    df_flows = df_flows.rename(columns={"id": "id:ID(Flow)"})
    df_flows.insert(1, ":LABEL", "Flow")
    df_flows.to_csv(flow_csv, index=False, encoding="utf-8")
    print(f"Wrote Flow nodes -> {flow_csv}")

    # export edges by relationship type
    for rel_type, grp in df_rels.groupby("type"):
        if rel_type == "SENT":
            start_space, end_space = "IP", "Flow"
        elif rel_type == "RECEIVED_BY":
            start_space, end_space = "Flow", "IP"
        else:  # COMMUNICATED
            start_space, end_space = "IP", "IP"

        df_e = grp[["start", "end"]].copy()
        df_e[":TYPE"] = rel_type
        df_e = df_e[["start", ":TYPE", "end"]]
        df_e.columns = [
            f":START_ID({start_space})",
            ":TYPE",
            f":END_ID({end_space})"
        ]

        edge_csv = os.path.join(out_dir, f"{rel_type.lower()}_edges.csv")
        df_e.to_csv(edge_csv, index=False, encoding="utf-8")
        print(f"Wrote {rel_type} edges -> {edge_csv} ({len(df_e)} rows)")

    print("Neo4j CSVs ready for bulk import.")
