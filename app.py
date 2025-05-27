import os
import re
import glob
import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network

# paths
base_path      = os.getcwd()
model_root     = os.path.join(base_path, "data", "output", "model")
inference_root = os.path.join(base_path, "data", "output", "inference")

def collect_variants(directory, pattern, key_regex):
    files = glob.glob(os.path.join(directory, pattern), recursive=True)
    variants = {}
    for p in files:
        m = re.search(key_regex, os.path.basename(p))
        if m:
            variants[m.group(1)] = p
    return variants

# discover variants
variants_in_model     = {
    d for d in os.listdir(model_root)
    if os.path.isdir(os.path.join(model_root, d))
}
variants_in_inference = {
    d for d in os.listdir(inference_root)
    if os.path.isdir(os.path.join(inference_root, d))
}
common_variants = sorted(variants_in_model & variants_in_inference)

# sidebar: pick split/run once for everything
selected_variant = st.sidebar.selectbox(
    "Variant (_split_X_run_Y)",
    common_variants,
    help="Choose which split/run to load everywhere"
)

# build file-paths scoped to that variant
model_tables = os.path.join(model_root, selected_variant, "tables")
infer_path   = os.path.join(inference_root, selected_variant)

#gather all the CSVs under that variant
pred_variants = collect_variants(
    model_tables,
    "predictions_with_all_features_*.csv",
    r"predictions_with_all_features_(.+)\.csv"
)
fp_variants   = collect_variants(
    model_tables,
    "false_positives_*.csv",
    r"false_positives_(.+)\.csv"
)
fn_variants   = collect_variants(
    model_tables,
    "false_negatives_*.csv",
    r"false_negatives_(.+)\.csv"
)
thr_variants  = collect_variants(
    os.path.join(infer_path, "thresholds"),
    "*.csv",
    r"thresholds_(.+)\.csv"
)
node_variants = collect_variants(
    os.path.join(infer_path, "nodes"),
    "*.csv",
    r"pr_nodes_(.+)\.csv"
)
edge_variants = collect_variants(
    os.path.join(infer_path, "edges"),
    "*.csv",
    r"pr_edges_(.+)\.csv"
)

# cache data
@st.cache_data
def load_df(path: str, nrows: int = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

@st.cache_data
def load_node_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    bounds = (
        df["New Bound"]
          .str.strip("[]")
          .str.split(",", n=1, expand=True)
          .astype(float)
    )
    df["lower"], df["upper"] = bounds[0], bounds[1]
    return df

# sidebar -> model outputs
st.sidebar.header("Model Outputs")

# 1) Prediction CSV – auto-select the one matching `selected_variant`
pred_keys = sorted(pred_variants.keys())
default_pred = pred_keys.index(selected_variant) if selected_variant in pred_keys else 0
pred_key = st.sidebar.selectbox(
    "Prediction variant",
    pred_keys,
    index=default_pred
)
df_preds = load_df(pred_variants[pred_key]).set_index("uid")

# 2) Threshold CSV – auto-select the one matching `selected_variant`
thr_keys = sorted(thr_variants.keys())
default_thr = thr_keys.index(selected_variant) if selected_variant in thr_keys else 0
thr_key = thr_keys[default_thr]
df_thr = load_df(thr_variants[thr_key])

# 3) False-Positive / False-Negative switch
family = st.sidebar.selectbox("View", ["False Positives", "False Negatives"])
if family == "False Positives":
    fp_keys = sorted(fp_variants.keys())
    default_fp = fp_keys.index(selected_variant) if selected_variant in fp_keys else 0
    fp_key = st.sidebar.selectbox(
        "False-Positive set",
        fp_keys,
        index=default_fp
    )
    df_fp = load_df(fp_variants[fp_key])
    df_fn = None
else:
    fn_keys = sorted(fn_variants.keys())
    default_fn = fn_keys.index(selected_variant) if selected_variant in fn_keys else 0
    fn_key = st.sidebar.selectbox(
        "False-Negative set",
        fn_keys,
        index=default_fn
    )
    df_fn = load_df(fn_variants[fn_key])
    df_fp = None

# sidebar -> inference traces
st.sidebar.header("Inference Traces")
node_key = st.sidebar.selectbox("Node-trace variant", sorted(node_variants.keys()))
df_nodes = load_node_df(node_variants[node_key])
edge_key = st.sidebar.selectbox("Edge-trace variant", sorted(edge_variants.keys()))
df_edges = load_df(edge_variants[edge_key])

# compute baseline metrics
y_true = df_preds["actual_label"].astype(int)
y_pred = df_preds["predicted_label"].astype(int)
tp_all = int(((y_pred == 1) & (y_true == 1)).sum())
fp_all = int(((y_pred == 1) & (y_true == 0)).sum())
fn_all = int(((y_pred == 0) & (y_true == 1)).sum())
tn_all = int(((y_pred == 0) & (y_true == 0)).sum())
precision_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
recall_all    = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
accuracy_all  = (tp_all + tn_all) / len(df_preds) if len(df_preds) else 0.0
total_attacks = (y_true == 1).sum()
total_benign  = (y_true == 0).sum()

# rule analytics setup
st.sidebar.header("Rule Analytics")
rule_options = sorted(
    c for c in df_nodes["Occurred Due To"].dropna().unique()
    if c.startswith("rule_")
)
rule       = st.sidebar.selectbox("1) Rule for analytics", rule_options)
is_explain = rule.startswith("rule_explain")

if not is_explain:
    mode = st.sidebar.radio("2) Evaluate on:", ("False-positives", "False-negatives"))

    # catch if the user hasn’t loaded a matching FP/FN set
    if mode == "False-positives" and df_fp is None:
        st.error("Please select a False-Positive set matching the selected variant.")
        st.stop()
    elif mode == "False-negatives" and df_fn is None:
        st.error("Please select a False-Negative set matching the selected variant.")
        st.stop()

    missed = set(df_fp["uid"]) if mode == "False-positives" else set(df_fn["uid"])
    low_cut, high_cut = st.sidebar.slider(
        "3) Confidence ∈ [low, high]", 0.0, 1.0, (0.0, 1.0), step=0.01
    )
else:
    missed, low_cut, high_cut = set(), 0.0, 1.0

mask_all   = df_nodes["Occurred Due To"] == rule
caught_all = set(df_nodes.loc[mask_all, "Node"])
mask_int   = mask_all & df_nodes["lower"].between(low_cut, high_cut)
caught_int = set(df_nodes.loc[mask_int, "Node"])
hit_all    = caught_all & missed
hit_int    = caught_int & missed

# render RA
st.title(f"Rule Analytics: `{rule}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total triggered by rule", len(caught_all))
c2.metric(f"In interval [{low_cut:.2f},{high_cut:.2f}]", len(caught_int))

if is_explain:
    c3.metric("Total attacks", total_attacks)
    c4.metric("Total benigns", total_benign)
else:
    c3.metric(
        "Total " + ("benigns" if mode=="False-positives" else "attacks"),
        total_benign if mode=="False-positives" else total_attacks
    )
    c4.metric(
        "Total caught by rule", len(hit_all),
        delta=(f"{len(hit_all)/len(missed):.1%} recall" if missed else None)
    )

if not is_explain:
    st.markdown("---")
    st.markdown("### Baseline performance")
    b1, b2 = st.columns(2)
    b1.metric("Precision", f"{tp_all}/{tp_all+fp_all}", f"{precision_all:.1%}")
    b2.metric("Accuracy",  f"{tp_all+tn_all}/{len(df_preds)}", f"{accuracy_all:.1%}")
    st.markdown("---")

    # compute interval vs. noise and new metrics
    delta   = len(hit_int)
    flagged = len(caught_int)
    noise   = flagged - delta

    if mode == "False-positives":
        new_tp, new_fp, new_fn, new_tn = tp_all, fp_all-delta, fn_all, tn_all+delta
    else:
        new_tp, new_fp, new_fn, new_tn = tp_all+delta, fp_all, fn_all-delta, tn_all

    precision_new = new_tp / (new_tp + new_fp) if (new_tp + new_fp) else 0.0
    recall_new    = new_tp / (new_tp + new_fn) if (new_tp + new_fn) else 0.0
    accuracy_new  = (new_tp + new_tn) / len(df_preds)

    # rule based metrics
    st.markdown("### Hit-rate vs. Triggered")
    h1, h2 = st.columns(2)
    h1.metric(
        "Interval hit-rate",
        f"{len(hit_int)}/{len(caught_int)}",
        f"{(len(hit_int)/len(caught_int) if caught_int else 0.0):.1%}"
    )
    h2.metric(
        "Overall hit-rate",
        f"{len(hit_all)}/{len(caught_all)}",
        f"{(len(hit_all)/len(caught_all) if caught_all else 0.0):.1%}"
    )
    st.markdown("---")

    # show after interval metrics
    st.markdown("### After interval performance")
    n1, n2 = st.columns(2)
    n1.metric(
        "Noise flagged",
        f"{noise}/{flagged}",
        delta=f"{(noise/flagged if flagged else 0.0):.1%}"
    )
    if mode == "False-positives":
        p, a = st.columns(2)
        p.metric(
            "Precision",
            f"{new_tp}/{new_tp+new_fp}",
            delta=f"{precision_new-precision_all:+.1%}"
        )
        a.metric(
            "Accuracy",
            f"{new_tp+new_tn}/{len(df_preds)}",
            delta=f"{accuracy_new-accuracy_all:+.1%}"
        )
    else:
        r, a = st.columns(2)
        r.metric(
            "Recall",
            f"{new_tp}/{new_tp+new_fn}",
            delta=f"{recall_new-recall_all:+.1%}"
        )
        a.metric(
            "Accuracy",
            f"{new_tp+new_tn}/{len(df_preds)}",
            delta=f"{accuracy_new-accuracy_all:+.1%}"
        )

    st.markdown("---")
    st.markdown("### Threshold trade-off chart")
    thresholds = np.linspace(0, 1, 21)
    plot_data = []
    for thr in thresholds:
        m    = mask_all & (df_nodes["lower"] >= thr)
        s    = set(df_nodes.loc[m, "Node"])
        hits = s & missed
        plot_data.append({"threshold": thr, "flagged": len(s), "caught": len(hits)})
    df_plot = pd.DataFrame(plot_data).set_index("threshold")
    st.line_chart(df_plot)

# inspect inidivudal flows
st.markdown("---")
st.markdown("### Inspect a triggered flow")

if not caught_int:
    st.info("No flows in that confidence interval.")
else:
    choices, mapping = [], {}
    for uid in caught_int:
        lbl = uid + (" ✔" if (not is_explain and uid in hit_all) else "")
        choices.append(lbl)
        mapping[lbl] = uid

    sel = st.selectbox("Choose a flow", choices)
    if st.button("Show details"):
        uid = mapping[sel]

        # meo4j setup
        uri, user, pwd = st.secrets["neo4j"].values()
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        cypher = """
        MATCH (src:IP)-[r1:SENT]->(f:Flow {id:$uid})-[r2:RECEIVED_BY]->(dst:IP)
        RETURN src.address AS src_ip, type(r1) AS rel1,
               f.id        AS flow_id,
               type(r2)    AS rel2, dst.address AS dst_ip
        """
        with driver.session() as sess:
            recs = [r.data() for r in sess.run(cypher, uid=uid)]

        if recs:
            st.write("#### Network path")
            st.write(pd.DataFrame(recs))
            G = nx.DiGraph()
            for r in recs:
                G.add_node(r["src_ip"]); G.add_node(r["flow_id"]); G.add_node(r["dst_ip"])
                G.add_edge(r["src_ip"], r["flow_id"], label=r["rel1"])
                G.add_edge(r["flow_id"], r["dst_ip"], label=r["rel2"])
            net = Network(height="300px", width="100%", directed=True)
            net.from_nx(G)
            st.components.v1.html(net.generate_html(), height=350)
        else:
            st.error("No graph path found.")

        # node level trace
        st.write("#### Node-level trace")
        df_n = df_nodes[df_nodes["Node"] == uid].reset_index(drop=True)

        # clause map for explainable ouputs
        clauses_map = {
            "rule_explain_flagged_http_attack": [
                "is_http","attack_flow","flow_duration_low",
                "payload_bytes_per_second_high",
                "fwd_pkts_per_sec_high",
                "fwd_pkts_tot_low","bwd_pkts_tot_low",
            ],
            "rule_short_tcp_bursts_c": [
                "flow_duration_low",
                "flow_RST_flag_count_high",
                "bwd_PSH_flag_count_high",
                "micro_0",
                "is_psh_present",
            ],
            "rule_short_tcp_bursts_c_w": [
                "flow_duration_low",
                "flow_RST_flag_count_high",
                "bwd_PSH_flag_count_high",
                "micro_0",
                "is_psh_present",
            ],
            "rule_short_tcp_bursts_a": [
                "flow_duration_low",
                "flow_RST_flag_count_high",
                "bwd_PSH_flag_count_high",
                "fwd_pkts_tot_low",
                "bwd_pkts_tot_low",
                "payload_bytes_per_second_high",
                "is_psh_present",
            ],
            "rule_short_tcp_bursts_a_w": [
                "flow_duration_low",
                "flow_RST_flag_count_high",
                "bwd_PSH_flag_count_high",
                "fwd_pkts_tot_low",
                "bwd_pkts_tot_low",
                "payload_bytes_per_second_high",
                "is_psh_present",
            ],
            "rule_fp_bruteforce": [
                "is_http",
                "flow_RST_flag_count_low",
                "bwd_PSH_flag_count_low",
                "fwd_pkts_per_sec_low",
                "stealth_burst",
            ]
        }
        rule_clauses = clauses_map.get(rule, [])

        def hl(r):
            return ["background-color:lightgreen" if r["Label"] in rule_clauses else "" for _ in r]

        st.dataframe(
            df_n.style
                .apply(hl, axis=1)
                .format({"lower":"{:.3f}", "upper":"{:.3f}"})
        )

        def exp(atom):
            if atom.endswith("_low"):
                return "low " + atom[:-4].replace("_", " ")
            if atom.endswith("_high"):
                return "high " + atom[:-5].replace("_", " ")
            if atom.startswith("is_"):
                return atom[3:].replace("_", " ") + " present"
            return atom.replace("_", " ")

        st.markdown(
            "**Why flagged?** " + (", ".join(exp(a) for a in rule_clauses) or "no clauses found")
        )
        
        # thresholds on demand
        st.markdown("#### Thresholds")
        st.dataframe(df_thr.style.format({"lower": "{:.3f}", "upper": "{:.3f}"}))

        # model & features
        st.markdown("#### Model & feature values")
        if uid in df_preds.index:
            df_t = df_preds.loc[[uid]].T
            df_t.columns = ["value"]
            st.write(df_t)
        else:
            st.warning("No prediction/features for this uid.")
