# Enhancing Flow-Level Intrusion Detection with Explainable Neural-Symbolic Reasoning

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  

**PyReason-based Flow Level Intrusion Detection System**

---

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Dataset](#dataset)  
- [Data Preparation](#data-preparation)  
- [Model Training](#model-training)  
- [Inference](#inference)  
- [Visualization](#visualization)  

---

## Features

- Uses **PyReason** for explainable NIDS
- Modular Pipeline    
- Interactive Streamlit dashboard with Neo4j graph visualizations  

---

## Prerequisites

- **Python** 3.10 or higher  
- **Neo4j** Community Edition (v5.x or later)  

---

## Installation
1. Clone this repository  
   ```bash
   git clone https://github.com/your-org/nsai_ids.git
   
2. Create & activate a virtual environment

3. Install required Python packages (Install the PyTorch version that matches your CUDA toolkit to enable GPU support.)
   ```bash
   pip install -r requirements.txt

## Dataset
1. Download the dataset from [WEB-IDS23 Dataset (Lanfer et al., 2025)](https://osnadata.ub.uni-osnabrueck.de/dataset.xhtml?persistentId=doi:10.26249/FK2/MOCIY8)
  
2. Save the folder "web-ids23" to:  
   ```bash
   data/raw/

## Data Preparation
1. Process raw CSV into cleaned format  
   ```bash
   python src/data_prep/process_csv.py
   
2. EDA and zero-day train/test splits generation  
   ```bash
   python src/data_prep/zero_day_splits.ipynb

## Model Training
1. Train the included CNN model or choose from a included pretrained model (or add one of your own to src/models)
   ```bash
   python src/data_modeling/train.py

## Inference
1. Run inference
   ```bash
   python src/inference/run_inference.py

## Visualization
1. Create a Neo4j database (but don't start it yet)

2. Configure credentials in
   ```bash
   .streamlit/secrets.toml

3. Import preprocessed CSVs into Neo4j with neo4j-admin import for fastest import
-  Move files from data/output/inference/_split_X_run_Y to neo4j import folder
-  Open database Terminal and run the command for neo4j admin import
     ```bash
   .\bin\neo4j-admin.bat database import full neo4j \
    --overwrite-destination \
    --id-type=string \
    --skip-duplicate-nodes \
    --nodes="IP=import/ip_nodes.csv" \
    --nodes="Flow=import/flow_nodes.csv" \
    --relationships="SENT=import/sent_edges.csv" \
    --relationships="RECEIVED_BY=import/received_by_edges.csv" \
    --relationships="COMMUNICATED=import/communicated_edges.csv" \
    --verbose

4. Launch the Streamlit app
   ```bash
   streamlit run app.py

Note: Cypher Queries can be integrated into the Streamlit App or the Graph can be queried in neo4j desktop. Example:
```bash
MATCH (src:IP)-[r1:SENT]->(f:Flow { id:"a_Ck4rZo46eeQupiWLB1" })-[r2:RECEIVED_BY]->(dst:IP)
RETURN src,r1,f,r2,dst
