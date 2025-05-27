import pandas as pd
import os
import glob

# define path
base_path = os.getcwd()
path = os.path.join (base_path, "data", "raw", "web-ids23", "*.csv")

# list to save all dfs
df_list = []

# loop through files and save 
csv_files = glob.glob(path)
print ("Found and saved csv files:")
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)
    print(file)

# concat to one file
df = pd.concat(df_list, ignore_index=True)

# save file to output path
output_path = os.path.join (base_path, "data", "output", "concatenated.csv")
df.to_csv(output_path, index=False)
print (f"-> File succesfully saved to {output_path}")