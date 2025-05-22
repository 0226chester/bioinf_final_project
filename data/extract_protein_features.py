import pandas as pd
import pickle
import os
import networkx as nx

# 路徑設定
data_dir = "processed_data"
input_csv = os.path.join(data_dir, "protein_list_id.csv")
interaction_csv = os.path.join(data_dir, "filtered_interactions_with_protein_id.csv")
output_csv = os.path.join(data_dir, "protein_list_with_features.csv")

# 載入資料
with open(os.path.join(data_dir, 'protein_to_go.pkl'), 'rb') as f:
    protein_to_go = pickle.load(f)

with open(os.path.join(data_dir, 'protein_sequences.pkl'), 'rb') as f:
    protein_sequences = pickle.load(f)

# 可選：載入 alias 對應表（如果你需要做補充查找）
alias_map_path = os.path.join(data_dir, 'string_to_alias.pkl')
string_to_alias = None
if os.path.exists(alias_map_path):
    with open(alias_map_path, 'rb') as f:
        string_to_alias = pickle.load(f)

# 載入欲查詢的蛋白清單
df = pd.read_csv(input_csv)

# 新增欄位
df['GO_term'] = ""
df['sequence'] = ""

# 處理每一筆蛋白
for idx, row in df.iterrows():
    pid = row['protein_name']  # e.g., Q0010
    
    ### === 取 GO terms === ###
    go_terms = set()
    has_go = False
    
    # 直接命中
    if pid in protein_to_go:
        go_dict = protein_to_go[pid]
        for aspect in ['P', 'F', 'C']:
            go_terms.update(go_dict.get(aspect, []))
        has_go = True

    # alias 查找（根據 coverage_analyzer.py）
    if not has_go and string_to_alias and pid in string_to_alias:
        for source, aliases in string_to_alias[pid].items():
            if source in ['UniProt_DR_SGD', 'SGD_ID']:
                for alias in aliases:
                    if alias in protein_to_go:
                        go_dict = protein_to_go[alias]
                        for aspect in ['P', 'F', 'C']:
                            go_terms.update(go_dict.get(aspect, []))
                        has_go = True
                        break
                if has_go:
                    break
    
    df.at[idx, 'GO_term'] = ';'.join(go_terms) if go_terms else ""

    ### === 取 sequence === ###
    has_seq = False
    seq_data = None

    # 直接命中
    if pid in protein_sequences:
        seq_data = protein_sequences[pid]
        has_seq = True

    # alias 查找（根據 coverage_analyzer.py）
    if not has_seq and string_to_alias and pid in string_to_alias:
        for source, aliases in string_to_alias[pid].items():
            if source in ['Ensembl_UniProt', 'UniProt_AC']:
                for alias in aliases:
                    if alias in protein_sequences:
                        seq_data = protein_sequences[alias]
                        has_seq = True
                        break
                if has_seq:
                    break

    if has_seq:
        if isinstance(seq_data, str):
            df.at[idx, 'sequence'] = seq_data
        # elif isinstance(seq_data, (list, tuple)) and len(seq_data) < 20:
        #     df.at[idx, 'sequence'] = ','.join(map(str, seq_data))
        # else:
        #     df.at[idx, 'sequence'] = f"[{type(seq_data).__name__}, len={len(seq_data)}]"
            df.at[idx, 'sequence'] = ''.join(map(str, seq_data))

    else:
        df.at[idx, 'sequence'] = ""


# 讀取互作資料並建立無向圖
interactions = pd.read_csv(interaction_csv)
edges = list(zip(interactions['protein1'], interactions['protein2']))
G = nx.Graph()
G.add_edges_from(edges)

# 計算圖論特徵
degree_dict = dict(G.degree())
betweenness_dict = nx.betweenness_centrality(G)
closeness_dict = nx.closeness_centrality(G)

# 加入到 dataframe
df['degree'] = df['protein_name'].map(degree_dict).fillna(0).astype(int)
df['betweenness'] = df['protein_name'].map(betweenness_dict).fillna(0)
df['closeness'] = df['protein_name'].map(closeness_dict).fillna(0)

# 儲存結果
df.to_csv(output_csv, index=False)
print(f"已輸出蛋白資訊至：{output_csv}")
