import pandas as pd
import networkx as nx
import numpy as np #主要用於 NaN

def run_protein_cleaning_pipeline(features_file_path, interactions_file_path,
                                  output_features_file, output_interactions_file):
    """
    Cleans protein data by iteratively removing proteins with no GO_term, no sequence,
    and degree <= 1. Updates IDs, recalculates network metrics, and adds binary
    feature presence flags.
    """
    print("開始處理蛋白質數據清理流程...")

    # --- 1. 載入資料 ---
    try:
        features_df = pd.read_csv(features_file_path)
        interactions_df = pd.read_csv(interactions_file_path)
        print(f"成功載入檔案: {features_file_path} (共 {len(features_df)} 列)")
        print(f"成功載入檔案: {interactions_file_path} (共 {len(interactions_df)} 列)")
    except FileNotFoundError:
        print(f"錯誤: 找不到一個或多個輸入檔案。請檢查路徑。")
        return

    if 'GO_term' not in features_df.columns or 'sequence' not in features_df.columns:
        print("錯誤: 'protein_list_with_features.csv' 缺少 'GO_term' 或 'sequence' 欄位。")
        return
    if 'degree' not in features_df.columns:
        print("警告: 'protein_list_with_features.csv' 缺少 'degree' 欄位。將從交互作用數據中計算初始 degree。")
        # 在此假設 degree 欄位存在

    # --- 2. 迭代清理 ---
    iteration_count = 0
    while True:
        iteration_count += 1
        print(f"\n--- 開始迭代清理第 {iteration_count} 次 ---")

        if not interactions_df.empty and not features_df.empty:
            all_proteins_in_interactions = pd.concat([interactions_df['protein1'], interactions_df['protein2']])
            current_degrees = all_proteins_in_interactions.value_counts().rename('current_degree')
            
            original_index_name = features_df.index.name # 保存原始索引名稱
            features_df = features_df.set_index('protein_name', drop=False)
            features_df['degree'] = features_df.index.map(current_degrees).fillna(0).astype(int)
            features_df = features_df.reset_index(drop=True)
            if original_index_name: # 如果原始索引有名稱，恢復它
                 features_df = features_df.rename_axis(original_index_name, axis='index')

            print(f"已根據當前交互作用更新 {len(features_df[features_df['degree'] > 0])} 個蛋白質的 degree。")
        elif features_df.empty:
            print("蛋白質特徵列表已空，無需更新 degree。")
        else:
            features_df['degree'] = 0
            print("交互作用列表已空，所有剩餘蛋白質的 degree 設為 0。")

        condition_go_term_missing = features_df['GO_term'].isna() | (features_df['GO_term'] == '')
        condition_sequence_missing = features_df['sequence'].isna() | (features_df['sequence'] == '')
        condition_degree_low = features_df['degree'] <= 1

        proteins_to_remove_df = features_df[
            condition_go_term_missing &
            condition_sequence_missing &
            condition_degree_low
        ]

        if proteins_to_remove_df.empty:
            print("沒有蛋白質符合刪除條件。迭代清理完成。")
            break

        proteins_to_remove_names = set(proteins_to_remove_df['protein_name'])
        print(f"在第 {iteration_count} 次迭代中，找到 {len(proteins_to_remove_names)} 個蛋白質待刪除: {proteins_to_remove_names if len(proteins_to_remove_names) < 10 else str(list(proteins_to_remove_names)[:10]) + '...'}")

        features_df = features_df[~features_df['protein_name'].isin(proteins_to_remove_names)]
        print(f"從 features_df 移除蛋白質後，剩餘 {len(features_df)} 個蛋白質。")

        interactions_df = interactions_df[
            ~interactions_df['protein1'].isin(proteins_to_remove_names) &
            ~interactions_df['protein2'].isin(proteins_to_remove_names)
        ]
        print(f"從 interactions_df 移除交互作用後，剩餘 {len(interactions_df)} 個交互作用。")

        if features_df.empty:
            print("所有蛋白質均已被移除。")
            break

    if features_df.empty:
        print("\n清理後沒有剩餘蛋白質。將儲存空的檔案。")
        empty_features_cols = ['protein_name', 'protein_id', 'GO_term', 'sequence',
                               'degree', 'betweenness', 'closeness',
                               'has_GO_term', 'has_sequence'] # 加入新欄位
        empty_features_df = pd.DataFrame(columns=empty_features_cols)
        empty_interactions_df = pd.DataFrame(columns=['protein1', 'protein2', 'protein1_id', 'protein2_id', 'combined_score'])
        empty_features_df.to_csv(output_features_file, index=False)
        empty_interactions_df.to_csv(output_interactions_file, index=False)
        print(f"已儲存空的清理後檔案: {output_features_file}, {output_interactions_file}")
        return

    print(f"\n--- 迭代清理完成，剩餘 {len(features_df)} 個蛋白質和 {len(interactions_df)} 個交互作用 ---")

    print("\n開始更新蛋白質 ID...")
    features_df = features_df.copy()
    features_df.loc[:, 'protein_id'] = range(1, len(features_df) + 1)
    name_to_new_id_map = pd.Series(features_df.protein_id.values, index=features_df.protein_name).to_dict()

    interactions_df.loc[:, 'protein1_id'] = interactions_df['protein1'].map(name_to_new_id_map)
    interactions_df.loc[:, 'protein2_id'] = interactions_df['protein2'].map(name_to_new_id_map)
    interactions_df.dropna(subset=['protein1_id', 'protein2_id'], inplace=True)
    interactions_df.loc[:, 'protein1_id'] = interactions_df['protein1_id'].astype(int)
    interactions_df.loc[:, 'protein2_id'] = interactions_df['protein2_id'].astype(int)
    print(f"蛋白質 ID 更新完成。新的 ID 範圍從 1 到 {len(features_df)}。")

    print("\n開始重新計算網絡指標 (degree, betweenness, closeness)...")
    if not interactions_df.empty:
        G = nx.Graph()
        G.add_nodes_from(features_df['protein_id'])
        edges_for_graph = interactions_df[['protein1_id', 'protein2_id']].values.tolist()
        G.add_edges_from(edges_for_graph)

        degree_dict = dict(G.degree())
        features_df.loc[:, 'degree'] = features_df['protein_id'].map(degree_dict).fillna(0).astype(int)
        print("Degree 計算完成。")

        print("計算 Betweenness Centrality (可能需要一些時間)...")
        betweenness_dict = nx.betweenness_centrality(G, normalized=True, endpoints=False)
        features_df.loc[:, 'betweenness'] = features_df['protein_id'].map(betweenness_dict).fillna(0)
        print("Betweenness Centrality 計算完成。")

        print("計算 Closeness Centrality (可能需要一些時間)...")
        closeness_dict = nx.closeness_centrality(G)
        features_df.loc[:, 'closeness'] = features_df['protein_id'].map(closeness_dict).fillna(0)
        print("Closeness Centrality 計算完成。")
    else:
        print("交互作用列表為空，所有網絡指標 (degree, betweenness, closeness) 設為 0。")
        features_df.loc[:, 'degree'] = 0
        features_df.loc[:, 'betweenness'] = 0.0
        features_df.loc[:, 'closeness'] = 0.0

    # --- 新增 has_GO_term 和 has_sequence 標籤 ---
    print("\n新增 has_GO_term 和 has_sequence 標籤...")
    # 檢查 GO_term 是否存在 (不是 NaN 也不是空字串)
    features_df.loc[:, 'has_GO_term'] = (~features_df['GO_term'].isna() & (features_df['GO_term'] != '')).astype(int)
    # 檢查 sequence 是否存在 (不是 NaN 也不是空字串)
    features_df.loc[:, 'has_sequence'] = (~features_df['sequence'].isna() & (features_df['sequence'] != '')).astype(int)
    print("標籤新增完成。")

    # --- 5. 儲存結果 ---
    try:
        # 確保欄位順序，將新欄位放在後面
        desired_columns_order = ['protein_name', 'protein_id', 'GO_term', 'sequence',
                                 'degree', 'betweenness', 'closeness',
                                 'has_GO_term', 'has_sequence']
        # 過濾掉可能在開發過程中加入但不在期望列表中的欄位 (如果有的話)
        # 並確保所有期望的欄位都存在，不存在的則補NaN (理論上這裡都應該存在)
        current_cols = [col for col in desired_columns_order if col in features_df.columns]
        features_df_to_save = features_df[current_cols]

        # 如果 features_df 中缺少某些 desired_columns_order 中的欄位 (例如 GO_term, sequence 被意外刪除)
        # 則需要更穩健的方式來處理欄位順序，或者確保它們一直存在
        # 但根據流程，這些欄位應該都還在
        
        features_df_to_save.to_csv(output_features_file, index=False)
        print(f"\n成功儲存清理後的蛋白質特徵檔案: {output_features_file}")
        interactions_df.to_csv(output_interactions_file, index=False)
        print(f"成功儲存清理後的蛋白質交互作用檔案: {output_interactions_file}")
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")

    print("\n蛋白質數據清理流程執行完畢。")

# --- 主程式執行部分 ---
if __name__ == "__main__":
    base_path = "processed_data/"
    features_file = base_path + "protein_list_with_features.csv"
    interactions_file = base_path + "filtered_interactions_with_protein_id.csv" # 假設這是您的交互作用檔案名

    output_features_cleaned_file = base_path + "protein_list_with_features_clean.csv"
    output_interactions_cleaned_file = base_path + "filtered_interactions_with_protein_id_clean.csv"

    run_protein_cleaning_pipeline(features_file, interactions_file,
                                  output_features_cleaned_file, output_interactions_cleaned_file)

    # --- 驗證範例 (可選) ---
    # print("\n--- 驗證清理後的檔案 (前5列) ---")
    # try:
    #     cleaned_features = pd.read_csv(output_features_cleaned_file)
    #     print(f"\n{output_features_cleaned_file} (前5列):")
    #     print(cleaned_features.head())
    #
    #     cleaned_interactions = pd.read_csv(output_interactions_cleaned_file)
    #     print(f"\n{output_interactions_cleaned_file} (前5列):")
    #     print(cleaned_interactions.head())
    # except FileNotFoundError:
    #     print("一個或多個清理後的檔案未找到，可能清理流程中所有蛋白質都被移除了。")
    # except pd.errors.EmptyDataError:
    #     print("一個或多個清理後的檔案是空的。")
  