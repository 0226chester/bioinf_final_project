import pandas as pd

# 1. 定義輸入和輸出檔案名稱
input_filename = 'filtered_interactions.csv'
output_filename_with_id = 'filtered_interactions_with_protein_id.csv'
output_protein_list_filename = 'protein_list_id.csv'

# 2. 載入原始的 interactions 檔案
try:
    df_interactions = pd.read_csv(input_filename)
    print(f"成功載入檔案: {input_filename}")
except FileNotFoundError:
    print(f"錯誤: 找不到檔案 '{input_filename}'。請檢查檔案路徑。")
    exit()

# 3. 收集所有唯一的蛋白質名稱
# 蛋白質名稱可能出現在 'protein1' 或 'protein2' 欄位
all_proteins = pd.concat([df_interactions['protein1'], df_interactions['protein2']]).unique()

# 4. 建立蛋白質名稱到 ID 的映射字典
# 我們將使用從 1 開始的連續數字作為 protein_id
protein_to_id = {protein: i + 1 for i, protein in enumerate(all_proteins)}

# 5. 建立蛋白質列表 DataFrame
df_protein_list = pd.DataFrame(list(protein_to_id.items()), columns=['protein_name', 'protein_id'])

# 6. 將 protein_id 添加到原始的 interactions DataFrame
df_interactions['protein1_id'] = df_interactions['protein1'].map(protein_to_id)
df_interactions['protein2_id'] = df_interactions['protein2'].map(protein_to_id)


### 刪除重複邊

# 7. 為了處理 (A,B) 和 (B,A) 視為同一個交互，我們將 ID 進行排序
# 建立一個新的暫時欄位來標準化每對交互
df_interactions['sorted_protein_ids'] = df_interactions.apply(
    lambda row: tuple(sorted([row['protein1_id'], row['protein2_id']])), axis=1
)

# 8. 刪除重複的交互關係
# 根據 sorted_protein_ids 欄位進行去重，只保留第一次出現的交互
original_rows = len(df_interactions)
df_interactions_unique = df_interactions.drop_duplicates(subset=['sorted_protein_ids'])
removed_rows = original_rows - len(df_interactions_unique)

print(f"原始交互數量: {original_rows}")
print(f"移除重複交互數量: {removed_rows}")
print(f"最終唯一交互數量: {len(df_interactions_unique)}")

# 9. 移除臨時的 sorted_protein_ids 欄位，保留需要的欄位
# 可以選擇只保留 'protein1', 'protein2', 'protein1_id', 'protein2_id'
final_df = df_interactions_unique[['protein1', 'protein2', 'protein1_id', 'protein2_id','combined_score']]


# 10. 儲存帶有 protein_id 和去重後的 interactions 檔案到新檔案
final_df.to_csv(output_filename_with_id, index=False)
print(f"已將帶有 protein_id 並移除重複交互的數據儲存到: {output_filename_with_id}")

# 11. 儲存獨立的蛋白質列表檔案
df_protein_list.to_csv(output_protein_list_filename, index=False)
print(f"已將蛋白質列表儲存到: {output_protein_list_filename}")

print("\n處理完成！")