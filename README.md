# 基於圖神經網路的蛋白質-蛋白質交互作用預測

本專案旨在使用圖神經網路（GNN）預測蛋白質之間的交互作用（PPI）。它涵蓋了從原始資料處理、特徵工程、模型訓練、評估到結果視覺化的完整流程。

## 目錄

1.  [專案概覽](#專案概覽)
2.  [專案結構](#專案結構)
3.  [環境設置](#環境設置)
    * [依賴安裝](#依賴安裝)
    * [資料準備](#資料準備)
4.  [設定檔說明 (`config.yaml`)](#設定檔說明-configyaml)
    * [資料 (`data`)](#資料-data)
    * [特徵 (`features`)](#特徵-features)
    * [模型 (`model`)](#模型-model)
    * [訓練 (`training`)](#訓練-training)
    * [實驗 (`experiment`)](#實驗-experiment)
5.  [使用說明](#使用說明)
    * [主要執行腳本 (`main.py`)](#主要執行腳本-mainpy)
    * [執行模式](#執行模式)
    * [命令列參數](#命令列參數)
    * [範例指令](#範例指令)
6.  [詳細流程](#詳細流程)
    * [1. 資料預處理 (`data/preprocess.py`)](#1-資料預處理-datapreprocesspy)
    * [2. 特徵工程 (`data/features.py`)](#2-特徵工程-datafeaturespy)
    * [3. 資料載入 (`data/loader.py`)](#3-資料載入-dataloaderpy)
    * [4. 模型架構 (`models/gnn.py`)](#4-模型架構-modelsgnnpy)
    * [5. 模型訓練 (`models/train.py`)](#5-模型訓練-modelstrainpy)
    * [6. 模型評估 (`evaluation/`)](#6-模型評估-evaluation)
    * [7. 結果視覺化 (`visualization/plots.py`)](#7-結果視覺化-visualizationplotspy)
7.  [輸出結果](#輸出結果)
8.  [參數調整建議](#參數調整建議)
9.  [公用程式 (`utils.py`)](#公用程式-utilspy)

## 專案概覽

蛋白質-蛋白質交互作用對於理解細胞功能至關重要。本專案利用圖神經網路從已知的蛋白質交互網路和蛋白質自身的多種特徵中學習，以預測潛在的、新的蛋白質交互作用。

## 環境設置

### 依賴安裝

1.  **克隆專案庫**（如果適用）：
    ```bash
    git clone <your-repository-url>
    cd bioinf_final_project
    ```
2.  **建立虛擬環境** (建議)：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
3.  **安裝依賴套件**：
    本專案使用 PyTorch 和 PyTorch Geometric。請確保您的環境與 PyTorch Geometric 的要求相符（例如 CUDA 版本）。
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` 包含以下主要套件：
    * `torch`
    * `torch-geometric`
    * `torch-scatter`
    * `torch-sparse`
    * `networkx`
    * `matplotlib`
    * `numpy`
    * `pandas`
    * `scikit-learn`
    * `pyyaml` (用於載入 `config.yaml`)
    * `goatools` (用於 GO 詞彙傳播，在 `data/features.py` 中可選)

### 資料準備

1.  **下載原始資料**：
    本專案預期使用來自 STRING 資料庫的資料以及蛋白質序列和 GO註解資料。根據 `config.yaml` 中的 `data.raw_dir` 設定（預設為 `"raw_data/"`），將原始資料檔案放置到對應的目錄下。
    預處理腳本 (`data/preprocess.py`) 中使用的特定檔案包括：
    * 蛋白質交互作用: `4932.protein.links.v12.0.txt`
    * 蛋白質資訊: `4932.protein.info.v12.0.txt`
    * 蛋白質別名: `4932.protein.aliases.v12.0.txt`
    * 蛋白質序列 (FASTA): `UP000002311_559292.fasta`
    * GO 註解 (GAF): `sgd.gaf`
    * GO Ontology (OBO, 可選，用於GO傳播): `go-basic.obo` (應放置在 `processed_data/` 目錄下供 `data/features.py` 使用)

    請確保這些檔案（或您使用的相應檔案）位於 `raw_data/` 目錄中。`4932` 是酵母菌的 STRING 生物體 ID，FASTA 檔案名也可能特定於某個物種/資料來源。

## 設定檔說明 (`config.yaml`)

`config.yaml` 檔案用於管理專案的所有重要參數。

### 資料 (`data`)
* `raw_dir`: 原始資料檔案的目錄路徑 (預設: `"raw_data/"`)。
* `processed_dir`: 儲存預處理後資料的目錄路徑 (預設: `"processed_data/"`)。
* `confidence_threshold`: STRING 資料庫中蛋白質交互作用的最小置信度分數 (預設: 700)。

### 特徵 (`features`)
* `use_go_propagation`: 是否使用 GO 詞彙向上傳播 (預設: `true`)。
* `use_ic_features`: 是否使用基於資訊含量的 GO 特徵 (預設: `true`)。
* `go_svd_components`: 對 GO 特徵進行 SVD 降維後的組件數量 (預設: 100)。設為 0 表示不進行 SVD。
* `use_sequence_features`: 是否使用蛋白質序列特徵 (預設: `true`)。
* `amino_acids`: 用於計算胺基酸組成的胺基酸列表。
* `use_graph_features`: 是否使用基於圖拓撲的特徵 (預設: `true`)。
* `graph_features`: 要使用的圖拓撲特徵列表 (例如: `'degree'`, `'betweenness'`, `'closeness'`)。
* `scale_features`: 是否對特徵進行標準化縮放 (預設: `true`)。

### 模型 (`model`)
* `type`: GNN 編碼器的類型，可選 `"GraphSAGE"` 或 `"GAT"` (預設: `"GraphSAGE"`)。
* `hidden_dim`: GNN 中間隱藏層的維度 (預設: 128)。
* `embed_dim`: GNN 輸出的節點嵌入維度 (預設: 64)。
* `dropout`: Dropout 比率 (預設: 0.5)。

### 訓練 (`training`)
* `epochs`: 最大訓練輪次 (預設: 100)。
* `lr`: 學習率 (預設: 0.005)。
* `patience`: 提前停止的耐心輪次 (預設: 10)。
* `val_ratio`: 驗證集佔總邊的比例 (預設: 0.1)。
* `test_ratio`: 測試集佔總邊的比例 (預設: 0.15)。
* `seed` (在 `utils.py` 中預期，但 `config.yaml` 中未明確列出，可以自行添加): 隨機種子，用於可重現性。

### 實驗 (`experiment`)
* `name`: 實驗名稱，用於組織輸出檔案 (預設: `"ppi_link_prediction_test"`)。
* `dir` (由 `utils.py` 自動產生): 實驗結果儲存的根目錄，通常是 `experiments/experiment_name_timestamp`。

## 使用說明

### 主要執行腳本 (`main.py`)

`main.py` 是執行不同階段流程的主要入口點。

### 執行模式

您可以透過 `--mode` 參數指定不同的執行模式：

* `preprocess`: 僅執行資料預處理。
* `train`: 僅執行模型訓練（假設資料已預處理）。
* `evaluate`: 僅執行模型評估（假設模型已訓練且資料已預處理）。
* `full_pipeline`: 依序執行預處理、訓練和評估。
* `custom_eval`: 一個自訂的評估模式範例，用於獲取新的預測並視覺化嵌入。

### 命令列參數

* `--config`: 設定檔的路徑 (預設: `config.yaml`)。
* `--mode`: 執行模式 (預設: `full_pipeline`)。
* `--seed`: 覆寫設定檔中的隨機種子。
* `--no_cuda`: 強制使用 CPU，即使 CUDA 可用。

### 範例指令

* **執行完整流程**：
    ```bash
    python main.py --config config.yaml --mode full_pipeline
    ```
* **僅預處理資料**：
    ```bash
    python main.py --mode preprocess
    ```
* **僅訓練模型**：
    ```bash
    python main.py --mode train
    ```
* **僅評估模型**：
    ```bash
    python main.py --mode evaluate
    ```
* **使用特定設定檔並設定種子**：
    ```bash
    python main.py --config my_custom_config.yaml --mode train --seed 123
    ```

## 詳細流程

### 1. 資料預處理 (`data/preprocess.py`)

此階段處理原始資料並為後續步驟做準備。
* **讀取 STRING 資料庫檔案**：載入蛋白質交互作用、蛋白質資訊和別名。
* **過濾交互作用**：根據 `config.data.confidence_threshold` 過濾低置信度的交互作用。
* **ID 清理**：移除蛋白質 ID 中的生物體特定前綴 (例如，"4932.")。
* **移除重複邊**：確保每個交互作用的唯一性。
* **整合外部資料**：載入蛋白質序列 (FASTA) 和 GO 註解 (GAF)。
* **ID 對映**：使用別名資訊將不同來源的蛋白質 ID 對映到統一的內部 ID。
* **特徵提取 (初步)**：
    * 為每個蛋白質收集其 GO 詞彙和序列。
    * 計算初步的圖拓撲特徵（度、介數、接近度）。
* **蛋白質清理**：迭代移除缺乏足夠特徵（無 GO 詞彙、無序列且度 <= 1）的蛋白質。
* **儲存處理後資料**：將清理後的蛋白質列表、交互作用列表、對映關係和網路圖儲存到 `config.data.processed_dir` 目錄下，供後續使用。主要輸出檔案為 `proteins_processed.csv` 和 `interactions_processed.csv`。

### 2. 特徵工程 (`data/features.py` 及其在流程中的整合)

此階段的目標是從蛋白質資料中提取並建構資訊豐富的數值特徵，供圖神經網路模型學習。理想情況下，特徵工程應在資料預處理之後、模型訓練之前進行。

**`data/features.py` 模組的核心功能 (`FeatureEngineer` 類別)：**

該模組定義了一個 `FeatureEngineer` 類別，能夠執行以下進階特徵提取任務：

* **GO (Gene Ontology) 詞彙特徵**:
    * 將蛋白質的 GO 詞彙集合轉換為多熱編碼 (multi-hot encoding) 的數值向量，使用 `MultiLabelBinarizer`。
    * **GO 詞彙傳播 (可选)**: 如果設定 (`config.features.use_go_propagation`) 且已安裝 `goatools` 並提供 `go-basic.obo` 檔案，可以將每個 GO 詞彙擴展到其所有父詞彙，以捕捉更廣泛的生物學功能資訊。
    * **GO 資訊含量 (IC) 特徵 (可选)**: 如果設定 (`config.features.use_ic_features`)，計算基於 GO 詞彙在資料集中出現頻率的資訊含量特徵，如 IC 總和、平均值和最大值。
    * **SVD 降維 (可选)**: 如果設定 (`config.features.go_svd_components > 0`)，可以使用 `TruncatedSVD` 對高維度的 GO 特徵向量進行降維，以減少特徵數量並可能去除雜訊。
* **蛋白質序列特徵 (可选)**: 如果設定 (`config.features.use_sequence_features`)：
    * 計算序列長度 (並進行 `log1p` 轉換)。
    * 計算各種胺基酸的組成百分比。
    * 計算簡化的疏水性比例和電荷。
* **圖拓撲特徵 (可选)**: 如果設定 (`config.features.use_graph_features`)：
    * 使用在預處理階段計算的節點度 (degree)、介數中心性 (betweenness centrality) 和接近中心性 (closeness centrality) 等圖指標。
    * 對這些指標進行 `log1p` 轉換，以幫助其分佈更接近常態分佈。
* **二進位指示特徵**:
    * 蛋白質是否具有 GO 詞彙的指示 (1 或 0)。
    * 蛋白質是否具有序列的指示 (1 或 0)。
* **特徵組合與縮放**:
    * 將上述所有選定的特徵水平堆疊 (hstack) 成一個統一的蛋白質特徵矩陣。
    * **特徵縮放 (可选)**: 如果設定 (`config.features.scale_features`)，使用 `StandardScaler` 對非二進位特徵進行標準化處理，使其具有零均值和單位方差。
* **儲存/載入特徵處理元件**: `FeatureEngineer` 類別還提供了儲存和載入擬合好的轉換器（如 `MultiLabelBinarizer`, `TruncatedSVD`, `StandardScaler`）的功能，這對於確保在不同階段（如訓練和推斷）使用一致的特徵轉換非常重要。

**目前 `main.py` 流程中的特徵處理：**

在您提供的 `main.py` 腳本中，特徵的處理流程與 `data/features.py` 的設計有所不同：

1.  **`data/preprocess.py` 的角色**：
    * 此腳本負責從原始檔提取初步的蛋白質資訊，包括：
        * 原始的 GO 詞彙字串 (以分號分隔)。
        * 原始的蛋白質序列字串。
        * 基本的圖拓撲特徵 (degree, betweenness, closeness)。
        * `has_GO_term` 和 `has_sequence` 的二進位指示。
    * 這些資訊被整理並儲存到 `proteins_processed.csv` 檔案中。

2.  **`data/loader.py` 的角色**：
    * `load_custom_ppi_data` 函數讀取 `proteins_processed.csv`。
    * 它將 `proteins_processed.csv` 中除了 `protein_id` 和 `protein_name`（如果存在）之外的所有列都當作節點的初始特徵，並嘗試將它們轉換為數值型態。
    * **重要**: 這意味著 `loader.py` 目前會直接使用 `proteins_processed.csv` 中的 `degree`, `betweenness`, `closeness`, `has_GO_term`, `has_sequence` 這些已經是數值或可以輕易轉換為數值的列作為特徵。然而，它也會嘗試將 `GO_term`（原始GO詞彙字串）和 `sequence`（原始序列字串）這兩列直接轉換為數值。由於這些是字串，`pd.to_numeric(errors='coerce')` 會將它們轉換為 `NaN`，然後被填充為 0。這並不是一種有效的特徵化方法，也沒有利用到 `data/features.py` 中定義的更複雜的 GO 和序列特徵提取邏輯。

**整合建議與釐清：**

為了充分利用 `data/features.py` 中定義的進階特徵工程能力，並使流程更清晰，建議進行以下調整：

1.  **明確各模組職責**：
    * `data/preprocess.py`: 專注於從原始資料中清洗、提取並儲存結構化的蛋白質資訊（如蛋白質列表、交互作用、原始GO詞彙列表、原始序列字串、基本圖指標）。其輸出 `proteins_processed.csv` 應包含這些結構化但**未完全數值化**的特徵。
    * `data/features.py` (`FeatureEngineer`): 接收 `preprocess.py` 輸出的結構化蛋白質資料（例如 `proteins_df` 和 `mappings`），並根據 `config.yaml` 中的設定，執行詳細的數值化特徵工程（GO多熱編碼、SVD、序列特徵計算、縮放等），最終輸出一個**完全數值化的特徵矩陣 `X`** 和對應的特徵名稱。
    * `data/loader.py`: 其主要職責是接收圖的連接關係 (`interactions_processed.csv`) 和由 `FeatureEngineer` 生成的**最終數值化節點特徵矩陣 `X`**，然後將它們組合成 PyTorch Geometric 的 `Data` 物件。

2.  **修改 `main.py` 流程**：
    * 在 `run_training` (以及 `run_evaluation`, `custom_eval`) 函數中，在呼叫 `preprocess_ppi_data` 之後，但在呼叫 `load_custom_ppi_data` 之前，插入一個新的步驟來執行特徵工程。
    * 這個新步驟會：
        * 載入 `preprocess.py` 產生的 `proteins_processed.csv` (得到 `proteins_df`) 和 `mappings.pkl`。
        * 實例化 `FeatureEngineer`。
        * 呼叫 `feature_engineer.extract_features(proteins_df, mappings)` 來得到數值化的特徵矩陣 `node_features`。
        * 然後，將這個 `node_features` 和 `interactions_processed.csv` 傳遞給 `load_custom_ppi_data`（或者 `load_custom_ppi_data` 需要被修改以接受這些輸入）。

**若不進行上述流程調整，則 `data/features.py` 中的大部分進階特徵工程邏輯 (如 GO 詞彙傳播、SVD、胺基酸組成等) 將不會被實際使用，模型將僅依賴於 `preprocess.py` 提供的基本圖指標和二進位指示特徵。**

請根據您的需求決定是否進行這樣的流程調整。如果希望使用 `data/features.py` 中更豐富的特徵，則上述調整是必要的。

### 3. 資料載入 (`data/loader.py`)

此模組負責將預處理後的資料和特徵轉換為 PyTorch Geometric 的 `Data` 物件，以便 GNN 模型使用。
* **讀取處理後資料**：從 `interactions_processed.csv` 讀取交互作用，從 `proteins_processed.csv` 讀取蛋白質特徵。
* **ID 對映**：建立從原始蛋白質 ID 到從 0 開始的連續節點索引的對映。
* **節點特徵矩陣 (`x`)**：
    * 從 `proteins_processed.csv` 中選取特徵列。
    * 確保特徵是數值型別，處理潛在的 NaN。
    * 對於在交互作用中但沒有特徵的蛋白質，其特徵向量通常初始化為零。
* **邊索引 (`edge_index`)**：根據交互作用建立圖的邊連接資訊，通常表示為 `[2, num_edges]` 的張量。對於無向圖，會為每條邊添加兩個方向。
* **邊屬性 (`edge_attr`)**：可選地，可以包含邊的特徵，例如交互作用的置信度分數。
* **輸出 `Data` 物件**：包含 `x`, `edge_index`, `edge_attr` (如果使用) 以及節點 ID 對映等資訊。

### 4. 模型架構 (`models/gnn.py`)

定義了 GNN 編碼器和鏈接預測器。
* **`ImprovedGNNEncoder`**:
    * 支援 GraphSAGE (`SAGEConv`) 或 Graph Attention Network (`GATConv`) 層。
    * 包含多個 GNN 層，層間使用 ReLU 激活函數和 BatchNorm。
    * 應用 Dropout 以防止過擬合。
    * 對於 GraphSAGE，如果維度匹配，則在第二層後使用殘差連接。
* **鏈接預測器**:
    * `DotProductLinkPredictor`: 計算節點對嵌入的點積作為鏈接存在的分數。
    * `MLPLinkPredictor`: 將節點對嵌入串聯後輸入一個多層感知器 (MLP) 以預測鏈接分數，能學習更複雜的交互模式。
* **`ImprovedLinkPredictionGNN`**:
    * 整合了 GNN 編碼器和鏈接預測器。
    * `forward` 方法首先通過編碼器獲取所有節點的嵌入，然後針對 `edge_label_index` 中指定的邊（正負樣本）提取對應節點的嵌入，最後通過鏈接預測器得到預測分數。

### 5. 模型訓練 (`models/train.py`)

包含模型訓練、驗證和測試的邏輯。
* **資料分割**: 在 `main.py` 中，使用 `torch_geometric.transforms.RandomLinkSplit` 將資料集中的邊分割為訓練集、驗證集和測試集。此轉換會添加 `edge_label` (邊的標籤，0或1) 和 `edge_label_index` (用於監督的邊)。
* **訓練循環 (`train` function)**:
    * 模型設為訓練模式 (`model.train()`)。
    * 遍歷訓練資料中的每個圖 (對於 `RandomLinkSplit` 的輸出，通常是一個包含訓練邊的圖物件)。
    * 梯度清零、前向傳播獲取預測 logits。
    * 使用 `torch.nn.BCEWithLogitsLoss` 計算損失。
    * 反向傳播計算梯度。
    * 可選的梯度裁剪。
    * 優化器更新模型參數。
* **驗證/測試循環 (`test` function)**:
    * 模型設為評估模式 (`model.eval()`)。
    * 不計算梯度。
    * 收集所有預測和真實標籤。
    * 計算 ROC AUC 和 Average Precision (AP) 分數。
* **帶提前停止的訓練 (`train_with_early_stopping` function)**:
    * 在每個 epoch 後，在驗證集上評估模型。
    * 如果驗證集的 AP 分數在 `patience` 個 epoch 內沒有提升，則提前停止訓練。
    * 儲存驗證集上表現最佳的模型狀態。
    * 可選地使用學習率排程器 (例如 `ReduceLROnPlateau`)。
    * 記錄訓練過程中的損失、指標和學習率。

### 6. 模型評估 (`evaluation/`)

#### `evaluation/metrics.py`
* **`evaluate_model`**: 計算多種標準評估指標，包括 ROC AUC, Average Precision, Accuracy, Precision, Recall, F1-score。同時計算混淆矩陣的各個值 (TP, FP, TN, FN)。
* **`plot_precision_recall_curve`**: 繪製並儲存 Precision-Recall 曲線圖（此功能已整合到 `visualization/plots.py` 中，建議使用那裡的統一版本）。
* **`get_novel_predictions`**: 識別模型高置信度預測為存在但實際上是負樣本（或未標記）的鏈接，這些可視為潛在的新發現。
* **`generate_evaluation_report`**: 生成包含所有評估指標和潛在新預測的文字報告。

#### `evaluation/analysis.py`
* **`analyze_network_topology`**: 分析 PPI 網路的拓撲屬性，如節點數、邊數、平均度、集聚係數、最大連通分量大小等。
* **`generate_topology_report`**: 生成網路拓撲分析的文字報告。
* **`identify_functional_modules`**: 使用節點嵌入（來自 GNN 模型）和聚類演算法（如 K-Means 或 DBSCAN）來識別潛在的蛋白質功能模組或複合物。
* **`analyze_prediction_patterns`**: 分析模型對不同連接模式（例如，Hub-Hub, Hub-Peripheral, Peripheral-Peripheral）的鏈接預測表現。
* **`generate_prediction_patterns_report`**: 生成預測模式分析的文字報告。
* `plot_prediction_pattern_statistics`: 繪製預測模式統計圖（此功能已整合到 `visualization/plots.py`，建議使用那裡的統一版本）。

### 7. 結果視覺化 (`visualization/plots.py`)

此模組提供多種繪圖功能，以視覺化資料、模型結果和分析。
* **`visualize_graph`**: 視覺化 PPI 網路圖。
* **`visualize_embeddings`**: 使用 t-SNE 將高維節點嵌入降維到二維並繪製散點圖。
* **`visualize_predicted_links`**: 在網路圖中高亮顯示模型預測的新鏈接。
* **`visualize_degree_distribution`**: 繪製節點度的分佈直方圖。
* **`plot_training_history`**: 繪製訓練過程中的損失和評估指標變化曲線。
* **`plot_precision_recall_curve`**: 繪製 Precision-Recall 曲線。
* **`visualize_functional_modules`**: 視覺化基於節點嵌入聚類得到的功能模組。
* **`plot_prediction_pattern_statistics`**: 以長條圖形式視覺化不同連接模式下的預測統計數據。

## 輸出結果

執行 `main.py` 後，所有結果（模型、日誌、圖表、報告）將儲存到由 `config.experiment.name` 和時間戳決定的實驗目錄中，通常位於 `experiments/` 資料夾下。
例如：`experiments/ppi_link_prediction_test_20240101_120000/`

* `models/best_model.pt`: 儲存的驗證集上表現最佳的模型權重。
* `plots/`: 包含所有生成的圖表，如訓練歷史、PR 曲線、嵌入視覺化等。
* `logs/ppi_link_prediction_test.log` (或類似名稱): 執行的詳細日誌。
* `config.yaml`: 當次執行的設定檔副本。
* `training_history.json`: 訓練過程指標的 JSON 檔案。
* `evaluation_report.txt`: 模型在測試集上的詳細評估報告。
* `topology_report.txt`: 網路拓撲分析報告。
* `prediction_patterns_report.txt`: 預測模式分析報告。
* `novel_predictions_top50.csv` (在 `custom_eval` 模式下): 潛在新交互作用的列表。

## 參數調整建議

* **`data.confidence_threshold`**: 提高此閾值會得到更可靠但可能更稀疏的初始網路。降低則相反。
* **`features.go_svd_components`**: SVD 組件數影響 GO 特徵的維度和表達能力。需要根據 GO 詞彙總數和計算資源進行調整。
* **`model.type`**: `GAT` 通常在捕捉不同鄰居重要性方面更強大，但計算成本也更高。`GraphSAGE` 更高效。
* **`model.hidden_dim` / `model.embed_dim`**: 增加維度可以提升模型容量，但也可能導致過擬合和更高的計算需求。
* **`training.lr`**: 學習率是關鍵超參數。過高可能導致不收斂，過低則訓練緩慢。通常需要實驗來找到最佳值。
* **`training.patience`**: 提前停止的耐心值。較小的值可能導致訓練過早停止，較大的值則可能在過擬合後才停止。
* **特徵選擇 (`features.use_...`)**: 可以嘗試不同的特徵組合，看哪些對您的特定資料集最有效。

## 公用程式 (`utils.py`)

`utils.py` 包含整個專案中使用的輔助函數：
* `load_config`: 從 YAML 檔案載入並驗證設定。
* `_validate_config`: 驗證設定檔中的基本參數。
* `_setup_experiment_dir`: 建立用於儲存當前執行結果的實驗目錄。
* `setup_logging`: 設定日誌記錄器，將日誌同時輸出到控制台和檔案。
* `set_seed`: 設定 Python、NumPy 和 PyTorch 的隨機種子以確保實驗可重現性。
* `get_device`: 自動選擇使用 CUDA (GPU) 或 CPU。
* `save_results`: 將字典形式的結果儲存為 JSON 檔案。
* `create_edge_index_from_adjacency` (未使用): 從鄰接矩陣建立邊索引。
* `calculate_class_weights` (未使用): 計算用於處理類別不平衡的權重。