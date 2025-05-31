# 生物資訊與計算分子生物學期末報告

這是一個生物資訊課程的期末專案，旨在利用圖神經網路預測酵母菌 (Saccharomyces cerevisiae) 中的蛋白質-蛋白質交互作用 (PPI)。專案涵蓋了從原始資料處理、特徵工程、模型訓練到結果評估與視覺化的完整流程。

## 專案特色

* **資料來源**: 使用 [STRING 資料庫](https://string-db.org/) 的交互作用資料、蛋白質序列資料 (FASTA)，以及基因本體論 (Gene Ontology, GO)註解資料。
* **資料前處理**:
    * 根據信心分數篩選交互作用。
    * 移除冗餘資料並標準化蛋白質 ID。
    * 整合蛋白質序列與 GO 註解資訊。
    * 清理低品質的蛋白質資料。
* **特徵工程**:
    * **GO 特徵**: 利用 GO Term 的傳播 (propagation) 與資訊量 (Information Content)。可選擇使用 SVD 進行降維。
    * **序列特徵**: 計算胺基酸組成與序列長度。
    * **圖特徵**: 提取節點的度 (degree)、介數中心性 (betweenness centrality)、緊密中心性 (closeness centrality) 等拓撲特徵。
    * 特徵縮放以提升模型效能。
* **模型建構**:
    * 採用圖神經網路 (GNN)，可選擇 **GraphSAGE** 或 **GAT** 架構。
    * 包含多層 GNN Encoder，並可選用 MLP 或點積 (Dot Product) 作為連結預測器。
* **模型訓練與評估**:
    * 使用帶有提早停止 (Early Stopping) 機制的訓練流程。
    * 評估指標包含 AUC-ROC、AUC-PR、F1-Score、精確度 (Precision)、召回率 (Recall) 等。
    * 支援進階分析，如網路拓撲分析、功能模組識別、預測模式分析。
* **視覺化**:
    * 訓練過程視覺化 (損失、AUC、AP 變化)。
    * 精確度-召回率曲線 (Precision-Recall Curve)。
    * 蛋白質網路圖、節點嵌入 (Embeddings) t-SNE 視覺化。
    * 度分佈、功能模組、預測連結等視覺化呈現。

## 環境需求

主要的 Python 函式庫需求請見 `requirements.txt`

## **準備原始資料**:
    * 將原始資料檔案 (如 `4932.protein.links.v12.0.txt`, `4932.protein.info.v12.0.txt`, `UP000002311_559292.fasta`, `sgd.gaf` 等) 放入 `raw_data/` 資料夾 (預設路徑，可在 `config.yaml` 中修改)。
    * go-basic.obo 放入 processed_data/

## 使用方法

本專案透過 `main.py` 腳本執行不同的階段。  
`explore.py` 腳本進行初步資料探索。

### 設定檔

專案的主要設定皆透過 `config.yaml` 檔案進行管理。主要的設定區塊包含：

* `data`: 原始資料與處理後資料的路徑、交互作用信心閾值等。
* `features`: 特徵工程的選項，如是否使用 GO 傳播、序列特徵、圖特徵的種類、是否進行特徵縮放等。
* `model`: GNN 模型的類型 (GraphSAGE/GAT)、隱藏層維度、嵌入層維度、dropout 率、預測器類型等。
* `training`: 訓練相關參數，如學習率、epochs 數、patience (用於提早停止)、資料切割比例、隨機種子等。
* `evaluation`: 評估相關選項，如是否執行進階分析、新穎預測的閾值與數量、功能模組分析方法等。
* `experiment`: 實驗名稱與輸出目錄相關設定。

### 執行流程

你可以透過指定 `--mode` 參數來執行專案的不同部分。

1.  **完整流程 (Full Pipeline)**:
    執行資料前處理、特徵工程、模型訓練與評估。
    ```bash
    python main.py --config config.yaml --mode full_pipeline
    ```

2.  **僅執行資料前處理 (Preprocessing)**:
    ```bash
    python main.py --config config.yaml --mode preprocess
    ```
    這會處理原始資料並儲存於 `processed_data/` 中，同時會產生初步的資料視覺化圖表。

3.  **僅執行特徵工程 (Feature Engineering)**:
    *注意：需先完成資料前處理。*
    ```bash
    python main.py --config config.yaml --mode feature_engineer
    ```
    這會根據設定檔提取節點特徵，並儲存於 `processed_data/node_features_matrix.npy`。

4.  **僅執行模型訓練 (Training)**:
    *注意：需先完成資料前處理與特徵工程。*
    ```bash
    python main.py --config config.yaml --mode train
    ```
    訓練好的模型將儲存於 `experiments/<experiment_name_timestamp>/models/best_model.pt`。

5.  **僅執行模型評估 (Evaluation)**:
    *注意：需先完成模型訓練。*
    ```bash
    python main.py --config config.yaml --mode evaluate
    ```
    評估報告與相關圖表將儲存於 `experiments/<experiment_name_timestamp>/` 對應的子目錄中。

### 命令列參數

* `--config`: 指定設定檔路徑 (預設: `config.yaml`)。
* `--mode`: 執行模式，可選 `preprocess`, `feature_engineer`, `train`, `evaluate`, `full_pipeline`。
* `--seed`: (可選) 覆寫設定檔中的隨機種子。
* `--no_cuda`: (可選) 強制使用 CPU。

## 輸出結果

執行完成後，相關的輸出將儲存於 `experiments/<experiment_name_timestamp>/` 資料夾中，包含：

* `config.yaml`: 該次實驗使用的設定檔副本。
* `logs/`: 執行日誌檔案。
* `models/`: 儲存訓練好的模型 (`best_model.pt`)。
* `plots/`: 各式視覺化圖表，如訓練歷史、PR 曲線、網路圖等。
* `evaluation_report.txt`: 模型效能評估報告。
* `topology_report.txt`: 網路拓撲分析報告 (若啟用)。
* `prediction_patterns_report.txt`: 預測模式分析報告 (若啟用)。
* `training_history.json`: 訓練過程的詳細數據。

在 `processed_data/` 資料夾中，你會找到：

* `proteins_processed.csv`: 處理後的蛋白質資訊與提取的初步特徵。
* `interactions_processed.csv`: 處理後的交互作用資料。
* `node_features_matrix.npy`: 最終用於 GNN 的節點特徵矩陣。
* `feature_names.pkl`: 特徵名稱列表。
* `feature_info/`: FeatureEngineer 儲存的元件 (如 scaler, SVD 物件)。
* `initial_visualizations/`: 資料前處理階段產生的視覺化圖表。

## 主要模組說明

* **`main.py`**: 主執行腳本，協調整個流程的運行。
* **`config.yaml`**: 專案的設定檔，用於配置所有參數。
* **`utils.py`**: 工具函式模組，包含設定檔載入、日誌設定、隨機種子設定、設備選擇、資料分割等。
* **`data/preprocess.py`**: 負責原始資料的完整前處理流程，從讀取 STRING 資料庫檔案到產生清理過的蛋白質與交互作用列表，並整合序列與 GO 資訊。
* **`data/features.py`**: 實現特徵工程，包括 GO 特徵、序列特徵、圖論特徵的提取與轉換。
* **`data/loader.py`**: 負責將前處理和特徵工程後的資料載入為 PyTorch Geometric 的 `Data` 物件，以供 GNN 模型使用。
* **`models/gnn.py`**: 定義圖神經網路 (GNN) 的架構，可選擇 GraphSAGE 或 GAT，並包含連結預測層。
* **`models/train.py`**: 實現模型訓練、驗證、測試的迴圈，包含提早停止邏輯與學習率調整。
* **`evaluation/metrics.py`**: 計算模型預測效能的各種指標 (如 AUC, AP, F1-score)，並產生評估報告，識別潛在新交互作用。
* **`evaluation/analysis.py`**: 提供進階分析功能，例如分析網路的拓撲特性、基於節點嵌入識別潛在的功能模組、分析模型的預測模式等。
* **`visualization/plots.py`**: 包含繪製各種圖表的函式，例如訓練歷史、PR 曲線、t-SNE 嵌入視覺化、預測連結視覺化等。
* **`visualization/ppi_visualizer.py`**: 一個專門用於在資料前處理階段生成 PPI 網路各種視覺化圖表的類別。
* **`explore.py`**: 一個獨立的腳本，用於初步的資料探索與視覺化分析，在專案早期開發階段使用。
