=== PPI PREDICTION PATTERN ANALYSIS ===
Degree threshold for High/Low classification: 14.00

--- Prediction Statistics by Node Connectivity ---
Total supervised edges analyzed: 31240

Hub-Hub (High-High) Predictions:
  Count: 16424 (52.6% of total supervised edges)
  Average confidence score: 0.7903
  Precision within category: 0.7711
  Recall within category: 0.9753

Hub-Peripheral (High-Low/Low-High) Predictions:
  Count: 9724 (31.1% of total supervised edges)
  Average confidence score: 0.1850
  Precision within category: 0.1469
  Recall within category: 0.7829

Peripheral-Peripheral (Low-Low) Predictions:
  Count: 5092 (16.3% of total supervised edges)
  Average confidence score: 0.1442
  Precision within category: 0.1090
  Recall within category: 0.6843

--- Biological Interpretation ---
1. Hub-Hub: Interactions between highly connected proteins, potentially bridging modules.
2. Hub-Peripheral: Interactions connecting hubs to less connected proteins, often regulatory.
3. Peripheral-Peripheral: Interactions between less connected proteins, often within specific pathways/complexes.
4. Confidence/Precision Patterns: Observe if the model is more confident or precise for certain types of interactions.