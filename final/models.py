import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class ImprovedGNNEncoder(torch.nn.Module):
    """
    Enhanced Graph Neural Network encoder with multiple layers,
    residual connections and batch normalization.
    
    This architecture is designed to capture complex protein interaction patterns
    in PPI networks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, use_gat=False):
        super().__init__()
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Choose between GraphSAGE and GAT implementations
        if use_gat:
            # Graph Attention Network layers (better at capturing important interactions)
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_channels * 4, out_channels, heads=1, dropout=dropout)
        else:
            # GraphSAGE layers (efficient and effective for PPI networks)
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, out_channels)
        
        # Batch normalization helps with training stability
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels if not use_gat else hidden_channels * 4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels if not use_gat else hidden_channels * 4)

    def forward(self, x, edge_index):
        # First layer
        x1 = self.conv1(x, edge_index)
        if not self.use_gat:
            x1 = x1.relu()
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Second layer with residual connection if dimensions match
        x2 = self.conv2(x1, edge_index)
        if not self.use_gat:
            x2 = x2.relu()
        x2 = self.bn2(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        if not self.use_gat and x1.shape == x2.shape:
            x2 = x2 + x1  # Residual connection
        
        # Final layer
        x3 = self.conv3(x2, edge_index)
        return x3


class DotProductLinkPredictor(torch.nn.Module):
    """
    Simple dot product link predictor.
    
    Computes the inner product between node embeddings to predict link existence.
    Efficient for large graphs.
    """
    def forward(self, x_i, x_j):
        # Dot product predicts link existence
        return (x_i * x_j).sum(dim=-1)


class MLPLinkPredictor(torch.nn.Module):
    """
    MLP-based link predictor that learns a more complex function
    to predict link existence from node embeddings.
    
    This can capture more nuanced protein interaction patterns than a simple dot product.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Concatenate node features and apply MLP
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, 1)
    
    def forward(self, x_i, x_j):
        # Concatenate features (captures more complex interactions)
        x = torch.cat([x_i, x_j], dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.view(-1)


class ImprovedLinkPredictionGNN(torch.nn.Module):
    """
    Complete model for link prediction in PPI networks.
    
    Combines a GNN encoder with a link predictor.
    """
    def __init__(self, in_channels, hidden_channels, embed_channels, 
                 dropout=0.5, use_gat=False, use_mlp_predictor=True):
        super().__init__()
        self.encoder = ImprovedGNNEncoder(
            in_channels, hidden_channels, embed_channels, dropout, use_gat)
        
        if use_mlp_predictor:
            self.predictor = MLPLinkPredictor(embed_channels)
        else:
            self.predictor = DotProductLinkPredictor()

    def forward(self, data):
        # Get node embeddings using message-passing edges
        node_embeddings = self.encoder(data.x, data.edge_index)

        # Get edges for supervision
        edge_src = data.edge_label_index[0]
        edge_dst = data.edge_label_index[1]

        # Get embeddings for these specific source and target nodes
        embed_src = node_embeddings[edge_src]
        embed_dst = node_embeddings[edge_dst]

        # Predict link existence (raw scores/logits)
        pred_logits = self.predictor(embed_src, embed_dst)
        return pred_logits
    
    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings for visualization or downstream tasks.
        """
        return self.encoder(x, edge_index)