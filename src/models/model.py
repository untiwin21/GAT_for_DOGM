import torch
import torch.nn as nn
import torch.nn.functional as F
# torch_geometric is required for GAT layers.
# This dependency will need to be added to the package.xml and installed.
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("PyTorch Geometric is not installed. Please install it to use the GAT model.")
    GATConv = None

class GATLayer(nn.Module):
    """
    Graph Attention Network layer to process spatial relationships in a single sensor frame.
    """
    def __init__(self, in_features, hidden_features, out_features, heads=8):
        super(GATLayer, self).__init__()
        if GATConv is None:
            raise ImportError("torch_geometric.nn.GATConv could not be imported.")
        self.conv1 = GATConv(in_features, hidden_features, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_features * heads, out_features, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        """
        Forward pass for the GAT layer.
        Args:
            data: A PyTorch Geometric Data object with attributes:
                  - x: Node feature matrix [num_nodes, in_features]
                  - edge_index: Graph connectivity in COO format [2, num_edges]
        Returns:
            A tensor representing the graph embedding.
        """
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling (e.g., mean) to get a single vector for the entire frame
        # This represents the spatial context of one frame.
        return torch.mean(x, dim=0)


class GATTransformer(nn.Module):
    """
    Combines GAT and Transformer to learn spatio-temporal features.
    - GAT processes spatial relationships in each frame.
    - Transformer processes temporal relationships across 5 frames.
    """
    def __init__(self, gat_in_features=13, gat_hidden_features=8, gat_out_features=64, 
                 nhead=4, num_encoder_layers=3, dim_feedforward=128):
        super(GATTransformer, self).__init__()

        self.gat_layer = GATLayer(gat_in_features, gat_hidden_features, gat_out_features)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=gat_out_features, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.5, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Prediction Head
        # Takes the final temporal embedding and predicts the sigmas.
        # Output: 2 values (sigma for LiDAR position, sigma for Radar velocity)
        self.prediction_head = nn.Sequential(
            nn.Linear(gat_out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, frame_window):
        """
        Forward pass for the main model.
        Args:
            frame_window: A list of 5 PyG Data objects, where each object represents a sensor frame.
        """
        
        # 1. Process each frame in the window with GAT to get spatial embeddings
        # TODO: This needs to be parallelized properly. A simple loop is for illustration.
        spatial_embeddings = [self.gat_layer(frame) for frame in frame_window]
        
        # 2. Stack the embeddings to create a sequence for the transformer
        # Shape: [batch_size=1, sequence_length=5, features=gat_out_features]
        temporal_sequence = torch.stack(spatial_embeddings).unsqueeze(0)
        
        # 3. Process the sequence with the Transformer Encoder
        # The transformer will learn the temporal dependencies between the 5 frames.
        temporal_output = self.transformer_encoder(temporal_sequence)
        
        # 4. Use the output of the last time step for prediction
        # We assume the most relevant temporal context is in the final output.
        final_embedding = temporal_output[:, -1, :]
        
        # 5. Predict the sigma values
        predicted_sigmas = self.prediction_head(final_embedding)
        
        return predicted_sigmas

# Example of how to create the model
if __name__ == '__main__':
    # This is for demonstration and debugging
    
    # Check if PyG is installed
    if GATConv is None:
        exit()

    model = GATTransformer()
    print("GAT+Transformer Model Architecture:")
    print(model)

    # --- Create Dummy Input Data (5 frames) ---
    # This simulates the input the model expects: a list of 5 graph objects.
    dummy_window = []
    num_nodes_per_frame = 20 # e.g., 20 LiDAR/Radar points
    for _ in range(5):
        # Dummy features for each point (now with 13 features)
        node_features = torch.randn(num_nodes_per_frame, 13) 
        # Dummy edges (e.g., fully connected graph for simplicity)
        edge_index = torch.randint(0, num_nodes_per_frame, (2, 50))
        
        # PyG Data object
        from torch_geometric.data import Data
        dummy_window.append(Data(x=node_features, edge_index=edge_index))

    # --- Run Inference ---
    with torch.no_grad():
        output = model(dummy_window)
    
    print(f"\nDummy Input: A window of 5 frames, each with {num_nodes_per_frame} nodes.")
    print(f"Model Output (predicted sigmas): {output}")
    print(f"Output shape: {output.shape}")
