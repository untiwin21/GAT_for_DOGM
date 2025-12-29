import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import numpy as np

# Make sure the model is importable
import sys
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import GATTransformer

# We need torch_geometric for the data representation
try:
    from torch_geometric.data import Data
except ImportError:
    print("PyTorch Geometric is not installed. This script requires it for data representation.")
    Data = None

class SensorFusionDataset(Dataset):
    """
    Custom Dataset to handle 5-frame windows of sensor data.
    
    This class now simulates a more realistic data loading and preprocessing pipeline
    based on the user-specified data structures.
    """
    def __init__(self, data_path, num_samples=1000, frames_per_sample=5, k_neighbors=5):
        if Data is None:
            raise ImportError("Cannot create dataset, PyTorch Geometric is missing.")
            
        self.data_path = data_path
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample
        self.k_neighbors = k_neighbors
        
        # --- Placeholder Data Generation ---
        # This section now generates data that mimics the specified structures.
        # TODO: Replace this with your actual data loading logic.
        self.samples = []
        for _ in range(num_samples):
            sample_window = []
            for _ in range(self.frames_per_sample):
                # 1. Generate raw structured data for one frame
                raw_data = self._generate_raw_frame_data()
                
                # 2. Preprocess and fuse the raw data into a graph
                graph_data = self._preprocess_frame(raw_data)
                sample_window.append(graph_data)
            
            # TODO: The "ground truth" sigmas need to be generated or loaded.
            ground_truth_sigmas = torch.rand(1, 2)
            
            self.samples.append((sample_window, ground_truth_sigmas))

    def _generate_raw_frame_data(self):
        """Generates a single frame of dummy sensor data in the specified structure."""
        num_lidar_points = np.random.randint(20, 50)
        num_radar_points = np.random.randint(5, 15)
        
        lidar_data = [{'t': 123.45, 'x': np.random.rand(), 'y': np.random.rand(), 'intensity': np.random.rand()} for _ in range(num_lidar_points)]
        radar_data = [{'t': 123.45, 'x': np.random.rand(), 'y': np.random.rand(), 'SNR': np.random.rand() * 10, 'vel': np.random.rand()} for _ in range(num_radar_points)]
        odom_data = {'t': 123.45, 'x': np.random.rand(), 'y': np.random.rand(), 'yaw': np.random.rand(), 'linear_vel': np.random.rand(), 'angular_vel': np.random.rand()}
        
        return {'lidar': lidar_data, 'radar': radar_data, 'odom': odom_data}

    def _preprocess_frame(self, raw_data):
        """Fuses LiDAR, Radar, and Odom data into a single PyG graph object."""
        from sklearn.neighbors import kneighbors_graph

        lidar_pts = raw_data['lidar']
        radar_pts = raw_data['radar']
        odom = raw_data['odom']

        # --- Feature Engineering ---
        node_features = []
        positions = []

        # Odom features are common for all nodes in the frame
        odom_features = [odom['x'], odom['y'], odom['yaw'], odom['linear_vel'], odom['angular_vel']]

        # LiDAR features: [x, y, intensity, 0, 0, 0, odom..., is_lidar, is_radar] (padded)
        for pt in lidar_pts:
            features = [pt['x'], pt['y'], pt['intensity'], 0, 0, 0] + odom_features + [1, 0]
            node_features.append(features)
            positions.append([pt['x'], pt['y']])

        # Radar features: [x, y, 0, SNR, raw_vel, calibrated_vel, odom..., is_lidar, is_radar]
        for pt in radar_pts:
            # Velocity calibration
            raw_vel = pt['vel']
            # NOTE: This is a simplification. A proper rotation using odom['yaw'] is needed.
            calibrated_vel = odom['linear_vel'] + raw_vel 
            features = [pt['x'], pt['y'], 0, pt['SNR'], raw_vel, calibrated_vel] + odom_features + [0, 1]
            node_features.append(features)
            positions.append([pt['x'], pt['y']])
            
        if not node_features: # Handle empty frames
            # The feature count is now 6 (base) + 5 (odom) + 2 (one-hot) = 13
            return Data(x=torch.empty(0, 13), edge_index=torch.empty(2, 0))

        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        positions_tensor = torch.tensor(positions, dtype=torch.float)

        # --- Graph Construction (k-NN) ---
        if positions_tensor.shape[0] > 1:
            edge_index = kneighbors_graph(positions_tensor, self.k_neighbors, mode='connectivity')
            edge_index = torch.tensor(edge_index.toarray(), dtype=torch.long).nonzero().t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)

        return Data(x=node_features_tensor, edge_index=edge_index)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    dataset = SensorFusionDataset(args.data_path)
    # Collate_fn is needed if samples are lists of PyG Data objects
    def collate_fn(batch):
        return batch[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # --- Model, Loss, and Optimizer ---
    model = GATTransformer().to(device)
    criterion = torch.nn.MSELoss() # Mean Squared Error is a reasonable choice for regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (window, target_sigmas) in enumerate(dataloader):
            # Move data to device
            window = [frame.to(device) for frame in window]
            target_sigmas = target_sigmas.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_sigmas = model(window)
            
            # Calculate loss
            loss = criterion(predicted_sigmas, target_sigmas)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f} ---")

    # --- Save the trained model ---
    save_path = os.path.join(args.save_dir, 'gat_transformer.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAT+Transformer model for sensor data fusion.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--save_dir', type=str, default='../trained_models', help='Directory to save the trained model.')
    
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
