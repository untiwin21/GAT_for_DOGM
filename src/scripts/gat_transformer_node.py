#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
# TODO: Define and import custom message for Radar and Sigma
# from dogm_msgs.msg import RadarData, FilteredSigma

import torch
import numpy as np
from collections import deque
import message_filters
from sklearn.neighbors import kneighbors_graph

# Make sure the model is importable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import GATTransformer, GATConv
if GATConv is not None:
    from torch_geometric.data import Data


class GATTransformerNode:
    def __init__(self):
        rospy.init_node('gat_transformer_node', anonymous=True)

        if GATConv is None:
            rospy.logerr("PyTorch Geometric not found. Shutting down.")
            return

        # --- Parameters ---
        self.lidar_topic = rospy.get_param('~lidar_topic', '/lidar/points')
        self.radar_topic = rospy.get_param('~radar_topic', '/radar/points') # Assuming PointCloud2 for radar too
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.FRAME_WINDOW_SIZE = 5
        self.K_NEIGHBORS = 5

        # --- Data Buffer for 5-frame window ---
        self.frame_window_buffer = deque(maxlen=self.FRAME_WINDOW_SIZE)

        # --- Model ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GATTransformer().to(self.device)
        # TODO: Load pre-trained model weights
        # model_path = rospy.get_param('~model_path', 'path/to/weights.pth')
        # self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        rospy.loginfo(f"Model loaded on {self.device}.")

        # --- Subscribers using message_filters for synchronization ---
        lidar_sub = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        radar_sub = message_filters.Subscriber(self.radar_topic, PointCloud2) # ASSUMING PointCloud2 for Radar
        odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)

        # TODO: Adjust queue_size and slop for real-world conditions
        ts = message_filters.ApproximateTimeSynchronizer([lidar_sub, radar_sub, odom_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synchronized_callback)

        # --- Publisher ---
        # self.sigma_pub = rospy.Publisher('/filtered_sigma', FilteredSigma, queue_size=10)
        
        rospy.loginfo("GAT+Transformer Node initialized and waiting for synchronized messages.")

    def synchronized_callback(self, lidar_msg, radar_msg, odom_msg):
        """
        Callback for synchronized Lidar, Radar, and Odom data.
        It preprocesses the data into a graph and adds it to the window buffer.
        """
        # 1. Parse ROS messages into the specified dictionary structure
        raw_data = self._parse_ros_messages(lidar_msg, radar_msg, odom_msg)
        
        # 2. Preprocess and fuse the raw data into a graph object
        graph_data = self._preprocess_frame(raw_data)
        
        # 3. Add the processed frame to our window buffer
        self.frame_window_buffer.append(graph_data)
        
        # 4. If the buffer is full, run inference
        if len(self.frame_window_buffer) == self.FRAME_WINDOW_SIZE:
            self.run_inference()

    def _parse_ros_messages(self, lidar_msg, radar_msg, odom_msg):
        """Parses ROS messages into a structured dictionary."""
        # Lidar: {t, x, y, intensity}
        # Assuming fields 'x', 'y', 'intensity' exist in the PointCloud2
        lidar_generator = pc2.read_points(lidar_msg, field_names=("x", "y", "intensity"), skip_nans=True)
        lidar_data = [{'t': lidar_msg.header.stamp.to_sec(), 'x': p[0], 'y': p[1], 'intensity': p[2]} for p in lidar_generator]

        # Radar: {t, x, y, SNR, vel}
        # NOTE: This is a MAJOR assumption. Radar data might not be PointCloud2.
        # This needs to be adapted to the actual radar message type.
        # Assuming fields 'x', 'y', 'snr', 'velocity'
        try:
            radar_generator = pc2.read_points(radar_msg, field_names=("x", "y", "snr", "velocity"), skip_nans=True)
            radar_data = [{'t': radar_msg.header.stamp.to_sec(), 'x': p[0], 'y': p[1], 'SNR': p[2], 'vel': p[3]} for p in radar_generator]
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Could not parse radar message, likely not a PointCloud2 with expected fields. Error: {e}")
            radar_data = []

        # Odom: {t, x, y, yaw, linear_vel, angular_vel}
        from tf.transformations import euler_from_quaternion
        orientation_q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        odom_data = {
            't': odom_msg.header.stamp.to_sec(),
            'x': odom_msg.pose.pose.position.x,
            'y': odom_msg.pose.pose.position.y,
            'yaw': yaw,
            'linear_vel': odom_msg.twist.twist.linear.x,
            'angular_vel': odom_msg.twist.twist.angular.z
        }
        
        return {'lidar': lidar_data, 'radar': radar_data, 'odom': odom_data}

    def _preprocess_frame(self, raw_data):
        """Fuses LiDAR, Radar, and Odom data into a single PyG graph object."""
        lidar_pts = raw_data['lidar']
        radar_pts = raw_data['radar']
        odom = raw_data['odom']

        node_features, positions = [], []
        odom_features = [odom['x'], odom['y'], odom['yaw'], odom['linear_vel'], odom['angular_vel']]

        # LiDAR features: [x, y, intensity, 0, 0, 0, odom..., is_lidar, is_radar] (padded)
        for pt in lidar_pts:
            features = [pt['x'], pt['y'], pt['intensity'], 0, 0, 0] + odom_features + [1, 0]
            node_features.append(features)
            positions.append([pt['x'], pt['y']])

        # Radar features: [x, y, 0, SNR, raw_vel, calibrated_vel, odom..., is_lidar, is_radar]
        for pt in radar_pts:
            raw_vel = pt['vel']
            # NOTE: This is a simplification. A proper rotation using odom['yaw'] is needed.
            calibrated_vel = odom['linear_vel'] + raw_vel
            features = [pt['x'], pt['y'], 0, pt['SNR'], raw_vel, calibrated_vel] + odom_features + [0, 1]
            node_features.append(features)
            positions.append([pt['x'], pt['y']])

        if not node_features:
            # The feature count is now 6 (base) + 5 (odom) + 2 (one-hot) = 13
            return Data(x=torch.empty(0, 13, device=self.device), edge_index=torch.empty(2, 0, device=self.device))

        node_features_tensor = torch.tensor(node_features, dtype=torch.float, device=self.device)
        positions_tensor = torch.tensor(positions, dtype=torch.float)

        if positions_tensor.shape[0] > 1:
            edge_index = kneighbors_graph(positions_tensor, self.K_NEIGHBORS, mode='connectivity')
            edge_index = torch.tensor(edge_index.toarray(), dtype=torch.long, device=self.device).nonzero().t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)

        return Data(x=node_features_tensor, edge_index=edge_index)

    def run_inference(self):
        """
        Runs model inference on the 5-frame window and publishes the result.
        """
        rospy.loginfo("Processing a 5-frame window for inference...")
        
        # The buffer is ready, convert it to a list
        frame_window = list(self.frame_window_buffer)

        # --- Model Inference ---
        with torch.no_grad():
            predicted_sigmas = self.model(frame_window)
        
        predicted_sigmas = predicted_sigmas.cpu().numpy()

        # --- Publish Results ---
        # TODO: Create and publish the custom FilteredSigma message
        # The message should contain the predicted sigmas and an index/timestamp
        # to map them back to the original data points in the C++ node.
        # For now, just log the output.
        
        rospy.loginfo(f"Predicted Sigmas [LiDAR_pos, Radar_vel]: {predicted_sigmas[0]}")

        # The deque with maxlen will automatically discard the oldest frame on the next append.
        # No need to clear it manually if we want a sliding window.

if __name__ == '__main__':
    try:
        GATTransformerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except ImportError as e:
        rospy.logerr(f"Failed to import dependencies: {e}")


