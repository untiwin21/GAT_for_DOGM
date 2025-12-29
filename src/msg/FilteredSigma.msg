# Custom message to publish the learned sigma values from the GAT+Transformer model.

# Header for timestamp and frame_id
std_msgs/Header header

# The predicted standard deviation for LiDAR measurement positions (x, y).
# This single value can be used for both x and y, assuming isotropic uncertainty.
float32 lidar_position_sigma

# The predicted standard deviation for Radar measurement velocities.
float32 radar_velocity_sigma
