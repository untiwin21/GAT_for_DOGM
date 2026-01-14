
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def se2_apply(pose, points):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return (points @ R.T) + np.array([x, y])

def visualize_v6():
    data = np.load('sigma_v6.npz')
    poses = data['pose']
    F = int(data['F'])

    lidar_out = data['lidar_out']
    lidar_off = data['lidar_off']
    r1_out = data['r1_out']
    r1_off = data['r1_off']
    r2_out = data['r2_out']
    r2_off = data['r2_off']

    all_lidar_pts = []
    all_lidar_sigmas = []
    all_radar_pts = []
    all_radar_sigmas = []

    # The first few frames have no output, start from where there is data
    start_frame = 0
    for i in range(F):
        if lidar_off[i+1] > lidar_off[i] or r1_off[i+1] > r1_off[i] or r2_off[i+1] > r2_off[i]:
            start_frame = i
            break

    for f in range(start_frame, F):
        pose = poses[f]

        # Process LiDAR
        l_start, l_end = lidar_off[f], lidar_off[f+1]
        if l_end > l_start:
            lidar_frame_data = lidar_out[l_start:l_end]
            points_sensor_frame = lidar_frame_data[:, :2]
            points_world_frame = se2_apply(pose, points_sensor_frame)
            
            all_lidar_pts.extend(points_world_frame)
            sigmas = np.sqrt(lidar_frame_data[:, 2]**2 + lidar_frame_data[:, 3]**2)
            all_lidar_sigmas.extend(sigmas)

        # Process Radar1
        r1_start, r1_end = r1_off[f], r1_off[f+1]
        if r1_end > r1_start:
            radar1_frame_data = r1_out[r1_start:r1_end]
            points_sensor_frame = radar1_frame_data[:, :2]
            points_world_frame = se2_apply(pose, points_sensor_frame)

            all_radar_pts.extend(points_world_frame)
            sigmas_v = radar1_frame_data[:, 3]
            all_radar_sigmas.extend(sigmas_v)
            
        # Process Radar2
        r2_start, r2_end = r2_off[f], r2_off[f+1]
        if r2_end > r2_start:
            radar2_frame_data = r2_out[r2_start:r2_end]
            points_sensor_frame = radar2_frame_data[:, :2]
            points_world_frame = se2_apply(pose, points_sensor_frame)

            all_radar_pts.extend(points_world_frame)
            sigmas_v = radar2_frame_data[:, 3]
            all_radar_sigmas.extend(sigmas_v)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 20))

    if all_lidar_pts:
        all_lidar_pts = np.array(all_lidar_pts)
        all_lidar_sigmas = np.array(all_lidar_sigmas)
        # Normalize sigmas for color mapping
        lidar_norm = mcolors.Normalize(vmin=np.percentile(all_lidar_sigmas, 5), vmax=np.percentile(all_lidar_sigmas, 95))
        lidar_cmap = plt.cm.Greens
        
        ax.scatter(all_lidar_pts[:, 0], all_lidar_pts[:, 1], 
                   c=all_lidar_sigmas, cmap=lidar_cmap, norm=lidar_norm, 
                   marker='s', s=5, label='LiDAR')

    if all_radar_pts:
        all_radar_pts = np.array(all_radar_pts)
        all_radar_sigmas = np.array(all_radar_sigmas)
        # Normalize sigmas for color mapping
        radar_norm = mcolors.Normalize(vmin=np.percentile(all_radar_sigmas, 5), vmax=np.percentile(all_radar_sigmas, 95))
        radar_cmap = plt.cm.Reds

        ax.scatter(all_radar_pts[:, 0], all_radar_pts[:, 1], 
                   c=all_radar_sigmas, cmap=radar_cmap, norm=radar_norm, 
                   marker='o', s=5, label='Radar')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('LiDAR and Radar Points for v6 (colored by uncertainty)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig('v6_visualization.png', dpi=300)
    print("Saved visualization to v6_visualization.png")

if __name__ == '__main__':
    visualize_v6()
