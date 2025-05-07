import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
# 画一帧的坐标系，主动和被动用不同的颜色
def draw_frame(image, transform_matrix, camera_intrinsics=None, axis_length=0.1, colors=((0, 0, 255), (0, 255, 0), (255, 0, 0))):
    """
    Draw a coordinate frame on the given image based on the transform matrix and camera intrinsics.

    Args:
        image (numpy.ndarray): The input image.
        transform_matrix (numpy.ndarray): A 4x4 transformation matrix.
        camera_intrinsics (numpy.ndarray): A 3x3 camera intrinsic matrix.
        axis_length (float): Length of the coordinate axis in the 3D space.
        colors (tuple): Colors for the X, Y, Z axes in BGR format.

    Returns:
        numpy.ndarray: The image with the coordinate frame drawn.
    """
    if transform_matrix.shape != (4, 4):
        raise ValueError("transform_matrix must be a 4x4 matrix.")

    if camera_intrinsics is None:   
        camera_intrinsics_dict = json.load(open('/home/yan20/tianshuwu/coi/test_data/000037/camera_intrinsics.json'))
        camera_intrinsics = np.array([[camera_intrinsics_dict['fx'], 0, camera_intrinsics_dict['cx']],
                                      [0, camera_intrinsics_dict['fy'], camera_intrinsics_dict['cy']],
                                      [0, 0, 1]])

    if camera_intrinsics.shape != (3, 3):
        raise ValueError("camera_intrinsics must be a 3x3 matrix.")



    # Define the origin and axes points in the object frame (homogeneous coordinates)
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([axis_length, 0, 0, 1])
    y_axis = np.array([0, axis_length, 0, 1])
    z_axis = np.array([0, 0, axis_length, 1])

    # Transform points to the camera frame
    origin_cam = transform_matrix @ origin
    x_axis_cam = transform_matrix @ x_axis
    y_axis_cam = transform_matrix @ y_axis
    z_axis_cam = transform_matrix @ z_axis

    # Project points onto the image plane
    def project_point(point):
        point_normalized = point[:3] / point[2]
        return camera_intrinsics @ point_normalized

    origin_2d = project_point(origin_cam)
    x_axis_2d = project_point(x_axis_cam)
    y_axis_2d = project_point(y_axis_cam)
    z_axis_2d = project_point(z_axis_cam)

    # Convert points to pixel coordinates
    def to_pixel_coords(point_2d):
        return tuple(np.round(point_2d[:2]).astype(int))

    origin_pixel = to_pixel_coords(origin_2d)
    x_axis_pixel = to_pixel_coords(x_axis_2d)
    y_axis_pixel = to_pixel_coords(y_axis_2d)
    z_axis_pixel = to_pixel_coords(z_axis_2d)

    # Draw the axes on the image
    output_image = image.copy()
    cv2.arrowedLine(output_image, origin_pixel, x_axis_pixel, colors[0], 2, tipLength=0.4)
    cv2.arrowedLine(output_image, origin_pixel, y_axis_pixel, colors[1], 2, tipLength=0.4)
    cv2.arrowedLine(output_image, origin_pixel, z_axis_pixel, colors[2], 2, tipLength=0.4)


    return output_image


def save_frames_from_avi(video_path, output_folder, image_prefix="frame"):
    """
    Reads a video file and saves each frame as an image.

    Args:
        video_path (str): Path to the input .avi video file.
        output_folder (str): Directory where the frames will be saved.
        image_prefix (str): Prefix for the saved image files.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frame_count = 0
    
    while True:
        # Read a single frame
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Construct the filename for the frame
        frame_filename = os.path.join(output_folder, f"{image_prefix}_{frame_count:04d}.png")
        
        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count} to {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    # print(f"Finished saving {frame_count} frames to {output_folder}")

def feature_pca_vis(point_source,point_target,feature_source,feature_target,addtional_feature=None):
    '''
    point: n,3
    feature: n,c
    '''
    pca = PCA(n_components=3)
    all_feature = np.concatenate([feature_source,feature_target],axis=0)
    if addtional_feature is not None:
        all_feature = np.concatenate([all_feature,addtional_feature],axis=0)
    pca.fit(all_feature)
    feature_source_pca = pca.transform(feature_source)
    feature_target_pca = pca.transform(feature_target)
    
    scaler = MinMaxScaler()
    feature_source_pca_scaled = scaler.fit_transform(feature_source_pca)
    feature_target_pca_scaled = scaler.transform(feature_target_pca)
    
    # 浅一点的颜色
    feature_source_pca_scaled = 0.3 + 0.7 * feature_source_pca_scaled
    feature_target_pca_scaled = 0.3 + 0.7 * feature_target_pca_scaled


    
    # Create scatter plots for source and target points with PCA-transformed features
    # Create subplots for separate visualization
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Source Points", "Target Points")
    )
    
    # Source points visualization
    fig.add_trace(go.Scatter3d(
        x=point_source[:, 0],
        y=point_source[:, 1],
        z=point_source[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=feature_source_pca_scaled[:],  # Color by first PCA component
            # colorscale='Viridis',
            opacity=0.8
        ),
        name='Source Points'
    ), row=1, col=1)

    # Target points visualization
    fig.add_trace(go.Scatter3d(
        x=point_target[:, 0],
        y=point_target[:, 1],
        z=point_target[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=feature_target_pca_scaled[:],  # Color by first PCA component
            # colorscale='Viridis',
            opacity=0.8 
        ),
        name='Target Points'
    ), row=1, col=2)

    # Update layout to remove axes and background
    fig.update_layout(
        title="PCA Feature Visualization in Separate Windows",
        scene=dict(
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            zaxis=dict(visible=False),  # Hide z-axis
            bgcolor="white"            # Set background to white
        ),
        scene2=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white"
        ),
        paper_bgcolor="white",         # Set outer background to white
        width=1400,
        height=700
    )

    # Show the plot
    fig.show()
    
    
def show_match_color(point_source,point_target,match_id):
    
    match = np.concatenate([point_source[match_id[:,0]],point_target[match_id[:,1]]],axis=1)
    
    # Extract coordinates for the first point cloud (x1, y1, z1)
    x1 = match[:, 0]
    y1 = match[:, 1]
    z1 = match[:, 2]

    # cube_length_1 = 1.5*np.std(match[:, :3]) # 用3sigma方差来可视化吧
    # point_center_1 = match[:, :3].mean(axis=(0))

    # Extract coordinates for the second point cloud (x2, y2, z2)
    x2 = match[:, 3]
    y2 = match[:, 4]
    z2 = match[:, 5]

    # cube_length_2 = 1.5*np.std(match[:, 3:]) # 用3sigma方差来可视化吧
    # point_center_2 = match[:, 3:].mean(axis=(0))

    # Normalize x1, y1, z1 to [0, 1] for RGB mapping
    norm_x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
    norm_y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
    norm_z1 = (z1 - np.min(z1)) / (np.max(z1) - np.min(z1))

    # Combine normalized x1, y1, z1 into RGB colors
    colors = np.stack([norm_x1, norm_y1, norm_z1], axis=1)

    # Convert RGB values to hexadecimal for Plotly
    rgb_colors = ['rgb({}, {}, {})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    # Create subplots for visualizing both point clouds
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('source point', 'target point'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        # horizontal_spacing=0.05
    )

    # Point cloud 1
    trace1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=8,
            color=rgb_colors,  # Apply RGB colors
            opacity=0.8
        ),
        name='source point'
    )

    # Point cloud 2 (use the same colors as Point Cloud 1)
    trace2 = go.Scatter3d(
        x=x2,
        y=y2,
        z=z2,
        mode='markers',
        marker=dict(
            size=8,
            color=rgb_colors,  # Apply the same RGB colors
            opacity=0.8
        ),
        name='target point'
    )

    # Add traces to the figure
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    # Update layout
    fig.update_layout(
        title="3D Point Clouds with RGB Colors",
    scene=dict(
        xaxis=dict(visible=False),# range=[point_center_1[0]-cube_length_1/2,point_center_1[0]+cube_length_1/2]),
        yaxis=dict(visible=False),#range=[point_center_1[1]-cube_length_1/2,point_center_1[1]+cube_length_1/2]),
        zaxis=dict(visible=False),# range=[point_center_1[2]-cube_length_1/2,point_center_1[2]+cube_length_1/2]),
        # aspectmode='cube'
        bgcolor="white"
    ),
    scene2=dict(
        xaxis=dict(visible=False),# range=[point_center_2[0]-cube_length_2/2,point_center_2[0]+cube_length_2/2]),
        yaxis=dict(visible=False),#range=[point_center_2[1]-cube_length_2/2,point_center_2[1]+cube_length_2/2]),
        zaxis=dict(visible=False),# range=[point_center_2[2]-cube_length_2/2,point_center_2[2]+cube_length_2/2]),
        # aspectmode='cube',
        bgcolor="white"
    ) ,      paper_bgcolor="white",         # Set outer background to white
        width=1400,
        height=700
    )

    # Show the figure
    fig.show()


def show_match_projection(point_source, point_target, match_id, source_image, target_image, K_source, K_target):
    """
    Visualize matches by projecting 3D points onto two images using OpenCV.

    :param point_source: (N, 3) array of source 3D points.
    :param point_target: (M, 3) array of target 3D points.
    :param match_id: (P, 2) array of matching indices between source and target points.
    :param source_image: Path to the source image.
    :param target_image: Path to the target image.
    :param K_source: (3, 3) intrinsic matrix of the source camera.
    :param K_target: (3, 3) intrinsic matrix of the target camera.
    """
    # Load images
    img1 = source_image.copy()
    img2 = target_image.copy()

    # Extract matching points
    matched_source = point_source[match_id[:, 0]]  # (P, 3)
    matched_target = point_target[match_id[:, 1]]  # (P, 3)

    # Normalize for color mapping
    norm_x1 = (matched_source[:, 0] - np.min(matched_source[:, 0])) / (np.max(matched_source[:, 0]) - np.min(matched_source[:, 0]))
    norm_y1 = (matched_source[:, 1] - np.min(matched_source[:, 1])) / (np.max(matched_source[:, 1]) - np.min(matched_source[:, 1]))
    norm_z1 = (matched_source[:, 2] - np.min(matched_source[:, 2])) / (np.max(matched_source[:, 2]) - np.min(matched_source[:, 2]))

    # Colors for visualization
    colors = np.stack([norm_x1, norm_y1, norm_z1], axis=1)

    # Project 3D points to 2D for source and target
    def project_points(points_3d, K):
        points_2d = K @ points_3d.T
        points_2d /= points_2d[2, :]  # Normalize by z to get (u, v)
        return points_2d[:2, :].T  # Return (N, 2)

    projected_source = project_points(matched_source, K_source)
    projected_target = project_points(matched_target, K_target)

    # Overlay points on images with transparency and outlines
    def overlay_points_with_effects(img, points, colors, alpha=1.0):
        overlay = img.copy()
        for (x, y), (r, g, b) in zip(points, colors):
            x, y = int(round(x)), int(round(y))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                # Draw circle with color
                cv2.circle(overlay, (x, y), 3, (int(b * 255), int(g * 255), int(r * 255)), -1)  # Filled circle
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



    overlay_points_with_effects(img1, projected_source, colors)
    overlay_points_with_effects(img2, projected_target, colors)


    return img1, img2



from scipy.spatial import cKDTree


def upsample_point_cloud(points, colors, num_samples=2):
    """
    Upsample a point cloud by interpolating between neighboring points.

    :param points: (N, 3) array of 3D points.
    :param colors: (N, 3) array of RGB colors corresponding to points.
    :param num_samples: Number of new points to generate per original point.
    :return: Upsampled points and colors.
    """
    # Create a KDTree for finding neighbors
    kdtree = cKDTree(points)

    # Storage for new points and colors
    new_points = []
    new_colors = []

    for i, point in enumerate(points):
        # Find neighbors within a certain radius
        distances, indices = kdtree.query(point, k=32)  # Take 3 nearest neighbors (plus itself)

        if len(indices) < 2:
            continue  # Skip isolated points

        # Interpolate between neighbors
        for _ in range(num_samples):
            # Randomly select two neighbors
            neighbor_indices = np.random.choice(indices[1:], size=2, replace=False)
            neighbor_points = points[neighbor_indices]
            neighbor_colors = colors[neighbor_indices]

            # Generate random weights for interpolation
            weights = np.random.rand(2)
            weights /= np.sum(weights)  # Normalize to sum to 1

            # Interpolate position and color
            new_point = np.dot(weights, neighbor_points)
            new_color = np.dot(weights, neighbor_colors)

            new_points.append(new_point)
            new_colors.append(new_color)

    # Combine original and new points/colors
    all_points = np.vstack([points, np.array(new_points)])
    all_colors = np.vstack([colors, np.array(new_colors)])

    return all_points, all_colors


def project_points(points_3d, K):
    """
    Project 3D points into 2D using the intrinsic matrix.

    :param points_3d: (N, 3) array of 3D points.
    :param K: (3, 3) camera intrinsic matrix.
    :return: (N, 2) array of 2D points in image coordinates.
    """
    points_2d = K @ points_3d.T
    points_2d /= points_2d[2, :]  # Normalize by z to get (u, v)
    return points_2d[:2, :].T  # Return (N, 2)


def overlay_points_with_effects(img, points, colors):
    """
    Overlay points onto an image with effects.

    :param img: Image to overlay points on.
    :param points: (N, 2) array of 2D points.
    :param colors: (N, 3) array of RGB colors (values between 0 and 1).
    """
    overlay = img.copy()
    for (x, y), (r, g, b) in zip(points, colors):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            # Draw circle with color
            cv2.circle(overlay, (x, y), 3, (int(b * 255), int(g * 255), int(r * 255)), -1)  # Filled circle
    return overlay


def show_match_projection_with_upsampling(point_source, point_target, match_id, source_image, target_image, K_source, K_target):
    """
    Visualize matches by projecting upsampled 3D points onto two images.

    :param point_source: (N, 3) array of source 3D points.
    :param point_target: (M, 3) array of target 3D points.
    :param match_id: (P, 2) array of matching indices between source and target points.
    :param source_image: Source image (numpy array).
    :param target_image: Target image (numpy array).
    :param K_source: (3, 3) intrinsic matrix of the source camera.
    :param K_target: (3, 3) intrinsic matrix of the target camera.
    """
    # Extract matching points
    matched_source = point_source[match_id[:, 0]]
    matched_target = point_target[match_id[:, 1]]

    # Normalize for color mapping (retain original color logic)
    norm_x1 = (matched_source[:, 0] - np.min(matched_source[:, 0])) / (np.max(matched_source[:, 0]) - np.min(matched_source[:, 0]))
    norm_y1 = (matched_source[:, 1] - np.min(matched_source[:, 1])) / (np.max(matched_source[:, 1]) - np.min(matched_source[:, 1]))
    norm_z1 = (matched_source[:, 2] - np.min(matched_source[:, 2])) / (np.max(matched_source[:, 2]) - np.min(matched_source[:, 2]))
    colors_source = np.stack([norm_x1, norm_y1, norm_z1], axis=1)

    norm_x2 = (matched_target[:, 0] - np.min(matched_target[:, 0])) / (np.max(matched_target[:, 0]) - np.min(matched_target[:, 0]))
    norm_y2 = (matched_target[:, 1] - np.min(matched_target[:, 1])) / (np.max(matched_target[:, 1]) - np.min(matched_target[:, 1]))
    norm_z2 = (matched_target[:, 2] - np.min(matched_target[:, 2])) / (np.max(matched_target[:, 2]) - np.min(matched_target[:, 2]))
    colors_target = np.stack([norm_x2, norm_y2, norm_z2], axis=1)

    # Upsample the points
    upsampled_source, upsampled_colors_source = upsample_point_cloud(matched_source, colors_source, num_samples=5)
    upsampled_target, upsampled_colors_target = upsample_point_cloud(matched_target, colors_target, num_samples=5)

    # Upsample the points
    for i in range(2):
        upsampled_source, upsampled_colors_source = upsample_point_cloud(upsampled_source, upsampled_colors_source, num_samples=5)
        upsampled_target, upsampled_colors_target = upsample_point_cloud(upsampled_target, upsampled_colors_target, num_samples=5)

    # Project the points
    projected_source = project_points(upsampled_source, K_source)
    projected_target = project_points(upsampled_target, K_target)

    # Overlay points onto images
    img1 = overlay_points_with_effects(source_image.copy(), projected_source, upsampled_colors_source)
    img2 = overlay_points_with_effects(target_image.copy(), projected_target, upsampled_colors_target)

    return img1, img2