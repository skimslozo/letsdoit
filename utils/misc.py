from typing import List, Dict
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

def inverseRigid(H):
    H_R = H[0:3, 0:3]
    H_T = H[0:3, 3]

    invH = np.eye(4)
    invH[0:3, 0:3] = H_R.T
    invH[0:3, 3] = -H_R.T @ H_T

    return invH

def select_ids(array, ids):
    return [array[idx] for idx in ids]

def sample_points(points, n=1000):
    # points have shape (3, n_points)
    return points[:, np.random.randint(0, points.shape[1], size=n)]


def number2str(i: int, n_digits: int=3):

    assert isinstance(i, int), 'i must be an int!'

    # count how many digits does i already have
    k = i
    i_digits = 1
    while k >= 10:
        k /= 10
        i_digits += 1

    n_digits = max(i_digits, n_digits)
    diff_digits = n_digits - i_digits

    str_number = '0' * diff_digits + str(i)

    return str_number

def get_instruction(instruction_dict: List[Dict], visit_id: str, desc_id: str) -> str:
    instructions_visit = [iv for iv in instruction_dict if iv['visit_id']==visit_id][0]
    instruction = [ins['instruction'] for ins in instructions_visit['instructions'] if ins['desc_id']==desc_id][0]
    return instruction

def plot_point_cloud_and_object(point_cloud, object_array, subsample_ratio=0.1):
    """
    Plots a subsampled Open3D point cloud and a 3D object array (Nx3) using Plotly.
    
    Parameters:
    - point_cloud: open3d.geometry.PointCloud
        The input point cloud to be subsampled and plotted.
    - object_array: np.ndarray
        A 3D array (Nx3) representing the 3D object to be plotted.
    - subsample_ratio: float
        The ratio of points to keep when subsampling the point cloud. Default is 0.1.
    """
    
    # Validate inputs
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("point_cloud must be an instance of open3d.geometry.PointCloud")
    if not isinstance(object_array, np.ndarray) or object_array.shape[1] != 3:
        raise ValueError("object_array must be a numpy array with shape (Nx3)")
    
    # Subsample the point cloud
    num_points = len(point_cloud.points)
    subsample_size = int(num_points * subsample_ratio)
    indices = np.random.choice(num_points, subsample_size, replace=False)
    subsampled_points = np.asarray(point_cloud.points)[indices]
    
    # Create Plotly scatter plot for point cloud
    point_cloud_trace = go.Scatter3d(
        x=subsampled_points[:, 0],
        y=subsampled_points[:, 1],
        z=subsampled_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.8),
        name='Point Cloud'
    )
    
    # Create Plotly scatter plot for object array
    object_trace = go.Scatter3d(
        x=object_array[:, 0],
        y=object_array[:, 1],
        z=object_array[:, 2],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.8),
        name='3D Object'
    )
    
    # Define the layout
    layout = go.Layout(
        title='3D Point Cloud and Object',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        showlegend=True
    )
    
    # Create the figure
    fig = go.Figure(data=[point_cloud_trace, object_trace], layout=layout)
    
    # Show the figure
    fig.show()