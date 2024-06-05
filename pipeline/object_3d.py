from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d
import plotly.graph_objects as go
import scipy
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from pipeline.object_instance import ObjectInstance
from utils.misc import sample_points



class Object3D:

    def __init__(self, object_instance: ObjectInstance, pcd: open3d.pybind.geometry.PointCloud, 
                 label: str=None, max_points: int=10000):

        self.max_points = max_points
        self.points = sample_points(np.unique(object_instance.mask_3d, axis=1), n=self.max_points)
        self.confidence = object_instance.confidence
        self.image_features = object_instance.image_features

        if label is None:
            # this assumes that all object_instance are going to have the same label
            self.label = object_instance.label 
        else:
            # otherwise you can assign explicitly the label
            self.label = label

        self.object_instances = [object_instance]

        self.pcd = pcd

        self._center_3d = None
        self._pcd_points = None
        self._pcd_mask = None

    @property
    def center_3d(self):
        if self._center_3d is None:
            self._center_3d = self._compute_center_3d()
        return self._center_3d

    def _compute_center_3d(self):
        points = self._pcd_points if self._pcd_points is not None else self.points
        return np.mean(points, axis=1)

    @property
    def pcd_points(self, n_sampled=500, proximity_thresh=0.1):
        if self._pcd_points is None:
            self._pcd_points, self._pcd_mask = self.select_pcd_points(n_sampled=500, proximity_thresh=0.1)
        return self._pcd_points

    @property
    def pcd_mask(self, n_sampled=500, proximity_thresh=0.1):
        if self._pcd_mask is None:
            self._pcd_points, self._pcd_mask = self.select_pcd_points(n_sampled=500, proximity_thresh=0.1)
        return self._pcd_mask

    def add_object_instance(self, object_instance: ObjectInstance):
        points = np.hstack([self.points, np.unique(object_instance.mask_3d, axis=1)])

        self.points = sample_points(points, n=self.max_points)

        n_objs = len(self.object_instances)
        # take the average of the previous image_features with the one coming from the new object_instance
        self.image_features = (n_objs * self.image_features + object_instance.image_features) / (n_objs + 1)
        self.confidence = (n_objs * self.confidence + object_instance.confidence) / (n_objs + 1)

        self.object_instances.append(object_instance)
        self._center_3d = self._compute_center_3d()


    def plot_3d(self, fig=None, subsample=False, show=True):
        if fig is None:
            fig = go.Figure()

        num_points = self.points.shape[1]

        # Subsample for viz, if too many points:
        if subsample:
            subsample_points = num_points // 10
            indices = np.linspace(0, self.points.shape[1] - 1, subsample_points, dtype=int)
            points = self.points[:, indices]
        else:
            points = self.points

        color = np.random.rand(3,) * 255
        fig.add_trace(go.Scatter3d(
            x=points[0, :],
            y=points[1, :],
            z=points[2, :],
            mode='markers',
            marker=dict(size=2, color=f'rgb({color[0]}, {color[1]}, {color[2]})', opacity=0.8),
            name=f'{self.label}'
        ))

        fig.update_layout(title='Object 3D',
                        scene=dict(xaxis_title='X Axis',
                                    yaxis_title='Y Axis',
                                    zaxis_title='Z Axis',
                                    aspectmode='cube'),
                        legend={
                            'itemsizing': 'constant'
                        })

        
        if show:
            fig.show()

    def plot_2d(self, scale_factor=0.2):

        num_plots = len(self.object_instances)

        # Define the number of rows and columns for subplots
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        # Flatten the axes array for easy iteration if it's 2D
        if rows * cols > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
        
        for i, object_instance in enumerate(self.object_instances):
            object_instance.plot_2d(axs[i], scale_factor)
        
        plt.tight_layout()
        plt.show()


    def denoise_points(self, eps=1, min_samples=100, mode='biggest_cluster'):
        # mode in ['outliers', 'biggest_cluster']
        # apply DBSCAN clustering to remove outliers from the point cloud
        err_msg = "mode not recognized! it needs to be in ['outliers', 'biggest_cluster']"
        assert mode in ['outliers', 'biggest_cluster'], err_msg
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.points.T)

        if mode == 'outliers':
            # outliers have label -1
            ids_denoised = np.where(labels >= 0)[0]
        
        else:
            dict_count = dict(Counter(labels))
            print(dict_count)

            # find biggest cluster
            biggest_cluster = -1
            biggest_count = 0
            for label in dict_count:
                if dict_count[label] > biggest_count:
                    biggest_cluster = label
                    biggest_count = dict_count[label]

            # return only the biggest cluster
            ids_denoised = np.where(labels == biggest_cluster)[0]

        self.points = self.points[:,ids_denoised]

    def sampled_points(self, n_points):
        return sample_points(self.points, n_points)


    def select_pcd_points(self, n_sampled=500, proximity_thresh=0.1):
        
        pcd_points = np.asarray(self.pcd.points).T
        points = self.sampled_points(n_sampled)

        # 1. Select region of the pcd close to our object_3d
        #pcd_bbox, mask_pcd_bbox = self._select_roi(pcd_points, points, thresh=proximity_thresh/2)
        pcd_bbox, mask_pcd_bbox = self._select_roi(pcd_points, points, thresh=0)

        # 2. compute the distance between the points in the pcd and in the object_3d
        distances = scipy.spatial.distance_matrix(pcd_bbox.T, points.T)

        # 3. select points in the pt cloud based on their distance to the points in the pcd
        nn_distances = distances.min(axis=1)  # nearest neighbours distances
        mask_bbox_selected = nn_distances < proximity_thresh
        mask_pcd_selected = self._combine_masks(mask_pcd_bbox, mask_bbox_selected)

        pcd_points = pcd_bbox[:, mask_bbox_selected]
        pcd_mask = mask_pcd_selected

        return pcd_points, pcd_mask


    @staticmethod
    def _select_roi(points_target: np.array, points_reference:np.array, thresh=0.1):
        """
        Given a target point cloud and a reference point cloud, select the region of
        interest in the target point cloud which is enclosed in a bounding box that 
        tightly fit the reference point cloud. Use thresh to create some space between 
        the bounding box and the reference point cloud.

        Parameters
            - points_target (np.array): the point cloud that we want to crop. The
            shape is (3, n_points_target)
            - points_reference: a point cloud more confined in space than the
            points_target. We create a bbox around it that we use to crop the 
            points_target. The shape is (3, n_points_reference)

        Returns
            - points_target_bbox (np.array)
            - roi_mask (np.array): boolean mask to select points_target_bbox from 
            points_target
        """
        max_values = points_reference.max(axis=1)
        min_values = points_reference.min(axis=1)

        roi_mask = (points_target[0,:] > min_values[0] - thresh) & \
                (points_target[0,:] < max_values[0] + thresh) & \
                (points_target[1,:] > min_values[1] - thresh) & \
                (points_target[1,:] < max_values[1] + thresh) & \
                (points_target[2,:] > min_values[2] - thresh) & \
                (points_target[2,:] < max_values[2] + thresh)

        points_target_bbox = points_target[:,roi_mask]

        return points_target_bbox, roi_mask


    @staticmethod
    def _combine_masks(mask01: np.array, mask12: np.array) -> np.array:
        """
        The masks are such that len(mask01) = n, 
        mask01.sum() = m = len(mask12), and mask12.sum() = k.
        We are going to output the mask02 s.t. 
        len(mask02) = n and mask02.sum() = k
        """

        mask02 = mask01.copy()
        mask02[mask02] = mask12
        return mask02

    def plot_pcd(self, point_sample_factor=0.1, margin=1):

        pcd_points = np.asarray(self.pcd.points).T
        # select an area around the object for better visualization
        smaller_pcd, mask_smaller_pcd = self._select_roi(pcd_points, self.points, thresh=margin)

        pcd_selected = self.pcd.select_by_index(np.where(self.pcd_mask)[0])
        pcd_out = self.pcd.select_by_index(np.where(~self.pcd_mask & mask_smaller_pcd)[0])

        pcd_out = pcd_out.paint_uniform_color([0.8, 0.8, 0.8])

        open3d.visualization.draw_plotly([pcd_selected, pcd_out], point_sample_factor=point_sample_factor)



def plot_objects_3d(objects_3d: List[Object3D], subsample=False):
    fig = go.Figure()
    for object_3d in objects_3d:
        object_3d.plot_3d(fig=fig, subsample=subsample, show=False)
    fig.show()


def denoise_objects_3d(objects_3d: List[Object3D], eps=0.1, min_samples=10, mode='outliers'):

    for object_3d in tqdm(objects_3d, desc='Denoising point clouds'):
        object_3d.denoise_points(eps=eps, min_samples=min_samples, mode=mode)
    

def filter_objects_3d(object_label: str, all_objects: List[Object3D]) -> List[Object3D]:
    # all_objects is a list of object instances
    return [obj for obj in all_objects if obj.label==object_label]