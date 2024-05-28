import cv2
import uuid
import imageio
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from letsdoit.utils.misc import inverseRigid
from typing import List


class ObjectInstance():
    def __init__(self, image, image_name, depth, bbox ,mask, label, confidence, intrinsic, extrinsic, orientation):
        self.bbox = bbox
        self.mask = mask
        self.label = label
        self.confidence = confidence
        self.id = str(uuid.uuid4())
        self.orientation = orientation

        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.depth = depth
        self.image = image
        self.image_name = image_name
        self._center_2d = None
        self._center_3d = None
        self._mask_3d = None

    @property
    def mask_3d(self):
        if self._mask_3d is None:
            mask_points = np.argwhere(self.mask)
            mask_points = mask_points[:, [1, 0]].T
            self._mask_3d = self._get_3d(mask_points)
        return self._mask_3d
    
    @property
    def center_2d(self):
        if self._center_2d is None:
            self._center_2d = self._get_mask_center()
        return self._center_2d

    @property
    def center_3d(self):
        if self._center_3d is None:
            self._center_3d = np.mean(self.mask_3d, axis=1)
        return self._center_3d
    
    def _get_mask_center(self):
        # Calculate moments of the binary image
        M = cv2.moments(self.mask.astype(np.uint16))

        # Calculate x, y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # set coordinates as zero if the mask is empty

        return (cX, cY)
    
    def _get_3d(self, points):
        '''
        Get the 3D cooridnates of the specified points (if extrinsic specified - in the world frame, if not - in camera frame)

        Args:
            points (numpy.ndarray): a 2 x N matrix of pixel coordiates for the N points to get 3D coordinates for

         Returns:
            mask_3d (numpy.ndarray): a 3 x N matrix of metric coordinats for the provided points.  
        '''
        # Get inverse intrinsics
        K_inv = np.linalg.inv(self.intrinsic)

        # Get depth values corresponding to image points
        ones = np.ones((1, points.shape[1]))
        pix = np.vstack([points, ones]).astype(np.uint16)
        depth_values = self.depth[pix[1, :], pix[0, :]]
        # Calculate the 3D coordinates of the points
        mask_3d = depth_values * np.matmul(K_inv, pix)

        # Convert to homogeneous coordinates for transformation to world coords
        mask_3d = np.vstack([mask_3d, np.ones((1, mask_3d.shape[1]))])
        extrinsic_inv = inverseRigid(self.extrinsic)
        # mask_3d = extrinsic_inv @ mask_3d
        mask_3d = self.extrinsic @ mask_3d
        # Convert back to 3D
        mask_3d = mask_3d[:-1, :]
        return mask_3d

    def plot_2d(self):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        ax = plt.gca()
        # Image
        ax.imshow(self.image)
        #Mask
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = self.mask.shape[-2:]
        mask_img = self.mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_img)
        #bbox
        x0, y0, w, h = self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        txt = ax.text(x0, y0, self.label+f' ({self.confidence:.2f})')
        col = np.random.rand(3)
        ax.plot(self.center_2d[0], self.center_2d[1], 'o',  color=col, markersize=10)
    
    def plot_3d(self, fig=None, subsample=True):
        if fig is None:
            fig = go.Figure()

        num_points = self.mask_3d.shape[1]

        # Subsample for viz, if too many points:
        if subsample:
            subsample_points = num_points // 100
            indices = np.linspace(0, self.mask_3d.shape[1] - 1, subsample_points, dtype=int)
            mask_3d = self.mask_3d[:, indices]
        else:
            mask_3d = self.mask_3d
        mask_color = np.random.rand(3,) * 255
        fig.add_trace(go.Scatter3d(
            x=mask_3d[0, :],
            y=mask_3d[1, :],
            z=mask_3d[2, :],
            mode='markers',
            marker=dict(size=2, color=f'rgb({mask_color[0]}, {mask_color[1]}, {mask_color[2]})', opacity=0.8),
            name=f'{self.image_name} | {self.label} ({self.confidence:.2f})'
        ))

        # center_3d = np.expand_dims(self.center_3d, -1)
        # center_color = np.random.rand(3,)
        # fig.add_trace(go.Scatter3d(
        #     x=center_3d[0, :],
        #     y=center_3d[1, :],
        #     z=center_3d[2, :],
        #     mode='markers',
        #     marker=dict(size=5, color=center_color, opacity=1.0),
        #     name='Mask Center'
        # ))

        fig.update_layout(title='3D Masks',
                        scene=dict(xaxis_title='X Axis',
                                    yaxis_title='Y Axis',
                                    zaxis_title='Z Axis',
                                    aspectmode='cube'),
                        legend={
                            'itemsizing': 'constant'
                        })

        if fig is None:
            fig.show()

def plot_instances_3d(instances: List[ObjectInstance], subsample=True):
    fig = go.Figure()
    for instance in instances:
        instance.plot_3d(fig=fig, subsample=subsample)
    fig.show()

def initialize_object_instances(images, image_names, depths, bboxes, masks, labels, confidences, intrinsics, extrinsics, orientations):
    """
    Initialize a list of ObjectInstance objects
    """

    object_instances = []
    
    for image, image_name, depth, bbox, mask, label, confidence, intrinsic, extrinsic, orientation in zip(images,
                                                                                                   image_names,
                                                                                                   depths,
                                                                                                   bboxes, 
                                                                                                   masks, 
                                                                                                   labels, 
                                                                                                   confidences, 
                                                                                                   intrinsics, 
                                                                                                   extrinsics, 
                                                                                                   orientations):
        
        object_instances.append(ObjectInstance(image,
                                               image_name,
                                               depth,
                                               bbox,
                                               mask,
                                               label,
                                               confidence,
                                               intrinsic,
                                               extrinsic,
                                               orientation))

    return object_instances
        