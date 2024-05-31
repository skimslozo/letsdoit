from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from utils.object_instance import ObjectInstance
from utils.misc import sample_points



class Object3D:


    def __init__(self, object_instance: ObjectInstance, max_points=10000):

        self.max_points = max_points

        self.points = sample_points(object_instance.mask_3d, n=self.max_points)
        self.image_features = object_instance.image_features
        self.object_instances = [object_instance]
        self.label = object_instance.label

    def add_object_instance(self, object_instance: ObjectInstance):
        points = np.hstack([self.points, object_instance.mask_3d])

        self.points = sample_points(points, n=self.max_points)

        n_objs = len(self.object_instances)
        # take the average of the previous image_features with the one coming from the new object_instance
        self.image_features = (n_objs * self.image_features + object_instance.image_features) / (n_objs + 1)

        self.object_instances.append(object_instance)


    def plot(self, fig=None, subsample=False, show=True):
        if fig is None:
            fig = go.Figure()

        num_points = self.points.shape[1]

        # Subsample for viz, if too many points:
        if subsample:
            subsample_points = num_points // 100
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


def plot_objects_3d(objects_3d: List[Object3D], subsample=False):
    fig = go.Figure()
    for object_3d in objects_3d:
        object_3d.plot(fig=fig, subsample=subsample, show=False)
    fig.show()