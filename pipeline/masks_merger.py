from typing import List

import numpy as np
import open3d
import scipy
from torch.nn import CosineSimilarity
from tqdm import tqdm

from pipeline.object_instance import ObjectInstance
from pipeline.object_3d import Object3D
from letsdoit.utils.misc import sample_points


class MasksMerger:

    def __init__(self, 
                 dist_thresh=0.1, 
                 geo_similarity_thresh=0.7, 
                 feat_similarity_thresh=0.7,
                 n_points=1000):
        """
        Parameters
        ----------
        dist_thresh : float
            threshold used in the geometric similarity. Increase 
        n_points : int
            The point cloud is downsampled ton n_points before the geometric similarity is computed
        """

        self.dist_thresh = dist_thresh
        self.geo_similarity_thresh = geo_similarity_thresh
        self.feat_similarity_thresh = feat_similarity_thresh
        self.n_points = n_points
        
    
    def __call__(self, object_instances: List[ObjectInstance], pcd: open3d.pybind.geometry.PointCloud,
                 object_proximity_thresh: float=0.1) -> List[Object3D]:
        """
        Given a list of ObjectInstance, match them and return a list of Object3D.
        pcd is a point cloud
        """

        objects_3d = []

        for object_instance in tqdm(object_instances, desc='Merging the object instances'):

            # handle edge case
            if object_instance.mask_3d.shape[1] == 0:
                continue

            if len(objects_3d) == 0:
                objects_3d.append(Object3D(object_instance, pcd, proximity_thresh=object_proximity_thresh))

            else:
                # we need to assign the object_instance to a object_3d
                idx = self.find_match(objects_3d, object_instance)

                if idx == -1:
                    # no object_3d has been found as a match
                    objects_3d.append(Object3D(object_instance, pcd, proximity_thresh=object_proximity_thresh))

                else:
                    # the match has been found, we integrate the object_instance in the object_3d
                    objects_3d[idx].add_object_instance(object_instance)

        return objects_3d

    
    def find_match(self, objects_3d, object_instance):
        """
        Assign an object instance to the best fitting object_3d from the objects_3d list
        and return its index. If no fitting object_3d has been found, return -1, otherwise 
        return the index of the object_3d.
        """

        similarities = []


        for object_3d in objects_3d:
            similarity = self.similarity(object_3d, object_instance)
            similarities.append(similarity)

        if sum(similarities) == 0:
            return -1

        return np.argmax(similarities)


    def similarity(self, object_3d, object_instance):
        """
        Return the feature_similarity between the image_features of the object_3d and the 
        object_instance if the geometric_similarity > self.geo_similarity_thresh, otherwise
        return 0.
        """

        object_3d_pts = object_3d.points
        object_instance_pts = object_instance.mask_3d

        geo_similarity = self.geometric_similarity(object_3d_pts, object_instance_pts)

        if geo_similarity < self.geo_similarity_thresh:
            return 0

        object_3d_features = object_3d.image_features
        object_instance_features = object_instance.image_features
        feat_similarity = self.feature_similarity(object_3d_features, object_instance_features)

        if feat_similarity < self.feat_similarity_thresh:
            return 0

        return feat_similarity    
    
    
    def geometric_similarity(self, points1, points2):
        """
        Compute the similarity between 2 point clouds.
        points1 and points2 are numpy.array of shape (3, n_points).
        Compute the similarity as the percentage of points which has a nearest neighbour in the 
        other point cloud at a distance < threshold
        """

        if points1.shape[1] == 0 or points2.shape[1] == 0:
            return 0.

        points1 = sample_points(points1, n=self.n_points)
        points2 = sample_points(points2, n=self.n_points)

        distances = scipy.spatial.distance_matrix(points1.T, points2.T)

        # count the number of nearest neighbours closer than threshold for both poin clouds
        score1 = sum(distances.min(axis=1) < self.dist_thresh)/self.n_points
        score2 = sum(distances.min(axis=0) < self.dist_thresh)/self.n_points
        

        # return the minimum
        return min(score1, score2)


    @staticmethod
    def feature_similarity(features1, features2):
        cos = CosineSimilarity()
        similarity = cos(features1.unsqueeze(0), features2.unsqueeze(0))
        return similarity.item()




        

