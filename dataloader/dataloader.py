import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import imageio
import numpy as np
from tqdm import tqdm

from dataloader.data_parser import DataParser, decide_pose#, rotate_pose
from utils.misc import inverseRigid

TMP_STORAGE_PATH = Path('/teamspace/studios/this_studio/letsdoit/.tmp_runtime_files')

class DataLoader:

    def __init__(self, data_root_path, split = "train") -> None:
        
        self.parser = DataParser(data_root_path, split=split)

        self.visit_ids = os.listdir(self.parser.data_root_path)
        self.visit_paths = {visit_id: os.path.join(self.parser.data_root_path, visit_id) 
                             for visit_id in self.visit_ids}
        
        
    def get_video_ids(self, visit_id: str) -> List[List[str]]:
        """
        Return the video ids for a given scene 
        """
        # os walk returns (dirpath, dirnames, filenames) so index 1 is what we need
        DIRNAMES = 1 
        scene_path = self.visit_paths[visit_id]
        video_ids =  next(os.walk(scene_path))[DIRNAMES]
        return video_ids

    
    def get_video_paths(self, visit_id: str) -> List[List[str]]:
        """
        Return the video path folders for a given scene 
        """
        # os walk returns (dirpath, dirnames, filenames) so index 1 is what we need
        scene_path = self.visit_paths[visit_id]
        video_ids = self.get_video_ids(visit_id)
        video_paths = [os.path.join(scene_path, video_id) for video_id in video_ids]
        return video_paths
    

    def _get_framepaths_intrisics_poses(self, visit_id: str, video_id: str, format='rgb', asset_type="lowres_wide", sample_freq=1):
        """
        Return the frame paths, the intrinsics and the poses for a given visit id, video id and format (either depth or rgb).
        """

        frame_ids, frame_paths, intrinsics = self.parser.get_frame_id_and_intrinsic(visit_id, 
                                                                                    video_id,
                                                                                    format=format,
                                                                                    asset_type=asset_type)
        
        # convert from dictionary to list and sample according to sample_freq
        frame_paths = [frame_paths[frame_id] for i, frame_id in enumerate(frame_ids) if i%sample_freq == 0]
        intrinsics = [intrinsics[frame_id] for i, frame_id in enumerate(frame_ids) if i%sample_freq == 0]
        frame_ids = [frame_id for i, frame_id in enumerate(frame_ids) if i%sample_freq == 0]
        
        camera_trajectory = self.parser.get_camera_trajectory(visit_id, video_id)

        poses = []

        for i, frame_id in enumerate(frame_ids):
            pose = self.parser.get_nearest_pose(frame_id,
                                                camera_trajectory, use_interpolation=True)
            if pose is None:
                frame_paths.pop(i)
                intrinsics.pop(i)
                continue
            poses.append(pose)

        return frame_paths, intrinsics, poses
    

    def get_image_paths(self, visit_id: str, video_id: str, asset_type="lowres_wide", sample_freq=1):
        """
        return the path to all the images in the scene. Plus intrinsics and poses
        """

        image_paths, intrinsics, poses = self._get_framepaths_intrisics_poses(visit_id, video_id, format='rgb', asset_type=asset_type, sample_freq=sample_freq)
        return image_paths, intrinsics, poses
    

    def get_depth_paths(self, visit_id: str, video_id: str, asset_type="lowres_wide", sample_freq=1):
        """
        return the path to all the depths in the scene. Plus intrinsics and poses
        """

        depth_paths, intrinsics, poses = self._get_framepaths_intrisics_poses(visit_id, video_id, format='depth', asset_type=asset_type, sample_freq=sample_freq)    
        return depth_paths, intrinsics, poses
    

    def _get_frames_framepaths_intrinsics_poses_orientations(self, visit_id, video_id, format='rgb', asset_type="lowres_wide", sample_freq=1):
        """
        Return all the frames for a given scene  and format as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """

        frame_paths, intrinsics, poses = self._get_framepaths_intrisics_poses(visit_id, video_id, format=format, asset_type=asset_type, sample_freq=sample_freq)

        # frames = []
        orientations = []
        # frames_rotated = []

        description = f'Loading {format} frames from visit {visit_id} and video {video_id}'
        for i, (frame_path, pose) in tqdm(enumerate(zip(frame_paths, poses)), total=len(frame_paths), desc=description):
            # if format == 'rgb':
            #     # frame = self.parser.read_rgb_frame(frame_path)
            # else:
            #     # frame = self.parser.read_depth_frame(frame_path)
            orientation = decide_pose(pose)
            # frames.append(frame)
            # frames_rotated.append(rotate_pose(frame, orientation))
            orientations.append(orientation)

        # return frames, frames_rotated, frame_paths, intrinsics, poses, orientations
        return frame_paths, intrinsics, poses, orientations


    """
    def _rotate_intrinsic(self, intrinsic: Tuple[float], orientation: int) -> np.ndarray:
        '''rotate the intrinsic matrix with the following chain of transformations:
        tc1c -> translation to shift the zero of the uv coordinate system from top-left angle
        to the center of the image. tc1c = -c where c = (w/2, h/2)

        R12 -> rotate from original matrix to r
        
        '''
        w, h, au, av, u0, v0 = intrinsic
        intrinsic_matrix = np.array([[au, 0, u0],
                                     [0, av, v0],
                                     [0, 0, 1]])
        c = np.array([w/2, h/2])

        # apply swap when the image is rotated such that the previous x and y axis are swapped
        exchange_matrix = np.array([[0, 1],
                                    [1, 0]])

        identity_matrix = np.eye(2)

        match orientation:
            case 0:  # no rotation
                R = identity_matrix
                t = np.zeros(2)
            case 1:  # rotate 90 deg clockwise
                R = np.array([[0, -1],
                              [1, 0]])
                
                t = (-R + exchange_matrix) @ c

            case 2:  # rotate 180 deg
                R = np.array([[-1, 0],
                              [0, -1]])
                
                t = (-R + identity_matrix) @ c

            case 3:  # rotate 90 deg counter-clockwise
                R = np.array([[0, 1],
                              [-1, 0]])
                
                t = (-R + exchange_matrix) @ c

            case _:
                raise ValueError(f'Unexpected orientation int provided: {orientation}')


        transform = np.array([[R[0, 0], R[0, 1], t[0]],
                             [R[1, 0], R[1, 1], t[1]],
                             [0, 0, 1]])

        intrinsic_rotated = transform @ intrinsic_matrix
        return intrinsic_rotated
    """


    # def _rotate_intrinsic(self, intrinsic: Tuple[float], orientation: int) -> np.ndarray:
    #     '''Based on original orientation, adjust intrinsics matrix for upright-oriented image'''
    #     w, h, au, av, u0, v0 = intrinsic
    #     match orientation:
    #         case 0: # upright -> normal
    #             intrinsic_rotated = np.array([[au, 0, u0],
    #                                           [0, av, v0],
    #                                           [0, 0, 1]])

    #         case 1: # left -> apply 90deg clockwise rotation
    #             intrinsic_rotated = np.array([[0, -av, h-v0],
    #                                           [au, 0, u0],
    #                                           [0, 0, 1]])

    #         case 2: # upisde-down -> apply 180 deg rotation
    #             intrinsic_rotated = np.array([[-au, 0, w-u0],
    #                                           [0, -av, h-v0],
    #                                           [0, 0, 1]])

    #         case 3: # right -> apply 90 deg counter-clockwise rotation
    #             intrinsic_rotated = np.array([[0, av, v0],
    #                                           [-au, 0, w-u0],
    #                                           [0, 0, 1]])
    #         case _:
    #             raise ValueError(f'Unexpected orientation int provided: {orientation}')
    
    #     return intrinsic_rotated

    def get_images_video_id(self, visit_id, video_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the images for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """

        # images, images_rotated, image_paths, intrinsics, poses, orientations = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='rgb', 
        #                                                                                         asset_type=asset_type,
        #                                                                                         sample_freq=sample_freq)

        image_paths, intrinsics, poses, orientations = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='rgb', 
                                                                                                asset_type=asset_type,
                                                                                                sample_freq=sample_freq)

        # return images, images_rotated, image_paths, intrinsics, poses, orientations
        return image_paths, intrinsics, poses, orientations

    
    def get_images(self, visit_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the images for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """

        video_ids = self.get_video_ids(visit_id)

        # images_l, images_rotated_l, image_paths_l, intrinsics_l, poses_l, orientations_l = [], [], [], [], [], []

        image_paths_l, intrinsics_l, poses_l, orientations_l = [], [], [], []

        transformations = [self.parser.get_refined_transform(visit_id, video_id) for video_id in video_ids]
        rel_transformations = self._relative_transformation(transformations) 

        for video_id, trans in zip(video_ids, rel_transformations):
            # images, images_rotated, image_paths, intrinsics, poses, orientations = self.get_images_video_id(visit_id, 
            #                                                                                                 video_id, 
            #                                                                                                 asset_type=asset_type, 
            #                                                                                                 sample_freq=sample_freq)

            image_paths, intrinsics, poses, orientations = self.get_images_video_id(visit_id, 
                                                                                    video_id, 
                                                                                    asset_type=asset_type, 
                                                                                    sample_freq=sample_freq)
            # images_l += images
            # images_rotated_l += images_rotated
            image_paths_l += image_paths
            intrinsics_l += intrinsics
            poses_l += [trans @ pose for pose in poses]
            orientations_l += orientations


        # return images_l, images_rotated_l, image_paths_l, intrinsics_l, poses_l, orientations_l
        return image_paths_l, intrinsics_l, poses_l, orientations_l
    

    def get_depths_video_id(self, visit_id, video_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the depths for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """

        # depths, depths_rotated, depth_paths, intrinsics, poses, orientations = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='depth', 
        #                                                                                       asset_type=asset_type,
        #                                                                                       sample_freq=sample_freq)

        depth_paths, intrinsics, poses, orientations = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='depth', 
                                                                                                                asset_type=asset_type,
                                                                                                                sample_freq=sample_freq)

        return depth_paths, intrinsics, poses, orientations

    
    def get_depths(self, visit_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the depths for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """

        video_ids = self.get_video_ids(visit_id)

        depths_l, depths_rotated_l, depth_paths_l, intrinsics_l, poses_l, orientations_l = [], [], [], [], [], []

        transformations = [self.parser.get_refined_transform(visit_id, video_id) for video_id in video_ids]
        rel_transformations = self._relative_transformation(transformations) 

        for video_id, trans in zip(video_ids, rel_transformations):
            # depths, depths_rotated, depth_paths, intrinsics, poses, orientations = self.get_depths_video_id(visit_id, 
            #                                                                                                 video_id, 
            #                                                                                                 asset_type=asset_type, 
                                                                                                            # sample_freq=sample_freq)

            depth_paths, intrinsics, poses, orientations = self.get_depths_video_id(visit_id,
                                                                                    video_id, 
                                                                                    asset_type=asset_type,
                                                                                    sample_freq=sample_freq)
            
            # depths_l += depths
            # depths_rotated_l += depths_rotated
            depth_paths_l += depth_paths
            intrinsics_l += intrinsics
            poses_l += [trans @ pose for pose in poses]
            orientations_l += orientations


        # return depths_l, depths_rotated_l, depth_paths_l, intrinsics_l, poses_l, orientations_l
        return depth_paths_l, intrinsics_l, poses_l, orientations_l
    


    def get_instructions(self, instructions_path: str) -> Dict:
        """
        Return the instruction breakdown into descriptions, spatial primitives and objects as a dictionary
        from the json insutrction file in instructions_path
        """
        with open(instructions_path, 'r') as file:
            instructions_dict = json.load(file)
        return instructions_dict


    def load_pcd(self, visit_id):
        """
        Load the point cloud and transform it to the reference system of the video
        """

        video_id = self.get_video_ids(visit_id)[0]

        pcd = self.parser.get_laser_scan(visit_id)
        trans = self.parser.get_refined_transform(visit_id, video_id)

        pcd_trans = pcd.transform(trans)

        return pcd_trans

    
    def get_image_features_path(self, visit_id):
        # return the path to the image features for the given visit_id

        path_visit_id = self.visit_paths[visit_id]
        path_image_features = os.path.join(path_visit_id, f"{visit_id}_image_features.pt")
        return path_image_features


    @staticmethod
    def _relative_transformation(transformations):
        """
        Given transformations=[T_{pcd,w1}, T_{pcd,w2}, ..., T_{pcd,wN}] returns
        [T_{c1,c1}, T_{c2,c1}, ..., T_{cN,c1}]
        """

        inverse_transformations = [inverseRigid(trans) for trans in transformations]

        relative_trans = [transformations[0] @ inverse_t for inverse_t 
                          in inverse_transformations]
        return relative_trans


    def _tmp_frame_save(self, frame: np.ndarray, original_path: str, appendix: str) -> str:
        opath = Path(original_path)
        save_path = TMP_STORAGE_PATH / opath.parent.name
        os.makedirs(save_path, exist_ok=True)
        save_img_path = str(save_path / (appendix + str(opath.name)))
        imageio.v2.imwrite(save_img_path, frame)
        return save_img_path

    def cleanup(self):
        print('Removing temporary runtime files')
        for item in os.listdir(TMP_STORAGE_PATH):
            item_path = os.path.join(TMP_STORAGE_PATH, item)
            try:
                # Check if the item is a file
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                # Check if the item is a directory
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f'Failed to delete {item_path}. Reason: {e}')