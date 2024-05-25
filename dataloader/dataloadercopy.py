import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

from dataloader.data_parser import DataParser, rotate_pose, decide_pose

class DataLoaderCopy:

    def __init__(self, data_root_path, split = "train") -> None:
        
        self.parser = DataParser(data_root_path, split=split)

        self.visit_ids = os.listdir(self.parser.data_root_path)
        self.visit_paths = {scene: os.path.join(self.parser.data_root_path, scene) 
                             for scene in self.visit_ids}
        
        
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

        for frame_id in frame_ids:
            pose = self.parser.get_nearest_pose(frame_id,
                                                camera_trajectory)
            
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

        frames = []
        orientations = []

        description = f'Loading {format} frames from visit {visit_id} and video {video_id}'
        for i, (frame_path, pose) in tqdm(enumerate(zip(frame_paths, poses)), total=len(frame_paths), desc=description):
            if format == 'rgb':
                frame = self.parser.read_rgb_frame(frame_path)
            else:
                frame = self.parser.read_depth_frame(frame_path)
            orientation = decide_pose(pose)
            frame = rotate_pose(frame, orientation)
            frames.append(frame)

            # Rotate intrinsics, replace the old one
            intrinsic = self._rotate_intrinsic(intrinsics[i], orientation)
            intrinsics[i] = intrinsic

            # Rotate extrinsics, replace the old one
            extrinsic = self._rotate_extrinsic(pose, orientation)
            poses[i] = extrinsic

            orientations.append(orientation)

        return frames, frame_paths, intrinsics, poses, orientations
    
    def _rotate_intrinsic(self, intrinsic: Tuple[float], orientation: int) -> np.ndarray:
        '''Based on original orientation, adjust intrinsics matrix for upright-oriented image'''
        _, _, au, av, u0, v0 = intrinsic
        match orientation:
            case 0: # upright -> normal
                intrinsic_rotated = np.array([[au, 0, u0],
                                                [0, av, v0],
                                                [0, 0, 1]])

            case 1: # left -> apply 90deg clockwise rotation
                intrinsic_rotated = np.array([[0, -au, u0],
                                                [av, 0, v0],
                                                [0, 0, 1]])

            case 2: # upisde-down -> apply 180 deg rotation
                intrinsic_rotated = np.array([[-au, 0, u0],
                                                [0, -av, v0],
                                                [0, 0, 1]])

            case 3: # right -> apply 90 deg counter-clockwise rotation
                intrinsic_rotated = np.array([[0, au, u0],
                                                [-av, 0, v0],
                                                [0, 0, 1]])
            case _:
                raise ValueError(f'Unexpected orientation int provided: {orientation}')
    
        return intrinsic_rotated

    def _rotate_extrinsic(self, extrinsic: np.ndarray, orientation: int) -> np.ndarray:
        '''Based on original orientation, adjust extrinsics matrix for upright-oriented image'''
        match orientation:
            case 0: # upright
                T = np.eye(4)
            case 1: # left -> apply 90deg clockwise rotation
                T = np.array([[0, 1, 0, 0],
                              [-1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            case 2: # upisde-down -> apply 180 deg rotation
                T = np.array([[-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, ]])
            case 3: # right -> apply 90 deg counter-clockwise rotation
                T = np.array([[0, -1, 0, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1]])
            case _:
                raise ValueError(f'Unexpected orientation int provided: {orientation}')
        return T @ extrinsic

    def get_images(self, visit_id, video_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the images for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """
        images, image_paths, intrinsics, poses, orientations = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='rgb', 
                                                                                              asset_type=asset_type,
                                                                                              sample_freq=sample_freq)

        return images, image_paths, intrinsics, poses, orientations
    

    def get_depths(self, visit_id, video_id, asset_type="lowres_wide", sample_freq=1):
        """
        Return all the depths for a given scene as a numpy.ndarray with the RGB color values.
        Pluse return the frame_paths, the intrinsics and the poses.
        """
        depths, depth_paths, intrinsics, poses = self._get_frames_framepaths_intrinsics_poses_orientations(visit_id, video_id, format='depth', 
                                                                                              asset_type=asset_type,
                                                                                              sample_freq=sample_freq)

        return depths, depth_paths, intrinsics, poses, orientations


    def get_instructions(self, instructions_path: str) -> Dict:
        """
        Return the instruction breakdown into descriptions, spatial primitives and objects as a dictionary
        from the json insutrction file in instructions_path
        """
        with open(instructions_path, 'r') as file:
            instructions_dict = json.load(file)
        return instructions_dict
