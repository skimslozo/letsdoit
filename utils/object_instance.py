import cv2
import uuid
import imageio
import numpy as np
from pathlib import Path

from misc import inverseRigid

class ObjectInstance():
    def __init__(self, bbox, mask, label, confidence, img_path):
        self.bbox = bbox
        self.mask = mask
        self.label = label
        self.confidence = confidence
        self.id = str(uuid.uuid4())
        self.mask_center_2d = self._get_mask_center()
        self.img_path = Path(img_path)
        self.intrinsic = self.get_intrinsics()
        self.depth = self.get_depth()
        self.extrinsic = None #TODO: adpat this, if we need the extrinsic here at some moment
        self.center_3d = self.get_mask_center_3d(mask)

    
    def get_intrinsics(self):
        intrinsic_path = self.img_path.parent.parent / 'intrinsic_rotated' / 'intrinsic_color.txt'
        return np.loadtxt(intrinsic_path)
    
    def get_depth(self):
        depth_path = self.img_path.parent.parent / 'depth_rotated' / self.img_path.name
        return imageio.v2.imread(depth_path) / 1000 
    
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
    
    def get_3d(self, points):
        '''
        Get the 3D cooridnates of the specified points (if extrinsic specified - in the world frame, if not - in camera frame)

        Args:
            points (numpy.ndarray): a 2 x N matrix of pixel coordiates for the N points to get 3D coordinates for

         Returns:
            mask_3d (numpy.ndarray): a 3 x N matrix of metric coordinats for the provided points.  
                Note: if extrinsic matrix is not specified, we just project to 3D in camera coordinate system. If specified,
                we use the rotation and translation matrix to further project the points to world coordinate system (!! NOT IMPLEMENTED !!)
        '''
        # Get inverse intrinsics
        K_inv = np.linalg.inv(self.intrinsic)


        # Get depth values corresponding to image points
        ones = np.ones((1, points.shape[1]))
        pix = np.vstack([points, ones]).astype(np.uint16)
        depth_values = self.depth[pix[1, :], pix[0, :]]
        # Calculate the 3D coordinates of the points
        mask_3d = depth_values * np.matmul(K_inv, pix)

        if self.extrinsic is None:
            return mask_3d
        else:
            extrinsic_inv = inverseRigid(self.extrinsic)
            mask_3d = extrinsic_inv @ mask_3d
            return mask_3d
        
    def get_mask_center_3d(self):
        '''
        Get the 3D coordinates of the center point of a mask
        '''
        center_in = np.asarray(self.mask_center_2d).reshape(2, 1)
        return self.get_3d(center_in)