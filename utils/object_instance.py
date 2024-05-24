import cv2
import uuid
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from letsdoit.utils.misc import inverseRigid

class ObjectInstance():
    def __init__(self, bbox, mask, label, confidence, img_path):
        self.bbox = bbox
        self.mask = mask
        self.label = label
        self.confidence = confidence
        self.id = str(uuid.uuid4())
        self.img_path = Path(img_path)

        self._intrinsic = None
        self._extrinsic = None
        self._image = None
        self._depth = None
        self._center_2d = None
        self._center_3d = None
    
    @property
    def intrinsic(self):
        if self._intrinsic is None:
            self._intrinsic = self._get_intrinsics()
        return self._intrinsic

    @property
    def extrinsic(self):
        if self._extrinsic is None:
            self._extrinsic = self._get_extrinsics()
        return self.extrinsic
    
    @property
    def image(self):
        if self._image is None:
            self._image = cv2.imread(str(self.img_path))
        return self._image

    @property
    def depth(self):
        if self._depth is None:
            self._depth = self._get_depth()
        return self._depth

    @property
    def center_2d(self):
        if self._center_2d is None:
            self._center_2d = self._get_mask_center()
        return self._center_2d

    @property
    def center_3d(self):
        if self._center_3d is None:
            self._center_3d = self._get_mask_center_3d()
        return self._center_3d

    def _get_intrinsics(self):
        intrinsic_path = self.img_path.parent.parent / 'intrinsic_rotated' / 'intrinsic_color.txt'
        return np.loadtxt(intrinsic_path)
    
    def _get_extrinsics(self):
        #TODO: just made an assumption on how extrinsics are saved/called, to be fixed later
        extrinsic_path = self.img_path.parent.parent / 'extrinsic_rotated' / self.img_path.name
        return np.loadtxt(extrinsic_path)
    
    def _get_depth(self):
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
    
    def _get_3d(self, points):
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

        extrinsic_inv = inverseRigid(self.extrinsic)
        mask_3d = extrinsic_inv @ mask_3d
        return mask_3d
        
    def _get_mask_center_3d(self):
        '''
        Get the 3D coordinates of the center point of a mask
        '''
        center_in = np.asarray(self.center_2d).reshape(2, 1)
        return self._get_3d(center_in)

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
    
    def plot_3d(self):
        pass

