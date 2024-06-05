




class MasksMatcher:

    def __init__(self):
        """placeholder"""
        pass

    def match_masks(self, object_instances):

        centers = []

        for object_instance in object_instances:
            centers.append(object_instance.center_3d)

        
        centers = np.array(centers)

        # TODO: decide what to do here: clustering or distance?

        # TODO: for example instead of using only the centers I could use all the points and cluster


        
        
        
        
