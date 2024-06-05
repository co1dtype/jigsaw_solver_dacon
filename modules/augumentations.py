from albumentations.core.transforms_interface import DualTransform
import numpy as np


class CenterCutout(DualTransform):
    def __init__(self, min_cutout_fraction=0.3, always_apply=False, p=0.5):
        super(CenterCutout, self).__init__(always_apply, p)
        self.min_cutout_fraction = min_cutout_fraction

    def apply(self, image, **params):
        height, width = image.shape[:2]
        # Divide the image into 16 equal parts
        h_step, w_step = height // 4, width // 4

        for h in range(4):
            for w in range(4):
                # Calculate the center of each block
                center_x, center_y = w_step * w + w_step // 2, h_step * h + h_step // 2
                # Calculate random cutout size
                min_size = int(min(h_step, w_step) * self.min_cutout_fraction)
                max_size = min(h_step, w_step) // 2
                cutout_size = np.random.randint(min_size, max_size)
                # Apply the cutout
                x1 = max(center_x - cutout_size // 2, 0)
                y1 = max(center_y - cutout_size // 2, 0)
                x2 = min(center_x + cutout_size // 2, width)
                y2 = min(center_y + cutout_size // 2, height)
                image[y1:y2, x1:x2] = 0
        return image
    
    def get_transform_init_args_names(self):
        return ("min_cutout_fraction", "always_apply", "p")
    
class EdgeCutout(DualTransform):
    def __init__(self, cutout_size=20, always_apply=False, p=0.5):
        super(EdgeCutout, self).__init__(always_apply, p)
        self.cutout_size = cutout_size

    def apply(self, image, **params):
        height, width = image.shape[:2]
        # Calculate the step size for each block
        h_step, w_step = height // 4, width // 4

        # Apply cutout at each intersection
        for h in range(1, 4):
            for w in range(1, 4):
                center_x, center_y = w * w_step, h * h_step
                x1 = max(center_x - self.cutout_size // 2, 0)
                y1 = max(center_y - self.cutout_size // 2, 0)
                x2 = min(center_x + self.cutout_size // 2, width)
                y2 = min(center_y + self.cutout_size // 2, height)
                image[y1:y2, x1:x2] = 0
        return image
    
    def get_transform_init_args_names(self):
        return ("cutout_size", "always_apply", "p")
