import numpy as np
import cv2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms

import random 
from PIL import ImageDraw


##### ##### ##### #####

# Taken from GitHub repository of AdaFace
# https://github.com/mk-minchul/AdaFace

class Augmenter():

    def __init__(self, crop_augmentation_prob=0.1, photometric_augmentation_prob=0.1, low_res_augmentation_prob=0.1):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.5, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    # def augment(self, sample):
    def __call__(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            sample, crop_ratio = self.crop_augment(sample)

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        return sample

    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        if hasattr(F, '_get_image_size'):
            orig_W, orig_H = F._get_image_size(sample)
        else:
            # torchvision 0.11.0 and above
            orig_W, orig_H = F.get_image_size(sample)
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        crop_ratio = min(h, w) / max(orig_H, orig_W)
        return sample, crop_ratio

    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        side_ratio = np.random.uniform(0.5, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

        return aug_img, side_ratio

    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample


##### ##### ##### #####

# Extending Augmenter with additional augmentations, i.e., sunglasses and mask.  

class Augmenter_EXT():

    def __init__(self, crop_augmentation_prob=0.1, photometric_augmentation_prob=0.1, low_res_augmentation_prob=0.1,
                 sunglasses_augmentation_prob=0.1, masking_augmentation_prob=0.1):
        
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob
        self.sunglasses_augmentation_prob = sunglasses_augmentation_prob
        self.masking_augmentation_prob = masking_augmentation_prob

        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.5, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    # def augment(self, sample):
    def __call__(self, sample):

        # sunglasses augmentation 
        sunglasses_flag = False
        if np.random.random() < self.sunglasses_augmentation_prob:
            sunglasses_flag = True
            sample = self.sunglass_augmentation(sample)
           
        # masking augmentation
        if sunglasses_flag is False:
            if np.random.random() < self.masking_augmentation_prob:
                sample = self.masking_augmentation(sample)
            
        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            sample, crop_ratio = self.crop_augment(sample)

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)
            
        return sample

    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        if hasattr(F, '_get_image_size'):
            orig_W, orig_H = F._get_image_size(sample)
        else:
            # torchvision 0.11.0 and above
            orig_W, orig_H = F.get_image_size(sample)
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        crop_ratio = min(h, w) / max(orig_H, orig_W)
        return sample, crop_ratio

    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        side_ratio = np.random.uniform(0.5, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

        return aug_img, side_ratio

    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample
    
    def sunglass_augmentation(self, sample):

        # if sample.mode == 'RGBA':
        #    sample = sample.convert('RGB')

        # Predefined landmarks representing centers of the eyes
        landmarks = np.array([[38.29459953, 51.69630051],
                              [73.53179932, 51.50139999]])

        left_eye_center, right_eye_center = landmarks

        # Calculate width and height for the sunglasses lenses based on eye distance
        eye_distance = np.linalg.norm(landmarks[0] - landmarks[1])
        lens_width = int(eye_distance * random.uniform(0.6, 0.8))
        lens_height = int(eye_distance * random.uniform(0.4, 0.6))

        # Create a drawing context
        # image_np = np.array(sample)
        # sample_copy = Image.fromarray(image_np.copy())
        # sample_copy = sample.copy()
        new = np.array(sample)
        sample = Image.fromarray(new.astype(np.uint8))
        draw = ImageDraw.Draw(sample)

        # Draw the sunglasses with full opacity in RGB mode
        fill_color = (0, 0, 0)  # Solid black color without alpha component
        left_rect_box = [left_eye_center[0] - lens_width // 2, left_eye_center[1] - lens_height // 2,
                         left_eye_center[0] + lens_width // 2, left_eye_center[1] + lens_height // 2]
        right_rect_box = [right_eye_center[0] - lens_width // 2, right_eye_center[1] - lens_height // 2,
                          right_eye_center[0] + lens_width // 2, right_eye_center[1] + lens_height // 2]

        # Draw opaque rectangles for sunglasses lenses
        draw.rectangle(left_rect_box, fill=fill_color)
        draw.rectangle(right_rect_box, fill=fill_color)

        return sample
    
    def masking_augmentation(self, sample):
        
        # Convert PIL Image to numpy array
        image_np = np.array(sample)
        
        # Example landmarks: [right_eye, left_eye, nose, left_mouth_corner, right_mouth_corner]
        landmarks = np.array([[38.29459953, 51.69630051],
                              [73.53179932, 51.50139999],
                              [56.02519989, 71.73660278],
                              [41.54930115, 92.3655014],
                              [70.72990036, 92.20410156]])
        
        # Use the nose landmark to determine the starting line of the mask
        nose_lm = landmarks[2]
        line_y = int(nose_lm[1])
        
        # Calculate width based on eye landmarks to estimate face width
        eyes_lm = landmarks[:2]
        face_width = int(np.linalg.norm(eyes_lm[0] - eyes_lm[1]))
        margin = face_width // 2
        
        # Apply mask only to the lower part of the face, below the nose
        mid_x = image_np.shape[1] // 2  # Midpoint of image width
        start_x = max(0, mid_x - margin)  # Avoid negative indices
        end_x = min(image_np.shape[1], mid_x + margin)  # Avoid indices beyond image width
        
        image_np[line_y:, :] = 0  # Apply mask
        
        # Convert numpy array back to PIL Image and return
        return Image.fromarray(image_np)
