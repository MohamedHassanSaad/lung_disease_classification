

import numpy as np
import cv2
from skimage import exposure

class CTPreprocessor:
    def __init__(self, input_size=(224, 224), hu_window=[-1000, 400], apply_clahe=True):
        self.input_size = input_size
        self.hu_window = hu_window
        self.apply_clahe = apply_clahe
        
    def preprocess(self, image):
        # Step 1: Convert to Hounsfield Units (if not already)
        image_hu = self._convert_to_hu(image)
        
        # Step 2: Apply windowing
        image_windowed = self._apply_window(image_hu, self.hu_window)
        
        # Step 3: Resize to input size
        image_resized = cv2.resize(image_windowed, self.input_size)
        
        # Step 4: Normalize to [0, 1]
        image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
        
        # Step 5: Apply CLAHE for contrast enhancement (optional)
        if self.apply_clahe:
            image_normalized = exposure.equalize_adapthist(image_normalized)
            
        # Step 6: Convert to 3-channel (if needed)
        if len(image_normalized.shape) == 2:
            image_normalized = np.stack([image_normalized] * 3, axis=-1)
            
        return image_normalized
    
    def _convert_to_hu(self, image):
        # Assuming image is in Hounsfield Units or already converted
        # If not, implement the conversion using slope and intercept from DICOM metadata
        return image
    
    def _apply_window(self, image, window):
        # Apply windowing (clipping) to the image
        window_min, window_max = window
        image = np.clip(image, window_min, window_max)
        return image
