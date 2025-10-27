
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalImageAugmentation:
    """
    Comprehensive augmentation pipeline for medical images
    """
    
    def __init__(self, augmentation_level: str = 'moderate'):
        self.augmentation_level = augmentation_level
        self.transform = self._build_pipeline()
        
    def _build_pipeline(self):
        """Build augmentation pipeline based on specified level"""
        
        if self.augmentation_level == 'minimal':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
            
        elif self.augmentation_level == 'moderate':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
            
        elif self.augmentation_level == 'full':  # Our approach
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.3),  # Zoom equivalent
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    alpha_affine=50, 
                    p=0.2
                ),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
    
    def __call__(self, image):
        return self.transform(image=image)['image']
