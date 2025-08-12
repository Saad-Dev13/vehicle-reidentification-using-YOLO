"""
@brief: Advanced Feature Extraction Module for Vehicle Re-identification
@project: AI Based Smart Parking Management System - Feature Extraction Component
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: Comprehensive feature extraction functions supporting traditional computer vision
             methods (HOG, SIFT, color histograms) and deep learning approaches (ResNet, MobileNetV2)
             with confidence scoring and quality validation.
"""

import cv2
import numpy as np
from skimage.feature import hog
import torch
from torchvision import transforms

def compute_hog_features(image):
    """
    Compute HOG features for the image

    Parameters:
    image (np.ndarray): Input image

    Returns:
    np.ndarray: HOG features
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    # normalize
    features /= np.linalg.norm(features)
    return features

def compute_color_histogram(image, bins=32):
    """
    Compute a color histogram for the image

    Parameters:
    image (np.ndarray): Input image
    bins (int): Number of bins for the histogram

    Returns:
    np.ndarray: Color histogram
    """
    histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(image.shape[2])]
    histogram = np.concatenate(histogram)
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def extract_dnn_features(model, image, confidence_threshold=0.8):
    """
    Extract deep features using a pre-trained model with quality validation

    Parameters:
    model (torch.nn.Module): Pre-trained model
    image (np.ndarray): Input image
    confidence_threshold (float): Minimum confidence for feature quality

    Returns:
    tuple: (features, confidence_score) - Extracted features and confidence score
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        features = model(image_tensor)
        
        # Calculate confidence based on feature magnitude and distribution
        feature_magnitude = torch.norm(features).item()
        feature_std = torch.std(features).item()
        feature_mean = torch.mean(features).item()
        
        # Confidence score based on feature characteristics
        # Higher magnitude and variance typically indicate better features
        confidence_score = min(1.0, (feature_magnitude * feature_std) / 100.0)
        
        # Alternative confidence calculation using entropy-like measure
        feature_abs = torch.abs(features)
        feature_normalized = feature_abs / (torch.sum(feature_abs) + 1e-8)
        entropy = -torch.sum(feature_normalized * torch.log(feature_normalized + 1e-8))
        entropy_confidence = min(1.0, entropy.item() / 10.0)
        
        # Combine both confidence measures
        final_confidence = (confidence_score + entropy_confidence) / 2.0
        
    return features.flatten(), final_confidence

def extract_dnn_features_simple(model, image):
    """
    Extract deep features using a pre-trained model (backward compatibility)

    Parameters:
    model (torch.nn.Module): Pre-trained model
    image (np.ndarray): Input image

    Returns:
    np.ndarray: Extracted features
    """
    features, _ = extract_dnn_features(model, image)
    return features

def extract_torchreid_features(model, image):
    """
    Extract features using Torchreid feature extractor

    Parameters:
    model (FeatureExtractor): Pre-trained TorchReid model
    image (np.ndarray): Input image (BGR format from OpenCV)

    Returns:
    np.ndarray: Extracted features
    """
    # Convert BGR to RGB (TorchReid expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # TorchReid's FeatureExtractor handles the preprocessing internally
    features = model(image_rgb)
    
    # Ensure features are flattened and converted to numpy
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    
    return features.flatten()