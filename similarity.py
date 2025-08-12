"""
@brief: Similarity Metrics and Distance Functions for Vehicle Re-identification
@project: AI Based Smart Parking Management System - Similarity Computing Component
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: Advanced similarity computation functions for various feature types including
             histogram correlation, cosine distance, L2 distance, and composite similarity
             measures for multi-model feature fusion.
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


def histogram_distance(hist1, hist2):
    """
    Compute histogram distance between two histograms.

    Parameters:
    hist1 (np.ndarray): First histogram
    hist2 (np.ndarray): Second histogram

    Returns:
    float: Histogram distance between the histograms
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def l2_distance(feature1, feature2):
    """
    Compute L2 distance between two features.

    Parameters:
    feature1 (np.ndarray): First feature
    feature2 (np.ndarray): Second feature

    Returns:
    float: L2 distance between the features
    """
    return np.linalg.norm(feature1 - feature2)

def cosine_distance(feature1, feature2):
    """
    Compute cosine distance between two features.

    Parameters:
    feature1 (np.ndarray): First feature
    feature2 (np.ndarray): Second feature

    Returns:
    float: Cosine distance between the features
    """
    # Reshape features to 2D array for sklearn function
    feature1 = feature1.reshape(1, -1)
    feature2 = feature2.reshape(1, -1)
    return 1 - cosine_similarity(feature1, feature2)[0][0]

def composite_distance(sift_matches, reference_keypoints, osnet_distance, sift_weight=0.5, osnet_weight=0.5):
    """
    Calculate a weighted average of SIFT and OSNet distances.

    Parameters:
    sift_matches (list): List of SIFT matches
    reference_keypoints (list): List of keypoints in the reference image
    osnet_distance (float): Distance from OSNet model
    sift_weight (float): Weight for SIFT distance
    osnet_weight (float): Weight for OSNet distance

    Returns:
    float: Composite distance
    """
    # Normalize the SIFT matches to a comparable scale as cosine distance
    normalized_sift_score = 1 - (len(sift_matches) / max(len(reference_keypoints), 1))
    
    # Weighted sum of distances
    total_distance = sift_weight * normalized_sift_score + osnet_weight * osnet_distance
    return total_distance

def composite_resnet_mobilenet_distance(resnet_distance, mobilenet_distance, resnet_weight=0.6, mobilenet_weight=0.4):
    """
    Calculate a weighted average of ResNet and MobileNetV2 distances.

    Parameters:
    resnet_distance (float): Distance from ResNet model
    mobilenet_distance (float): Distance from MobileNetV2 model
    resnet_weight (float): Weight for ResNet distance (default 0.6 - ResNet typically more accurate)
    mobilenet_weight (float): Weight for MobileNetV2 distance (default 0.4)

    Returns:
    float: Composite distance
    """
    # Weighted sum of distances
    total_distance = resnet_weight * resnet_distance + mobilenet_weight * mobilenet_distance
    return total_distance
