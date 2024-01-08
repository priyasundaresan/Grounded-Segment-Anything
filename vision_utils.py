import cv2
import time
import os
import numpy as np
import supervision as sv

import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model

from sklearn.neighbors import NearestNeighbors

def efficient_sam_box_prompt_segment(image, pts_sampled, model):
    bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
        bbox.cuda(),
        bbox_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou

def outpaint_masks(target_mask, other_masks):
    for mask in other_masks:
        ys,xs = np.where(mask > 0)
        target_mask[ys,xs] = 0
    return target_mask

def detect_blue(image):
    lower_blue = np.array([60,30,15]) 
    upper_blue = np.array([255,150,75]) 
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(image, lower_blue, upper_blue)
    return mask

def proj_pix2mask(px, mask):
    ys, xs = np.where(mask > 0)
    if not len(ys):
        return px
    mask_pixels = np.vstack((xs,ys)).T
    neigh = NearestNeighbors()
    neigh.fit(mask_pixels)
    dists, idxs = neigh.kneighbors(np.array(px).reshape(1,-1), 1, return_distance=True)
    projected_px = mask_pixels[idxs.squeeze()]
    return projected_px

def detect_densest(mask, kernel=(60,60)):
    # Find desired pixel in densest_masked_noodles
    kernel = np.ones(kernel, np.float32)/kernel[0]**2
    dst = cv2.filter2D(mask, -1, kernel)
    pred_y, pred_x = np.unravel_index(dst.argmax(), dst.shape)
    densest = proj_pix2mask((pred_x, pred_y), mask)
    densest = (int(densest[0]), int(densest[1]))
    return densest

def detect_sparsest(mask, densest):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    #cont = np.vstack(([c for c in contours if cv2.contourArea(c) > 100]))
    cont = np.vstack((contours))
    hull = cv2.convexHull(cont)
    hull = hull.reshape(len(hull), 2).astype(int)
    neigh = NearestNeighbors()
    neigh.fit(hull)
    dists, idxs = neigh.kneighbors(np.array(densest).reshape(1,-1), len(hull), return_distance=True)
    furthest = hull[idxs.squeeze()[-1]]
    furthest = proj_pix2mask(np.array(furthest), mask)
    furthest = (int(furthest[0]), int(furthest[1]))
    return furthest

def detect_centroid(mask):
    cX, cY = 0, 0
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
    centroid = proj_pix2mask(np.array([cX, cY]), mask)
    centroid = (int(centroid[0]), int(centroid[1]))
    return centroid

def cleanup_mask(mask, blur_kernel_size=(5, 5), threshold=127, erosion_size=3):
    """
    Applies low-pass filter, thresholds, and erodes an image mask.

    :param image: Input image mask in grayscale.
    :param blur_kernel_size: Size of the Gaussian blur kernel.
    :param threshold: Threshold value for binary thresholding.
    :param erosion_size: Size of the kernel for erosion.
    :return: Processed image.
    """
    # Apply Gaussian Blur for low-pass filtering
    blurred = cv2.GaussianBlur(mask, blur_kernel_size, 0)
    # Apply thresholding
    _, thresholded = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    # Create erosion kernel
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Apply erosion
    eroded = cv2.erode(thresholded, erosion_kernel, iterations=1)
    return eroded

def visualize_keypoints(image, keypoints):
    for k in keypoints:
        cv2.circle(image, k, 10, (100,100,100), -1)
    return image
