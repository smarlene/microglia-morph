import cv2
import numpy as np 
import json
import matplotlib.pyplot as plt
import pandas as pd 

def pre_process_binarize(img):
    """Binarize image patches for cell localization"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    inverted = ~equalized
    _, thresholded = cv2.threshold(inverted, 248, 255, 0)
    return thresholded


def create_patches(
    img, patch_size, stride, filter=(lambda a: False), preprocessing=(lambda a: a)
):
    """Split WSIs into tiles dictated by patch_size. Overlap is dictated by stride. Optionally filter and preprocess tiles before they are returned"""
    height = img.height
    width = img.width

    for x in range(0, width - patch_size + 1, stride):
        for y in range(0, height - patch_size + 1, stride):
            # x = left, y = top
            cropped = img.crop(x, y, patch_size, patch_size)
            patch = np.ndarray(
                buffer=cropped.write_to_memory(),
                dtype=np.uint8,
                shape=[cropped.height, cropped.width, cropped.bands],
            )
            if filter(patch):
                continue
            yield preprocessing(patch), patch, (y, x)


def calculate_bb(segmented_cells, segmentation_value=255):
    """Given segmented cells, calculate bounding boxes that surround them"""
    bboxs = []
    for cell in segmented_cells:
        segment = np.where(cell == segmentation_value)
        if segment[0].size == 0:
            continue
        x_min = int(np.min(segment[1]))
        x_max = int(np.max(segment[1]))
        y_min = int(np.min(segment[0]))
        y_max = int(np.max(segment[0]))
        bbox = x_min, y_min, x_max, y_max
        bboxs.append(bbox)
    return bboxs



def extract_points_from_json(json_path, debug=False):
    """Extract cell centroid coordinates from ground truth JSON file. If debug is true plot the corresponding heatmap"""
    with open(json_path, "r") as f:
        data = json.load(f)

    annotations_dict = {}
    for item in data["items"]:
        image_path = item["image"]["path"]
        points = []

        for i in item.get("annotations", []):
            if i.get("type") == "points":
                x, y = i["points"][:2]
                points.append((float(x), float(y)))

        annotations_dict[image_path] = points

    if debug:
        # DEBUG: PLOT HEATMAPS FOR IMAGES
        for f in annotations_dict:
            plt.imshow(make_gaussian_heatmap((512, 512), annotations_dict[f]))
            plt.show()

    return annotations_dict


def make_gaussian_heatmap(image_shape, points, sigma=15.0):
    """Given points and an image shape, create a heatmap by placing gaussians over each point with the specified sigma value"""
    h, w = image_shape
    yy, xx = np.mgrid[0:h, 0:w]
    heatmap = np.zeros((h, w), dtype=np.float32)

    two_sigma_sq = 2.0 * sigma * sigma

    for x, y in points:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / two_sigma_sq).astype(np.float32)
        heatmap += g

    return heatmap


def extract_bbs(file_path, results):
    file = pd.read_parquet(file_path)
    for index, r in file.iterrows():
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        results.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2)})

    return results