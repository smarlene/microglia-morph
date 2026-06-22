import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize, convex_hull_image
from skan import Skeleton, sholl_analysis
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, disk, binary_dilation, binary_closing
from scipy import ndimage
from skimage.measure import perimeter
import feret


def binarize_soma_dab(img_np, centroid, threshold=0.6):
    """
    Soma segmentation for DAB brightfield microglia images.
    Exploits: soma is darkest + thickest region.
    """

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (3,3), 1)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    dist_thresh = threshold
    _, soma_seed = cv2.threshold(dist, dist_thresh, 255, cv2.THRESH_BINARY)
    soma_seed = soma_seed.astype("uint8")

    n_comp, output, stats, centroids = cv2.connectedComponentsWithStats(soma_seed)
    cx, cy = centroid
    candidates = list(range(1, n_comp))

    if not candidates:
        _, soma_seed = cv2.threshold(dist, 0.2, 255, cv2.THRESH_BINARY)
        soma_seed = soma_seed.astype("uint8")
        n_comp, output, stats, centroids = cv2.connectedComponentsWithStats(soma_seed)
        candidates = list(range(1, n_comp))
    if not candidates:
        return None
    label = min(
        candidates, key=lambda i: np.hypot(centroids[i][0] - cx, centroids[i][1] - cy)
    )
    seed_mask = (output == label).astype("uint8") * 255

    markers = np.zeros_like(gray, dtype=np.int32)
    markers[seed_mask == 255] = 2
    markers[thresh == 0] = 1

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.watershed(img_bgr, markers)

    soma_mask = (markers == 2).astype("uint8") * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    soma_mask = cv2.morphologyEx(soma_mask, cv2.MORPH_CLOSE, kernel)

    soma = ndimage.binary_fill_holes(soma_mask).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    soma_mask = cv2.morphologyEx(soma, cv2.MORPH_OPEN, kernel)

    n_comp, output, stats, centroids = cv2.connectedComponentsWithStats(soma_mask)
    candidates = list(range(1, n_comp))
    label = min(
        candidates, key=lambda i: np.hypot(centroids[i][0] - cx, centroids[i][1] - cy)
    )
    soma_mask = (output == label).astype("uint8") * 255
    return soma_mask


def calculate_soma_area(mask):
    """Given image patch, extract soma and return area of mask, the mask, calculated soma centroid, and stats associated with soma (connectedComponentsWithStats)"""
    return np.count_nonzero(mask)


def calculate_soma_circularity(mask):
    binary_mask = mask > 0
    
    a = np.count_nonzero(binary_mask)
    p = perimeter(binary_mask)
    if p == 0.0:
        return 0
    return 4.0 * np.pi * a / (p**2)


def extract_skeleton(thresh, soma_mask):
    """Given image patch and soma mask, return skeleton corresponding to that specific cell"""

    skeleton = skeletonize(thresh).astype(np.uint8) * 255
    components, output, stats, centroids = cv2.connectedComponentsWithStats(skeleton)
    masked_labels = output * (soma_mask > 0)

    intersecting_labels = np.unique(masked_labels)
    intersecting_labels = intersecting_labels[intersecting_labels != 0]

    skeleton = np.isin(output, intersecting_labels).astype(np.uint8) * 255

    return skeleton


def extract_cell_mask_from_skeleton(thresh, skeleton):
    """
    Keep only the thresholded connected component that contains the selected skeleton.
    """
    bw = thresh > 0
    labels = measure.label(bw, connectivity=2)

    skel_labels = labels[skeleton > 0]
    skel_labels = skel_labels[skel_labels != 0]

    if skel_labels.size == 0:
        return None

    # pick the component most represented by the skeleton
    chosen_label = np.bincount(skel_labels).argmax()

    cell_mask = labels == chosen_label

    return (cell_mask.astype(np.uint8) * 255)

def calculate_sholl_analysis(skeleton, centroid, starting_val=10):
    """Given image patch, calculate sholl analysis and return intersection counds based on skeleton calculated in extract_skeleton"""
    radii = np.arange(starting_val, 250, 10)
    center, radii, counts = sholl_analysis(
        Skeleton(skeleton), center=(centroid[1], centroid[0]), shells=radii
    )
    return (radii, counts)

def sholl_decay(radii, counts):
        valid = counts > 0
        if valid.sum() < 3:
            return 0
        slope, _ = np.polyfit(radii[valid], np.log(counts[valid] + 1), 1)
        return slope

def calculate_terminal_points(skeleton):
    """Given a skeleton, localize terminal points of processes"""
    skel = skeleton / 255
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])

    filtered = cv2.filter2D(
        skel.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    rows, cols = np.where(filtered == 11)
    return (rows, cols)


def calculate_branching_nodes(skeleton):
    skel = skeleton / 255
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    filtered = cv2.filter2D(
        skel.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    branch_pixels = (skel == 1) & (filtered >= 3)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
        branch_pixels.astype(np.uint8)
    )
    
    total_branch_nodes = num_labels - 1
    return total_branch_nodes

def convex_hull_area(patch, thresh=True):
    if thresh:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        patch = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    conv_hull = convex_hull_image(patch)
    return conv_hull, np.count_nonzero(conv_hull)

def calculate_feret(soma_mask):
    maxf = feret.max(soma_mask)
    minf = feret.min(soma_mask)

    return maxf / minf


def fractal_dimension(Z):

    Z = np.asarray(Z, dtype=bool)

    def boxcount(arr, k):
        S = np.add.reduceat(
            np.add.reduceat(arr, np.arange(0, arr.shape[0], k), axis=0),
            np.arange(0, arr.shape[1], k),
            axis=1,
        )
        return np.count_nonzero(S)

    p = min(Z.shape)
    n = 2 ** int(np.floor(np.log2(p)))
    Z = Z[:n, :n]

    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

    counts = np.array([boxcount(Z, k) for k in sizes])

    valid = counts > 0
    sizes = sizes[valid]
    counts = counts[valid]

    coeffs = np.polyfit(np.log(1 / sizes), np.log(counts), 1)
    return coeffs[0]

def skeleton_length_corrected(skeleton):
    skel = (skeleton > 0).astype(np.uint8)
    
    straight = np.array([[0,1,0],[1,0,1],[0,1,0]])
    diagonal = np.array([[1,0,1],[0,0,0],[1,0,1]])
    
    straight_count = cv2.filter2D(skel.astype(np.float32), -1, straight)
    diag_count     = cv2.filter2D(skel.astype(np.float32), -1, diagonal)
    
    length = (np.sum(skel * straight_count) * 0.5 + np.sum(skel * diag_count) * np.sqrt(2) * 0.5)
    return length

def calculate_features(patch, centroid):
    """
    Extract microglia features (DEPRACATED)
    """
    try:

        soma_mask = binarize_soma_dab(patch, centroid)
        if soma_mask is None or np.count_nonzero(soma_mask) == 0:
            print("issue soma binarization")
            return None
        
        SOMA_AREA = np.count_nonzero(soma_mask)
        SOMA_CIRCULARITY = calculate_soma_circularity(soma_mask)
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        skeleton = extract_skeleton(thresh, soma_mask)

        if skeleton is None or np.count_nonzero(skeleton) == 0:
            print("issue skeletonization")
            return None

        cell_mask = extract_cell_mask_from_skeleton(thresh, skeleton)
        if cell_mask is None or np.count_nonzero(cell_mask) == 0:
            print("issue cell masking")
            return None

        CELL_AREA = np.count_nonzero(cell_mask)

        rows, cols = calculate_terminal_points(skeleton)
        TERMINAL_PTS = len(rows)

        BRANCHING_NODES = calculate_branching_nodes(skeleton)

        radii, counts = calculate_sholl_analysis(skeleton, centroid)
        
        SHOLL_COUNTS = counts
        RADII = radii
        if counts is None or len(counts) == 0:
            MAIN_BRANCHES = 0
            SHOLL_DEPTH = 0
            SHOLL_MAX = 0
            SHOLL_COUNTS = np.zeros(len(np.arange(10, 250, 10)))
            RADII = np.arange(10, 250, 10)
            SHOLL_AUC = 0
            SHOLL_DECAY = 0
        else:
            MAIN_BRANCHES = counts[0]
            index = np.where(counts == 0)[0]
            SHOLL_DEPTH = index[0] if len(index) > 0 else 250
            SHOLL_MAX = counts.max()
            SHOLL_AUC = np.trapz(SHOLL_COUNTS,RADII)
            SHOLL_DECAY = sholl_decay(radii,counts)

        conv, CHA = convex_hull_area(patch)
        DENSITY = CELL_AREA/CHA
        maxf = feret.max(conv)
        minf = feret.min(conv)
        SOMA_FERET_DIAM = maxf / minf
        
        skel_binary = skeleton > 0
        skeleton_length = skeleton_length_corrected(skeleton)
        hull_diameter = np.sqrt(np.count_nonzero(conv))
        TORTUOSITY = skeleton_length / hull_diameter if hull_diameter > 0 else 0

        dist = ndimage.distance_transform_edt(cell_mask > 0)
        skel_pixels = skel_binary & (cell_mask > 0)
        MEAN_THICKNESS = dist[skel_pixels].mean() * 2 if skel_pixels.any() else 0
        MAX_THICKNESS  = dist[skel_pixels].max()  * 2 if skel_pixels.any() else 0

        sk_obj = Skeleton(skeleton)
        branch_lengths = sk_obj.path_lengths()
        MEAN_SEGMENT_LENGTH = branch_lengths.mean() if len(branch_lengths) > 0 else 0
        MAX_SEGMENT_LENGTH  = branch_lengths.max()  if len(branch_lengths) > 0 else 0
        AMT_BRANCHES = len(branch_lengths)
        
        return [
            SOMA_AREA,
            CELL_AREA,
            TORTUOSITY,
            TERMINAL_PTS,
            MAIN_BRANCHES,
            AMT_BRANCHES,
            MEAN_SEGMENT_LENGTH,
            MAX_SEGMENT_LENGTH,
            MEAN_THICKNESS,
            MAX_THICKNESS,
            SHOLL_DEPTH,
            SHOLL_MAX,
            (RADII, SHOLL_COUNTS),
            SHOLL_AUC,
            SHOLL_DECAY,
            SOMA_CIRCULARITY,
            SOMA_FERET_DIAM,
            BRANCHING_NODES,
            CHA,
            DENSITY,
        ]
    except Exception:
        return None 
    

from skimage.filters import frangi, meijering, sato, hessian
import matplotlib.pyplot as plt
from skimage import io, color, exposure
from skimage.filters import meijering, threshold_otsu
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, disk
from skimage.morphology import binary_closing
from skimage import morphology, filters, measure
from scipy import ndimage
    
from skimage.segmentation import watershed
from scipy.spatial import cKDTree
from skimage import measure
import numpy as np
import warnings
from skan import draw, Skeleton

warnings.filterwarnings("ignore", category=FutureWarning)

def sholl_analysis_soma(skeleton, soma_mask, center, max_radius=250, step=10):
    """
    Carry out sholl analysis based on soma location for cell
    """
    boundary = measure.find_contours(soma_mask.astype(float), 0.5)[0]
    dists = np.sqrt(
        (boundary[:, 0] - center[1])**2 +
        (boundary[:, 1] - center[0])**2
    )
    soma_radius = dists.mean() + 5

    shells = np.arange(soma_radius, soma_radius + max_radius, step)

    skel_obj = Skeleton(skeleton)
    _, radii, counts = sholl_analysis(skel_obj, center=(center[1], center[0]), shells=shells)

    radii_from_boundary = radii - soma_radius

    first_valid = 0
    for i, (r, c) in enumerate(zip(radii_from_boundary, counts)):
        if r >= 0 and c >= 1:
            first_valid = i
            break

    radii_trimmed  = radii_from_boundary[first_valid:]
    counts_trimmed = counts[first_valid:]
    return radii_trimmed, counts_trimmed, soma_radius

def extract_feats2(patch,centroid,tree,df,lc):
    """
    Extract microglia features based on cell mask and skeletonization
    """

    H, W = patch.shape[:2]

    cx_local, cy_local = int(centroid[0]), int(centroid[1])
    x0, y0 = lc
    cx_abs = cx_local + x0
    cy_abs = cy_local + y0

    idx = tree.query_ball_point([(cx_abs, cy_abs)], r=max(H, W))[0]
    neighbors = df.iloc[idx]

    soma_mask = binarize_soma_dab(patch, centroid)
    if soma_mask is None or np.count_nonzero(soma_mask) == 0:
        print("issue soma binarization")
        return None
    
    SOMA_AREA = np.count_nonzero(soma_mask)

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    img_rescaled = exposure.equalize_adapthist(gray)

    ridge_map = meijering(img_rescaled, sigmas=range(1, 5), black_ridges=True)

    thresh = threshold_otsu(ridge_map)
    binary_map = ridge_map > thresh

    soma_bool = soma_mask > 0
    soma_distances, nearest_soma_idx = distance_transform_edt(~soma_bool, return_indices=True)
    
    valid_processes = binary_map & (soma_distances <= 15)
    
    bridges = np.zeros_like(binary_map, dtype=bool)
    
    proc_y, proc_x = np.where(valid_processes)
    

    if len(proc_y) > 0:
        target_y = nearest_soma_idx[0, proc_y, proc_x]
        target_x = nearest_soma_idx[1, proc_y, proc_x]
        

        from skimage.draw import line
        for py, px, ty, tx in zip(proc_y, proc_x, target_y, target_x):
            rr, cc = line(py, px, ty, tx)
            bridges[rr, cc] = True
            
    full_cell_mask = binary_map | bridges | soma_mask
    
    full_cell_mask = binary_closing(full_cell_mask, disk(2))

    cleaned = morphology.remove_small_objects(full_cell_mask, min_size=50)
    smoothed = morphology.binary_closing(cleaned, morphology.disk(3))
    cell_mask_processed = morphology.binary_opening(smoothed, morphology.disk(2))


    markers = np.zeros((H, W), dtype=np.int32)
    target_idx = None

    for local_idx, nrow in enumerate(neighbors.itertuples(index=False), start=1):
        lx = int(nrow.abs_cx) - lc[0]
        ly = int(nrow.abs_cy) - lc[1]

        if 0 <= lx < W and 0 <= ly < H:
            markers[ly, lx] = local_idx

            if int(nrow.abs_cx) == centroid[0]+lc[0] and int(nrow.abs_cy) == centroid[1]+lc[1]:
                target_idx = local_idx

    if target_idx is None:
        return None
        
    dist = distance_transform_edt(cell_mask_processed)
    territory = watershed(-dist, markers=markers, mask=cell_mask_processed > 0)

    cell_mask_processed = (territory == target_idx)
    skeleton = skeletonize(cell_mask_processed)


    RADII, COUNTS, soma_radius = sholl_analysis_soma(skeleton, soma_mask, center=centroid)
    SOMA_RADIUS = soma_radius - 5
    rows, cols = calculate_terminal_points(skeleton.astype(np.uint8) * 255)
    TERMINAL_PTS = len(rows)
    skel_obj = Skeleton(skeleton, spacing=0.2276) 
    branch_data = summarize(skel_obj, separator='-')
    cha_shape = convex_hull_image(cell_mask_processed)
    CHA = np.count_nonzero(cha_shape)
    CELL_AREA = np.count_nonzero(cell_mask_processed)

    if(CHA == 0):
        SOLIDITY = 0
    else:
        SOLIDITY = CELL_AREA/CHA
    
    a = CHA
    p = perimeter(cha_shape)
    if p == 0.0:
        CIRCULARITY = 0
    CIRCULARITY = 4.0 * np.pi * a / (p**2)

    features = {
        'SKEL_LENGTH': branch_data['branch-distance'].sum(),
        'N_BRANCHES': len(branch_data),
        'N_JUNCTIONS': (branch_data['branch-type'] == 2).sum(),
        'MEAN_BRANCH_LENGTH': branch_data['branch-distance'].mean(),
        'CHA': CHA,
        'CELL_AREA': CELL_AREA,
        'SOLIDITY': SOLIDITY,
        'CIRCULARITY': CIRCULARITY,
        'SOMA_AREA': SOMA_AREA,
        'SOMA_RADIUS': SOMA_RADIUS,
        'SHOLL_ANALYSIS': (RADII, COUNTS),
        'TERMINAL_PTS': TERMINAL_PTS
    }
    return features