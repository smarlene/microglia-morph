import numpy as np

no_tissue = [235.61109337, 234.91000469, 233.74045357]

def filter_tissue(patch):
    """Return true if the patch is not tissue, false is the patch is tissue"""
    if patch.shape[2] == 4:
        patch = patch[:, :, :3]
    img_mean = np.mean(patch, axis=(0, 1))
    abs_diff = np.abs(np.max(img_mean) - np.min(img_mean))
    return abs_diff < 4.5 or np.allclose(no_tissue, img_mean, rtol=0, atol=4)


def filter_size(labels, num_labels, threshold=4000):
    """Given masks for each cell, filter out cells below a certain threshold and array of binary masks representing each remaining cell"""
    segmented_cells = []
    for i in range(num_labels):
        if np.count_nonzero(labels == i + 1) < threshold:
            labels[labels == i + 1] = 0

        cell = np.zeros(labels.shape, dtype=np.int32)
        cell[labels == i + 1] = 255
        segmented_cells.append(cell)
    return labels, segmented_cells