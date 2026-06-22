import json
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def read_annotations(annotation_path):
    """Extract bounding box annotations from JSON ground truth"""

    with open(annotation_path) as f:
        data = json.load(f)

    annotations = data["annotations"]
    imgs = data["images"]
    img_dict = {i["id"]: i["file_name"] for i in imgs}
    annotations_dict = {}
    for i in annotations:
        img_name = img_dict[i["image_id"]]
        vals = i["bbox"]
        if img_name not in annotations_dict:
            annotations_dict[img_name] = []
        annotations_dict[img_name].append(vals)
    return annotations_dict


def read_annotations_points(json_path):
    """Extract point annotations from JSON ground truth"""

    with open(json_path, "r") as f:
        data = json.load(f)

    annotations_dict = {}
    for item in data["items"]:
        img_name = item["image"]["path"]
        points = []
        for i in item.get("annotations", []):
            if i.get("type") == "points":
                x, y = i["points"][:2]
                points.append((float(x), float(y)))

        annotations_dict[img_name] = points
    return annotations_dict


def iou(bb_gt, bb_pred):
    """Calculate IOU between predicted and ground truth bounding box, operating under the assumption that boxes are passed as: [x1,y1,x2,y2]"""

    x1 = max(bb_gt[0], bb_pred[0])
    x2 = min(bb_gt[2], bb_pred[2])
    y1 = max(bb_gt[1], bb_pred[1])
    y2 = min(bb_gt[3], bb_pred[3])

    w_inter = max(0.0, x2 - x1)
    h_inter = max(0.0, y2 - y1)

    area = w_inter * h_inter
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    area_pred = (bb_pred[2] - bb_pred[0]) * (bb_pred[3] - bb_pred[1])
    return area / (area_gt + area_pred - area)


def bulk_eval(
    ground_truth, preds, iou_threshold=0.5, dist_threshold=25, points=False, plot=False
):
    """Evaluate bounding box or soma localization for entire training or test split. Optionally plot the results for debugging"""

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_unit = 0
    if points:
        print("Pointwise evaluation")
    else:
        print("Bounding box evaluation")
    for img in preds.keys():
        if points:
            unit = "Pixelwise Euc. Dist."
            tp, fp, fn, mean_unit = calculate_accuracy_points(
                ground_truth, img, preds[img], dist_threshold, plot
            )
        else:
            unit = "IOU"
            tp, fp, fn, mean_unit = calculate_accuracy(
                ground_truth, img, preds[img], iou_threshold, plot
            )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_unit += mean_unit * tp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall > 0)
        else 0
    )
    mean_unit = total_unit / total_tp if total_tp > 0 else 0.0
    print(
        f"Evaluation output:\nTP:{total_tp}\nFP:{total_fp}\nFN:{total_fn}\nMean {unit}:{mean_unit}\nF1-score: {f1}"
    )
    return total_tp, total_fp, total_fn


def calculate_accuracy_points(ground_truth, img, preds, dist_threshold=25, plot=False):
    """Calcualte TP, FP, FN for soma localization given distance threshold. Optionally plot output for debugging"""

    if img in ground_truth:
        ground_truth_pts = ground_truth[img]
    else:
        ground_truth_pts = []

    if len(ground_truth_pts) == 0:
        if len(preds) != 0:
            print(f"Img:{img} not found in GT but cells were localized")
        return 0, len(preds), 0, 0.0

    if len(preds) == 0:
        return 0, 0, len(ground_truth_pts), 0.0

    preds_xy = [(x, y) for y, x in preds]
    cost = cdist(preds_xy, ground_truth_pts)
    pred_indices, gt_indices = linear_sum_assignment(cost)

    matches = [
        (preds[p], ground_truth_pts[g], cost[p, g])
        for p, g in zip(pred_indices, gt_indices)
        if cost[p, g] < dist_threshold
    ]

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(ground_truth_pts) - tp
    mean_dist = sum(m[2] for m in matches) / tp if tp > 0 else 0.0

    return tp, fp, fn, mean_dist


def calculate_accuracy(ground_truth, img, preds, iou_threshold=0.5, plot=False):
    """Calcualte TP, FP, FN for bounding box localization given IOU threshold. Optionally plot output for debugging"""

    if img in ground_truth:
        ground_truth_bbs = ground_truth[img]
    else:
        ground_truth_bbs = []

    ground_truth_bbs = [[x, y, x + w, y + h] for (x, y, w, h) in ground_truth_bbs]

    if len(ground_truth_bbs) == 0:
        if len(preds) != 0:
            print(f"Img:{img} not found in GT but cells were localized")
        return 0, len(preds), 0, 0.0

    matched_gt = set()
    matches = []
    gt_iou_labels = {}

    for pred in preds:
        max_iou = 0
        max_idx = None

        for i, gt in enumerate(ground_truth_bbs):
            if i in matched_gt:
                continue
            score = iou(pred, gt)
            if score > max_iou:
                max_iou = score
                max_idx = i

        if max_idx is not None and max_iou >= iou_threshold:
            matched_gt.add(max_idx)
            matches.append((pred, ground_truth_bbs[max_idx], max_iou))
            gt_iou_labels[max_idx] = max_iou

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(ground_truth_bbs) - tp

    mean_iou = sum(m[2] for m in matches) / tp if tp > 0 else 0.0

    return tp, fp, fn, mean_iou
