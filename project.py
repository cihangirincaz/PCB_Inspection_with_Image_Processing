import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # for XML parse
from collections import defaultdict
import os

############################################
# 1. AUXILIARY FUNCTIONS
############################################

def parse_annotation_xml(xml_path):
    """
    Reads filename and bounding box information from the given XML file (similar to Pascal VOC) and returns a dictionary.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text  # "01_open_circuit_01.jpg" vb.

    bboxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox  = obj.find('bndbox')
        xmin  = int(bbox.find('xmin').text)
        ymin  = int(bbox.find('ymin').text)
        xmax  = int(bbox.find('xmax').text)
        ymax  = int(bbox.find('ymax').text)

        bboxes.append({
            "label": label,
            "x_min": xmin,
            "y_min": ymin,
            "x_max": xmax,
            "y_max": ymax
        })

    return {
        "filename": filename,
        "bboxes": bboxes
    }

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def scale_image(image, scale_factor):
    (h, w) = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled

def compute_angle_and_scale(reference_image, test_image, max_matches=50):
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort by distance and use the first max_matches
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    pts_ref = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts_test = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    angles = []
    scales = []
    for i in range(len(pts_ref) - 1):
        p1 = pts_ref[i]
        p2 = pts_ref[i+1]
        p3 = pts_test[i]
        p4 = pts_test[i+1]

        vec_ref = p2 - p1
        vec_test = p4 - p3

        angle_ref = np.degrees(np.arctan2(vec_ref[1], vec_ref[0]))
        angle_test = np.degrees(np.arctan2(vec_test[1], vec_test[0]))
        angle_diff = angle_ref - angle_test
        angles.append(angle_diff)

        len_ref = np.linalg.norm(vec_ref)
        len_test = np.linalg.norm(vec_test)
        if len_test != 0:
            scale_ratio = len_ref / len_test
            scales.append(scale_ratio)

    mean_angle = np.mean(angles) if len(angles) > 0 else 0.0
    mean_scale = np.mean(scales) if len(scales) > 0 else 1.0

    return mean_angle, mean_scale

def match_and_align_images(reference_image, test_image):
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(test_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:50]

    pts_ref = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)
    (h, w) = reference_image.shape[:2]
    aligned_test_image = cv2.warpPerspective(test_image, matrix, (w, h))

    return aligned_test_image, matrix

def detect_defects(reference_image, aligned_test_image, threshold_val=50):
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_test_image, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, test_gray)
    _, diff_thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # If we want to eliminate very small boxes (noise), like if w*h > some_value:
        if w > 2 and h > 2:
            defect_boxes.append((x, y, x + w, y + h))

    return diff_thresh, defect_boxes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

############################################
# 2. MAIN STREAM
############################################

# 2.1. Reference/Test Image Paths
reference_path = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/Reference/01.JPG"
#test_path      = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/rotation/Open_circuit_rotation/01_open_circuit_02.jpg"
test_path      = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/rotation/Missing_hole_rotation/01_missing_hole_01.jpg"
#test_path      = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/rotation/Mouse_bite_rotation/01_mouse_bite_05.jpg"

reference_image = cv2.imread(reference_path)
test_image      = cv2.imread(test_path)

# Simple error checking
if reference_image is None:
    print("Reference image not found:", reference_path)
if test_image is None:
    print("Test image not found:", test_path)

# 2.2. Angle and Scale Calculation
angle_diff, scale_factor = compute_angle_and_scale(reference_image, test_image)
rotated_test_image       = rotate_image(test_image, angle_diff)
scaled_test_image        = scale_image(rotated_test_image, scale_factor)

# 2.3. Final Matching and Alignment with Homography
aligned_test_image, H_test_to_ref = match_and_align_images(reference_image, scaled_test_image)

# 2.4. Defect Detection
threshold_val = 40
diff_thresh, detected_defect_boxes = detect_defects(reference_image, aligned_test_image, threshold_val)

############################################
# 3. SHOWING DEFECTS
############################################
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1);
plt.title("Reference");
plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2);
plt.title("Aligned Test");
plt.imshow(cv2.cvtColor(aligned_test_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3);
plt.title("Differences");
plt.imshow(diff_thresh, cmap='gray')
plt.show()

############################################
# 4. XML ANNOTATION READING
############################################

# Example: We can parse a single XML file.
#xml_path = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/Annotations/Open_circuit/01_open_circuit_02.xml"
xml_path = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/Annotations/Missing_hole/01_missing_hole_01.xml"
#xml_path = "/Users/cihangirincaz/Desktop/image_processing_final/PCB_DATASET/Annotations/Mouse_bite/01_mouse_bite_05.xml"

annotation_data = parse_annotation_xml(xml_path)

# Now, we create our "annot_bboxes" list:
annot_bboxes = []
for box in annotation_data["bboxes"]:
    label = box["label"]
    xmin  = box["x_min"]
    ymin  = box["y_min"]
    xmax  = box["x_max"]
    ymax  = box["y_max"]

    # (x_min, y_min, x_max, y_max, label) We convert it to format
    annot_bboxes.append( (xmin, ymin, xmax, ymax, label) )

############################################
# 5. IoU COMPARISON AND REPORTING
############################################

IOU_THRESHOLD = 0.3

matched_annotations = []
matched_detections  = []

# To find the detection with the highest IoU for each annotation
for a_idx, (axmin, aymin, axmax, aymax, a_label) in enumerate(annot_bboxes):
    best_iou  = 0.0
    best_d_idx = -1
    for d_idx, (dxmin, dymin, dxmax, dymax) in enumerate(detected_defect_boxes):
        iou_val = compute_iou((axmin, aymin, axmax, aymax),
                              (dxmin, dymin, dxmax, dymax))
        if iou_val > best_iou:
            best_iou   = iou_val
            best_d_idx = d_idx

    if best_iou >= IOU_THRESHOLD:
        matched_annotations.append(a_idx)
        matched_detections.append(best_d_idx)

total_annotations = len(annot_bboxes)
total_detections  = len(detected_defect_boxes)
true_positives    = len(matched_annotations)
false_negatives   = total_annotations - true_positives
false_positives   = total_detections - true_positives

precision = true_positives / float(true_positives + false_positives + 1e-6)
recall    = true_positives / float(true_positives + false_negatives + 1e-6)
f1        = 2 * (precision * recall) / (precision + recall + 1e-6)

print("======== GENERAL STATISTICS ========")
print(f"Total Annotation (True Defect): {total_annotations}")
print(f"Total Detections (Detected Defects): {total_detections}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")

# Report based on defect type
type_counts = defaultdict(lambda: {"TP": 0, "FN": 0})

# Count all annotations type by type
for idx, bbox in enumerate(annot_bboxes):
    label = bbox[4]
    type_counts[label]["FN"] += 1

# Matching annotations -> TP
for matched_idx in matched_annotations:
    label = annot_bboxes[matched_idx][4]
    type_counts[label]["FN"] -= 1
    type_counts[label]["TP"] += 1

print("\n======== REPORT BASED ON DEFECT TYPE ========")
for defect_type, counts in type_counts.items():
    tp = counts["TP"]
    fn = counts["FN"]
    print(f"Defect Type: {defect_type}")
    print(f"  TP = {tp}, FN = {fn}")

########################################
# 6) DRAWING A BOX VISUALLY
########################################
result_vis = reference_image.copy()

# Annotation boxes (Red)
for (x_min, y_min, x_max, y_max, label) in annot_bboxes:
    cv2.rectangle(result_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(result_vis, label, (x_min, y_min-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Detected boxes (Green)
for (dxmin, dymin, dxmax, dymax) in detected_defect_boxes:
    cv2.rectangle(result_vis, (dxmin, dymin), (dxmax, dymax), (0, 255, 0), 2)

plt.figure(figsize=(10, 8))
plt.title("Annotation (Red) & Detection (Green)")
plt.imshow(cv2.cvtColor(result_vis, cv2.COLOR_BGR2RGB))
plt.show()

