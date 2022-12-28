# Import libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from mpl_toolkits.axes_grid1 import ImageGrid

# Observe images from dataset
im1 = cv2.imread(f"images/normal/aviao 1.jpg")
im2 = cv2.imread(f"images/normal/aviao 2.jpg")
im3 = cv2.imread(f"images/normal/aviao 3.jpg")

fig = plt.figure(figsize=(15., 15.))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1)

for ax, im in zip(grid, [im1, im2, im3]):
    ax.imshow(im, )

plt.show()

# Paths to models and configs
MODEL_FILE = "MobileNet/frozen_inference_graph.pb"
CONFIG_FILE = "MobileNet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
CLASS_FILE = "MobileNet/object_detection_classes_coco.txt"

# Class names
with open(CLASS_FILE, 'r') as f:
    class_names = f.read().split('\n')

# Define threshold
CONFIDENCE_THRESHOLD = 0.50

# Load model
model = cv2.dnn.readNet(model=MODEL_FILE, config=CONFIG_FILE, framework="TensorFlow")

predicted_boxes = pd.DataFrame({})
TP = 0
FP = 0

# Get location of Detected Objects
for image_path in os.listdir("images/normal"):

    # Read image and get shape
    img = cv2.imread(f"images/normal/{image_path}")
    img_height, img_width, img_channels = img.shape
    img_height, img_width, channels = img.shape

    # Normalize data dimensions
    blob = cv2.dnn.blobFromImage(image=img, size=(300, 300), swapRB=True)
    model.setInput(blob)
    output = model.forward()

    for detection in output[0, 0, :, :]:   #[0, class_id, trust, box_x, box_y, box_w, box_h]
        
        # Trust
        confidence = detection[2]
        if detection[1] == 5:
            if confidence > CONFIDENCE_THRESHOLD:

                class_id = detection[1] # Class id
                class_name = class_names[int(class_id) - 1]

                # obter as coordenadas e dimensoes das bounding boxes, normalizadas para coordenadas da imagem
                bbox_x = detection[3] * img_width
                bbox_y = detection[4] * img_height
                bbox_width = detection[5] * img_width
                bbox_height = detection[6] * img_height

                predicted_boxes = predicted_boxes.append({"name": image_path,
                                                          "x": bbox_x,
                                                          "y": bbox_y,
                                                          "x_size": bbox_width,
                                                          "y_size": bbox_height}, ignore_index=True)

                # colocar retangulos e texto a marcar os objetos identificados
                cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_width), int(bbox_height)), 1, thickness=2)
                cv2.putText(img, class_name, (int(bbox_x), int(bbox_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
                TP += 1
            else:
                FP += 1
            
    cv2.imwrite("images/predicted/"+image_path, img)

# Get locations of Ground Truth
actual_boxes = pd.DataFrame({})

for image_path in os.listdir("images/gt"):
    tree = ET.parse(f'images/gt/{image_path}')
    root = tree.getroot()

    sample_annotations = []

    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)
        
        actual_boxes = actual_boxes.append({"name": image_path,
                                            "x": xmin,
                                            "y": ymin,
                                            "x_size": xmax,
                                            "y_size": ymax}, ignore_index=True)

# Function to calculate the intersection over union

def IoU(boxA, boxB):

    # determine the (x, y) - coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[3], boxB[3])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[4], boxB[4])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Area of both boxes
    boxAArea = (boxA[2] - boxA[1] + 1) * (boxA[4] - boxA[3] + 1)
    boxBArea = (boxB[2] - boxB[1] + 1) * (boxB[4] - boxB[3] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

actual_boxes_sorted = actual_boxes.sort_values(by=["name", "x"])
predicted_boxes_sorted = predicted_boxes.sort_values(by=["name", "x"])

print(actual_boxes_sorted.head(5))

print(predicted_boxes_sorted.head(5))

iou_results = []

for i in range(0, len(actual_boxes_sorted)):
    iou_results.append({"img_name": predicted_boxes_sorted.iloc[i]["name"], 
                        "iou_result": IoU(predicted_boxes_sorted.iloc[i], actual_boxes_sorted.iloc[i])})
                        
    print(predicted_boxes_sorted.iloc[i]["name"], IoU(predicted_boxes_sorted.iloc[i], actual_boxes_sorted.iloc[i]))

# Average Intersection over Union of the detected objects
mean_iou = sum([x['iou_result'] for x in iou_results]) / len(iou_results)
print("Mean IoU = ", mean_iou)

print(TP)

print(FP)

FN = len(actual_boxes_sorted) - len(predicted_boxes_sorted)
print(FN)

precision = TP / (TP + FP)
print("Precision = ", precision)

recall = TP / (TP+FN)
print("Recall =", recall)