from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mask(row, box, img_width, img_height):
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8") * 255
    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    return np.array(img_mask)


def get_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return [[point[0][0], point[0][1]] for point in contours[0]]
    return []


def iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1, y1 = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    x2, y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return inter_area / (box1_area + box2_area - inter_area)


def apply_nms(objects, iou_threshold=0.7):
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while objects:
        result.append(objects[0])
        objects = [obj for obj in objects if iou(obj, objects[0]) < iou_threshold]
    return result


def draw_objects(img, objects):
    draw = ImageDraw.Draw(img, "RGBA")
    for obj in objects:
        x1, y1, x2, y2, label, prob, mask, polygon = obj
        polygon_points = [(int(x1 + point[0]), int(y1 + point[1])) for point in polygon]
        draw.polygon(polygon_points, fill=(0, 255, 0, 125))
    return img


def process_outputs(outputs, img_width, img_height):
    output0, output1 = outputs[0][0].transpose(), outputs[1][0].reshape(32, 160 * 160)
    boxes, masks = output0[:, :5], output0[:, 5:] @ output1
    boxes = np.hstack((boxes, masks))
    return boxes


def process_model_output(outputs, img, img_width, img_height, prob_threshold=0.5):
    boxes = process_outputs(outputs, img_width, img_height)

    yolo_classes = ["tumor"]
    objects = []
    for row in boxes:
        prob = row[4].max()
        if prob < prob_threshold:
            continue
        xc, yc, w, h = row[:4]
        class_id = row[4].argmax()
        x1, y1 = (xc - w / 2) / 640 * img_width, (yc - h / 2) / 640 * img_height
        x2, y2 = (xc + w / 2) / 640 * img_width, (yc + h / 2) / 640 * img_height
        label = yolo_classes[class_id]
        mask = get_mask(row[5:], (x1, y1, x2, y2), img_width, img_height)
        polygon = get_polygon(mask)
        objects.append([x1, y1, x2, y2, label, prob, mask, polygon])

    filtered_objects = apply_nms(objects)
    img_with_objects = draw_objects(img, filtered_objects)
    return img_with_objects
