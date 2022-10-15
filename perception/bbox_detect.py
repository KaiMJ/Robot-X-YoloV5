import matplotlib.pyplot as plt
import torch
import cv2
from copy import deepcopy

weights_path = '/home/mkim/Documents/RobotX/Buoy/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)

# plot_path = "/home/mkim/Documents/RobotX/Buoy/yolov5/perception/plots/"
# test_image_path = "/home/mkim/Documents/RobotX/Buoy/yolov5/RobotX-Buoy-Detection-7/test/images/9_jpg.rf.0742fd2b9691e8f00d5814ebac2fb097.jpg"
# test_image_path = "/home/mkim/Documents/RobotX/Buoy/yolov5/RobotX-Buoy-Detection-7/test/images/92_jpg.rf.5143d76b7afac39c07ae9619003f8b92.jpg"

# test_label_path = "/home/mkim/Documents/RobotX/Buoy/yolov5/RobotX-Buoy-Detection-7/test/labels/9_jpg.rf.0742fd2b9691e8f00d5814ebac2fb097.txt"
# test_label_path = "/home/mkim/Documents/RobotX/Buoy/yolov5/RobotX-Buoy-Detection-7/test/labels/92_jpg.rf.5143d76b7afac39c07ae9619003f8b92.txt"

label2class = {
    '0': 'blue-buoy',
    '1': 'pink-buoy',
    '2': 'white-buoy'
}

label2color = {
    '0': (0, 0, 255),
    '1': (255, 0, 0),
    '2': (0, 255, 0)
}

def getDetection(image, plt_path=None):
    '''
    Given input image, return bounding box coordinates and class label.

    Returns
        Class labels: (b, )
            '0' - 'blue-buoy'
            '1' - 'pink-buoy'
            '2' - 'white-buoy'
        Box coordinates in image (1448, 563, 3): (b, x1, y1, x2, y2)
        Labeled image: (1448, 563, 3)
    '''
    img = deepcopy(image)
    if image.shape == (1448, 568, 3): # resize
        img = cv2.resize(image, (640, 640, 3))
    
    detections = model(img)

    for pred in detections[0]:
        x1, y1, x2, y2, confidence, label = pred
        label = str(int(label))
        x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), label2class[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), label2color[label], 5)

    img = cv2.resize(img, (1448, 568))
    if plt_path:
        plt.imshow(img)
        plt.savefig(plt_path)

    return detections[0][:, -1].cpu(), detections[0][:, :4].cpu(), img
