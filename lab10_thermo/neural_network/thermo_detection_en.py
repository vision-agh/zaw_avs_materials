from os.path import join
import cv2
import numpy as np

############################### Functions #################################


def detect(net, img):
    size = img.shape
    height = size[0]
    width = size[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes


def filter_boxes(boxes):
    all_paired_boxes = list()
    for ii, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        center_x1 = x1 + int(w1 / 2)
        center_y1 = y1 + int(h1 / 2)
        to_connect = [ii]
        for jj, box2 in enumerate(boxes):
            if jj != ii:
                x2, y2, w2, h2 = box2
                center_x2 = x2 + int(w2 / 2)
                center_y2 = y2 + int(h2 / 2)
                if abs(center_x2 - center_x1) < 10 and abs(center_y2 - center_y1) < 10:
                    to_connect.append(jj)
        all_paired_boxes.append(to_connect)
    all_paired_boxes = sorted(all_paired_boxes, key=lambda x: len(x), reverse=True)
    all_paired = list()
    final_boxes = list()
    for conn in all_paired_boxes:
        if all([a not in all_paired for a in conn]):
            for a in conn:
                all_paired.append(a)
            final_boxes.append(conn)
    out_boxes = [[int(sum([boxes[i][a] for i in elem]) / len(elem)) for a in range(4)] for elem in final_boxes]
    return out_boxes


def IoU(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    left = max([x1, x2])
    right = min([x1+w1, x2+w2])
    top = max([y1, y2])
    bottom = min([y1+h1, y2+h2])
    area1 = max([(right - left), 0]) * max([(bottom - top), 0])
    area2 = (w1 * h1) + (w2 * h2) - area1
    IoU = area1/area2
    return IoU


############# METHOD ###############
# Choose your fusion method
#FUSION = "LATE"
FUSION = "EARLY"

############# TODO0 ###############
# Set the path
test_rgb = "test_rgb"  # Path to the test_rgb folder
test_thermal = "test_thermal"  # Path to the test_thermal folder
###################################

net_fus = None
net_therm = None
net_rgb = None
if FUSION == "EARLY":
    net_fus = cv2.dnn.readNet('yolov3_training_last_f.weights', 'yolov3_testing_f.cfg')
if FUSION == "LATE":
    net_therm = cv2.dnn.readNet('yolov3_training_last_t.weights', 'yolov3_testing_t.cfg')
    net_rgb = cv2.dnn.readNet('yolov3_training_last_c.weights', 'yolov3_testing_c.cfg')

for i in range(200, 300):  # you can change the range up to 518
    path_rgb = join(test_rgb, f"img{i}.png")
    path_thermal = join(test_thermal, f"img{i}.png")
    img_rgb = cv2.imread(path_rgb)
    img_thermal = cv2.imread(path_thermal)
    img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
    out_img = None
    boxes = None
    if FUSION == "EARLY":
        ############ TODO1 ##################
        # Combine RGB with Thermal by following the instructions
        # 1. Create a new frame (numpy array) with the dimensions of an RGB image and call it new_fus
        # 2. Copy the first two channels of the RGB (img_rgb[:, :, :2]) to the first two channels of the new frame (new_fus[:, :, :2])
        # 3. The value of the third channel of the new frame is the maximum of the value of the third RGB channel and the thermal image (single channel)
        #    Use for example np.maximum(a, b). Where a and b are the 3rd RGB channel and thermal
        # 4. Convert "new_fus" to "uint8" (new_fus.astype("uint8"))
        new_fus = None

        ####################################
        out_img = new_fus
        boxes = detect(net_fus, new_fus)
    if FUSION == "LATE":
        out_img = img_rgb
        Rect1 = detect(net_therm, img_thermal)
        Rect2 = detect(net_rgb, img_rgb)
        ############ TODO2 ##################
        # "Rect1" i "Rect2" have the format [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
        # 1. Create a list "boxes_iou". Iterating in a double loop through "Rect1" and "Rect2"
        # check the IoU value of each rectangle in these lists (use the IoU() function defined
        # above which takes as arguments the two surrounding rectangles). If the IoU value for a
        # given pair is greater than 0, append to "boxes_iou" a list consisting of a tuple
        # (containing the indices of the currently processed surrounding rectangles) and
        # the calculated IoU value for them.
        # Example: In a given iteration of the double loop, we have reached the 3rd rectangle
        # from "Rect1" and the 4th rectangle from "Rect2". Their common IoU value is 0.55.
        # So we add the list [(3, 4), 0.55] to the array "boxes_iou".
        # 2. Then sort the "boxes_iou" descending by IoU value. Use the sorted() function with the parameters
        # key=lambda a: a[1] oraz reverse=True.
        # 3. Create empty lists "Rect1_paired", "Rect2_paired" and "paired_boxes".
        # 4. Create a loop through the elements of "boxes_iou". In each iteration, extract a tuple with index(elem[0])
        # and IoU value(elem[1]) from the currently processed element. If the first index from the tuple is not present
        # in the list "Rect1_paired" and the second element from the tuple is not present in the list "Rect2_paired",
        #  we append the tuple with indices(elem[0]) to "paired_boxes", and append the corresponding indices from the
        # tuple to the lists "Rect1_paired" and "Rect2_paired" (the first to the first list and the second to the second).
        # In this way, we get the list "paired_boxes", which contains pairs of indexes of rectangles from the lists
        # "Rect1" and "Rect2", which need to be paired (average their elements), which will be described in section 5.
        # 5. Finally, we create an empty list of "boxes". Iterating through the tuples in "paired_boxes", we extract from
        # "Rect1" the rectangle with the index stored as the first element of the tuple, and from "Rect2" we extract the
        # rectangle with the index stored as the second element of the tuple. The rectangles are in the form of a 4 element
        # list ([x1, y1, w1, h1]). Having 2 rectangles, i.e. 2 4-element lists (let's call them "r1" and "r2"),
        # we create one new 4-element list (let's call it "avg_r"), whose elements are the average of elements from
        # both lists with rectangles (we remember, that after calculating the average, the result should be converted
        #  to int, avg_r[0] = int((r1[0]/r2[0])/2) and so for all 4 elements. Finally we append "avg_r" to the "boxes" list.
        # This way the format of the "boxes" list will be the same as the format of the lists "Rect1" and "Rect2".
        boxes = None

        ######################################
    out_boxes = filter_boxes(boxes)
    for box in out_boxes:
        x, y, w, h = box
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.imshow('Image', out_img)
    cv2.waitKey(10)
