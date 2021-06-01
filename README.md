# Yolo Keras Face Detection

Implement Face  Mask detection

# Overview

## Functions

Face Mask Detection


## Requirements

OpenCV

Python 3.9

Darknet (for Training)

# Train

## Install

### Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

### Create dataset

### Train using Darknet

Here is a training using YoloV4.

# Python Code

        ""# -*- coding: utf-8 -*-
    """
    Created on Wed May 26 11:46:18 2021

    @author: Oguzhan
    """
   ### Add the library 
    import cv2
    import numpy as np
    import keyboard
   ### Set up the canvas
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    canvas_buyuk = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cv2.line(canvas_buyuk, (256, 0), (256, 1024), (255, 0, 0), thickness=1)
    cv2.line(canvas_buyuk, (767, 0), (767, 1024), (255, 0, 0), thickness=1)
    cv2.putText(canvas_buyuk, "People With Mask", (0, 20), font1, 0.5, (255, 0, 0))
    cv2.putText(canvas_buyuk, "LIVE", (257, 20), font1, 0.5, (255, 0, 0))
    cv2.putText(canvas_buyuk, "People Without Mask", (768, 20), font1, 0.5, (255, 0, 0))

   ### It was provided to take images on the webcam.
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
   
   ### This function provide taking images from video
    while True:
        ret, frame = vid.read()
        frame1 = frame

   ### Taking size of images

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

   ### Image resized the yolov4 size
        frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

   ### Classes what we use in the video
        labels = ["Mask", "Improperly", "No Mask"]

   ### Colors what we use for rectangle
        colors = ["0,255,255", "0,0,255"]
        colors = [np.array(color.split(",")).astype("int") for color in colors]
        colors = np.array(colors)
        colors = np.tile(colors, (18, 1))

   ### Yolo's datas for find mask
        model = cv2.dnn.readNetFromDarknet("detect_mask.cfg",
                                           "yolov3_mask_last.weights")

   ### This part of code take matrix from yolos data and identity the object and add to lists
        layers = model.getLayerNames()

        output_layer = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

        model.setInput(frame_blob)

        detection_layers = model.forward(output_layer)

        ids_list = []
        boxes_list = []
        confidences_list = []

   ### This part measured the object size
        for detection_layer in detection_layers:
            for object_detection in detection_layer:

                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]

   ### Object cover up with rectangle
                if confidence > 0.3:
                    label = labels[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                    start_x = int(box_center_x - (box_width / 2))
                    start_y = int(box_center_y - (box_height / 2))

                    ids_list.append(predicted_id)
                    confidences_list.append(float(confidence))
                    boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

        max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        for max_id in max_ids:
            max_class_id = max_id[0]
            box = boxes_list[max_class_id]

            start_x = box[0]
            start_y = box[1]
            box_width = box[2]
            box_height = box[3]

            predicted_id = ids_list[max_class_id]
            label = labels[predicted_id]
            confidence = confidences_list[max_class_id]

            end_x = start_x + box_width
            end_y = start_y + box_height

            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]

            label = "{}: {:.2f}%".format(label, confidence * 100)
            print("predicted object {}".format(label))

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

   ###We made face detection within specific range because of better output
            if start_x > 190 and end_x < 424:

   ### predicted_id = 0 the mask of id
                if predicted_id == 0:
                    y = 0
                    m = m + 1
                    f = m
                    cv2.imwrite("Mask\Mask" + str(m) + ".jpg",
                                frame1[start_y: start_y + box_height, start_x:start_x + box_width])

                    if f > 4:
                        f = 4
   ### This part provides that images which person takes mask, slide to down 
                    for i in range(f):
                        mask = cv2.imread("Mask\Mask" + str(m - i) + ".jpg")
                        mask = cv2.resize(mask, (245, 245))
                        canvas_buyuk[(y * 245) + 30:((y * 245) + 245) + 30, 0:245] = mask
                        y = y + 1

   ### predicted_id = 0 the no_mask of id
                if predicted_id == 2:
                    q = 0
                    n = n + 1
                    t = n
                    cv2.imwrite("No_Mask\\No_Mask" + str(n) + ".jpg",
                                frame1[start_y: start_y + box_height, start_x:start_x + box_width])

                    if t > 4:
                        t = 4
                    for i in range(t):
                        no_mask = cv2.imread("No_Mask\\No_Mask" + str(n - i) + ".jpg")
                        no_mask = cv2.resize(no_mask, (245, 245))
                        canvas_buyuk[(q * 245) + 30:((q * 245) + 245) + 30, 768:(245) + 768] = no_mask
                        q = q + 1
   ### Show the real time video
            frame1 = cv2.resize(frame, (511, 511))
            canvas_buyuk[30:541, 256:767] = frame1

   ### Image show with rectangle
        cv2.imshow("Detection Window", canvas_buyuk)

        if cv2.waitKey(20) & keyboard.is_pressed('esc'):
            break

    vid.release()
    cv2.destroyAllWindows()








