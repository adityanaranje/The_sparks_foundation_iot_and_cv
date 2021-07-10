# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:47:17 2021

@author: ADITYA NARANJE

The Sparks Foundation
Task-2 IOT AND COMPUTER VISION 

"""

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-c","--confidence",type=float,
                default=0.5,help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold",type=float,
                default=0.3,help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0,255, size=(len(LABELS),3),dtype="uint8")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

cap = cv2.VideoCapture("Traffic.mp4")

while True:
    _, image  = cap.read()
    image = cv2.resize(image, (1860, 900))
    (H,W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image,1/255.0, (416,416),swapRB=True,crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    
    boxes  = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence>args["confidence"]:
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"]) 
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x,y),(x+w,y+h),color,2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.putText(image, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            
            
    cv2.imshow("Object Detection",image)
    key = cv2.waitKey(1)
    
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()
                
                
                
    
