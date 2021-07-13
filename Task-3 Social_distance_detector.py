"""
The Sparks Foundation
Task-3 Social Distance Detection
@author: ADITYA NARANJE
"""

# Importing required libraries
import numpy as np
import cv2


# Load Yolo Model
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
net = cv2.dnn.readNet(weightsPath, configPath)

labelsPath = "coco.names"
classes = open(labelsPath).read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
np.random.uniform(0, 255, size=(len(classes),3))


# Load image
img  = cv2.imread("cri.jpg")


#Euclidean distance
def E_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 +  (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = E_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2

    if 0 < c_d <= 0.15 * calib:
        return 1
    elif 0.15 < c_d <= 0.2 * calib:
        return 2
    else:
        return 0

img = cv2.resize(img,(1540,876))
   
height,width=img.shape[:2]
q=width

#height, width, channels = img.shape
img =img[0:height, 0:q]
height,width=img.shape[:2]

# Detecting objects 0.005
blob = cv2.dnn.blobFromImage(img,0.005, (416, 416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(output_layers)


# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #0.5 is the threshold for confidence
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
             # w, h = width, height of the box

            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

font = cv2.FONT_HERSHEY_DUPLEX   

if len(indexes)>0:        
    status=list()        
    idf = indexes.flatten()        
    close_pair = list()        
    s_close_pair = list()        
    center = list()        
    dist = list()  
    # Storing values of center      
    for i in idf:            
        (x, y) = (boxes[i][0], boxes[i][1])            
        (w, h) = (boxes[i][2], boxes[i][3])            
        center.append([int(x + w / 2), int(y + h / 2)])            
        status.append(0)            

    for i in range(len(center)):            
        for j in range(len(center)):                
            #compare the closeness of two values

            g=isclose(center[i], center[j])                
            if g ==1:                    
                close_pair.append([center[i],center[j]])                    
                status[i] = 1                    
                status[j] = 1                    
            elif g == 2:                    
                s_close_pair.append([center[i], center[j]])                    
                if status[i] != 1:                        
                    status[i] = 2                        
                if status[j] != 1:                        
                    status[j] = 2

    total_p = len(center)        
    low_risk_p = status.count(2)        
    high_risk_p = status.count(1)        
    safe_p = status.count(0)        
    kk = 0        
    for i in idf:               
        tot_str = "NUMBER OF PEOPLE: " + str(total_p)            
        high_str = "RED ZONE: " + str(high_risk_p)            
        low_str = "YELLOW ZONE: " + str(low_risk_p)            
        safe_str = "GREEN ZONE: " + str(safe_p) 

        cv2.putText(img, tot_str, (10, height - 150),font, 0.8, (255, 255, 255), 2)           
        cv2.putText(img, safe_str, (10, height - 125),font, 0.8, (0, 255, 0), 2)           
        cv2.putText(img, low_str, (10, height - 100),font, 0.8, (0, 255, 255), 2)          
        cv2.putText(img, high_str, (10, height - 75),font, 0.8, (0, 0, 1255),2)        

        (x, y) = (boxes[i][0], boxes[i][1])            
        (w, h) = (boxes[i][2], boxes[i][3])       
        
        #color of the rectangle
        if status[kk] == 1:                
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif status[kk] == 0:                
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        kk += 1


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
