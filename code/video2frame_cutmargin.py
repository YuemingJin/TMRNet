# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021. 
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------


import cv2
import os
import numpy as np
import PIL
from PIL import Image


source_path = "xxx/Videos/"  # original path
save_path = "xxx/frame/"  # save path


def change_size(image):
 
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom  

    pre1_picture = image[left:left + width, bottom:bottom + height]  

    #print(pre1_picture.shape) 
    
    return pre1_picture  


Video_num = 0

while True:

    Video_num = Video_num+1
    frame_num = 0
    if not os.path.exists(save_path+str(Video_num)):
        os.mkdir(save_path+str(Video_num)) 

    cap = cv2.VideoCapture(source_path+"Chole"+str(Video_num)+".mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        img_save_path = save_path+str(Video_num)+'/'+ str(frame_num)+".jpg"
        
        dim = (int(frame.shape[1]/frame.shape[0]*300), 300)
        
        frame = cv2.resize(frame,dim)
        frame = change_size(frame)
        img_result = cv2.resize(frame,(250,250))
        print(img_result.shape)
        print(img_result.dtype)

        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        img_result = PIL.Image.fromarray(img_result)
        print(img_result.mode)

        cv2.imwrite(img_save_path, img_result)
        print(img_save_path) 
        frame_num = frame_num+1
        cv2.waitKey(1)


    if Video_num==1:
        break
                

cap.release()
cv2.destroyAllWindows()
print("Cut Done")

