# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

@author : Lucas Dienis
"""
#######################################################################################################################
#                                                  LIBRARIES                                                          #
#######################################################################################################################
import os, cv2, random
import numpy as np
import pandas as pd


from pprint import pprint
import json
import xml.etree.ElementTree as ET
from lxml import etree

#######################################################################################################################
#                                                  TEST WORKSPACE                                                     #
#######################################################################################################################

main_dir_serv = 'C:/Users/Paperspace/Anaconda3/envs/Tensorflow/darkflow/train_videos_2/'
test_dir_serv = 'C:/Users/Paperspace/Anaconda3/envs/Tensorflow/darkflow/test_videos_2/test_videos/'

main_dir = 'C:/Users/20007488/Documents/DataDriven_Fish_Challenge/Deep-learning-fish-20171023T131338Z-001/Deep-learning-fish/training_dataset/'


video_test='.mp4'
csv_file = '.csv'
json_file = '.json'
xml_type = '.xml'

file = '00WK7DR6FyPZ5u3A'


img_height = 720
img_width = 1280

training_folder = 'training_dataset/'
image_folder = 'Images'
annotations_folder = 'Annotations'

training_folder_serv = 'training_dataset/'
image_folder_serv = 'Images2'
annotations_folder_serv = 'Annotations2'
#######################################################################################################################
#                                                   EXTRACTING DATA FROM VIDEOS                                       #
#######################################################################################################################
xmlTemplate = """<?xml version="1.0" ?>
	<annotation>
	<folder>%(training_folder)s</folder>
	<filename>%(filename)s</filename>
	<size>
		<width>%(width)d</width>
		<height>%(height)d</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>%(name_class)s</name>
		<bndbox>
			<xmin>%(x_min)d</xmin>
			<ymin>%(y_min)d</ymin>
			<xmax>%(x_max)d</xmax>
			<ymax>%(y_max)d</ymax>
		</bndbox>
	</object>
</annotation>"""


for file in os.listdir(main_dir):
  if file[-5:] == '.json':     
    with open(main_dir + file) as data_file:    
      data = json.load(data_file)       
    frames = list()
    x_mins = list()
    y_mins = list()
    x_maxs = list()
    y_maxs = list()
    classes = list()
    filenames = list()         
    for x in range(0,len(data['detections'])):
      a = 0
      b = 0
      tempxmin=int(data['detections'][x]['w'])        	
      tempymin=int(data['detections'][x]['h'])
      tempxmax=int(data['detections'][x]['x'])
      tempymax=int(data['detections'][x]['y'])
      #Normalizing labelisation for x's
      if tempxmin > tempxmax:
        print('xmin and xmax interverted')
        print('before x-',tempxmin,'-',tempxmax,'-',tempymin,'-',tempymax)
        a = tempxmax
        tempxmax = tempxmin
        tempxmin = a
        print('after x-',tempxmin,'-',tempxmax,'-',tempymin,'-',tempymax)
      #Normalizing labelisation for y's
      if tempymin > tempymax:
        print('ymin and ymax interverted')
        print('before y-',tempxmin,'-',tempxmax,'-',tempymin,'-',tempymax)
        b = tempymax
        tempymax = tempymin
        tempymin = b
        print('after y-',tempxmin,'-',tempxmax,'-',tempymin,'-',tempymax)
      #Getting more data by enhancing labelisation
      print('xmin before enhancements : ',tempxmin)
      print('xmax before enhancements : ',tempxmax)
      print('ymin before enhancements : ',tempymin)
      print('ymax before enhancements : ',tempymax)
      #If it's needed to get more data with a wide effect
      if (tempymax - tempymin) >= (tempxmax - tempxmin) :
        tempxmax = tempxmax + ((tempxmax-tempxmin) / 2)
        tempxmin = tempxmin - ((tempxmax-tempxmin) / 2)
        print("Annotation enhanced with more width")
        #If it's needed to get more data with a height effect (if (tempxmax - tempxmin) > (tempymax - tempymin))
      else : 
        tempymax = tempymax + ((tempymax - tempymin) / 2)
        tempymin = tempymin - ((tempymax - tempymin) / 2)
        print("Annotation enhanced with more height")
        #Making sure that the annotations are not exceeding the picture width and height
      print('xmin after enhancements : ',tempxmin)
      print('xmax after enhancements : ',tempxmax)
      print('ymin after enhancements : ',tempymin)
      print('ymax after enhancements : ',tempymax)
      if tempxmin <= 0:
        tempxmin = 1
        print("Annotation crossing the xmin possible")
      if tempymin <= 0:
        tempymin = 1
        print("Annotation crossing the ymin possible")
      if tempxmax >= 1280:
        tempxmax = 1279
        print("Annotation crossing the xmax possible")
      if tempymax >= 720:
        tempymax = 719
        print("Annotation crossing the ymax possible")
      frames.append(data['detections'][x]['frame'])
      x_mins.append(tempxmin)
      y_mins.append(tempymin)
      x_maxs.append(tempxmax) 
      y_maxs.append(tempymax) 
      classes.append(data['tracks'][x]['subspecies'])
      vidcap = cv2.VideoCapture(main_dir + file[:-5] + video_test)
      success,image = vidcap.read()
      count = 0
      count_frame = 0
      success = True
      while success:
        success,image = vidcap.read()
        if count > int(frames[-1]):
          break
        if count == int(frames[count_frame]):
          data_xml = {'training_folder':"training_dataset",'filename':file[:-5] + "_frame_%d.jpg" % int(frames[count_frame]),'width':img_width,'height':img_height,'name_class':classes[count_frame],'x_min':int(x_mins[count_frame]),'y_min':int(y_mins[count_frame]),'x_max':int(x_maxs[count_frame]),'y_max':int(y_maxs[count_frame])}
          xml_string = xmlTemplate%data_xml
          xml_file = ET.ElementTree(ET.fromstring(xml_string))
          xml_file.write(os.path.join(main_dir + annotations_folder,file[:-5] + "_frame_%d" % int(frames[count_frame]) + xml_type))
          cv2.imwrite(os.path.join(main_dir + image_folder,file[:-5] + "_frame_" + frames[count_frame] + ".jpg"), image)     # save frame as JPEG file  
          print ("Frame caught: %d :" % int(frames[count_frame]),file,success)
          count_frame += 1
        count += 1
            
          

#######################################################################################################################
#                                                    FILE CLEANING                                                    #
#######################################################################################################################

file_to_delete = list()
for filename in os.listdir(main_dir + training_folder + image_folder):
    if os.path.getsize(main_dir + training_folder + image_folder + '/' + filename) == 0:
        file_to_delete.append(filename)
        
for filename in file_to_delete:
    os.remove(main_dir + training_folder + image_folder + '/' + filename)
    os.remove(main_dir + training_folder + annotations_folder + '/' + filename[:-4] + xml_type)
    print(filename)
    



#######################################################################################################################
#                                                    TRAINING INSTRUCTIONS                                            #
#######################################################################################################################
#To be executed in CLI, usage of a good GPU required, I advice to rent servers from Paperspace.com, they are pretty cheap.
python flow --model cfg/tiny-yolo-voc-fish.cfg --train --gpu .8 --trainer adam --dataset "C:/Users/Paperspace/Anaconda3/envs/Tensorflow/darkflow/train_videos_2/training_dataset/Images2" --annotation "C:/Users/Paperspace/Anaconda3/envs/Tensorflow/darkflow/train_videos_2/training_dataset/Annotations2" --lr 0.000001 --batch 16 --load -1 






#######################################################################################################################
#                                                    DETECTIONS COMMANDS                                              #
#######################################################################################################################

from darkflow.net.build import TFNet
import cv2
import pandas as pd
import math

options = {"model": "cfg/tiny-yolo-voc-fish.cfg", "load": -1, "threshold": 0.2, "gpu": 0.8, "backup":"ckpt/"}
tfnet = TFNet(options)




for file in os.listdir(test_dir):
#	if file[:-3] == '.csv':
#		break
	vidcap = cv2.VideoCapture(test_dir +  file[:-4] + video_test)
	data = pd.DataFrame()
	df = pd.DataFrame()
	count_frame = 0
	success = True
	while success:
		success,image = vidcap.read()
		if success is True:
			count_frame += 1
			result = tfnet.return_predict(image)
			print(result)
		if success is not True:
			break
		if len(result) > 0:
			df = pd.DataFrame(result,columns=['confidence','label','bottomright','topleft'])
			df['frame'] = count_frame
			df['file'] = file[:-4]
			df = df.sort_values(by='confidence')
			xmin = df['topleft'][0]['x']
			xmax = df['bottomright'][0]['x']
			ymin = df['topleft'][0]['y']
			ymax = df['bottomright'][0]['y']
			length =  math.sqrt(((ymax-ymin)**2) + ((xmax-xmin)**2))
			df['length'] = length 
			cv2.rectangle(image,(xmin,ymax),(xmax,ymin),(0,255,0),3)

		cv2.imshow(file,image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	#	data = data.append(df)
	# When everything done, release the capture
#	data.to_csv(test_dir + file[:-4] + csv_file, sep=',')
	vidcap.release()
	print('Video released at frame : ',count_frame)
	print('file released : ',file)
	cv2.destroyAllWindows()



imgcv = cv2.imread("./train_videos_2/training_dataset/Images2/00WK7DR6FyPZ5u3A_frame_0.jpg")
result = tfnet.return_predict(imgcv)
print(result)   

xmin = result[0]['topleft']['x']
xmax = result[0]['bottomright']['x']
ymin = result[0]['topleft']['y']
ymax = result[0]['bottomright']['y']

cv2.rectangle(imgcv,(xmin,ymax),(xmax,ymin),(0,255,0),3)
cv2.imshow('frame',imgcv)
cv2.waitKey(0)
cv2.destroyAllWindows()