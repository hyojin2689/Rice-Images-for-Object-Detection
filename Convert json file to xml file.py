import json as j
import os
import random
import pandas as pd
import json
import xml.etree.cElementTree as e

path = '/content/drive/MyDrive/객체탐지/json_label'
file_list = os.listdir(path)
image_file_list_py = [file for file in file_list if file.endswith('.json')] 
image_file_list_py.sort()
image_file_list_py

for i in range(len(image_file_list_py)):

  with open("/content/drive/MyDrive/객체탐지/json_label/"+image_file_list_py[i]) as json_format_file:
   d = j.load(json_format_file)
 
  r = e.Element("annotation")

  e.SubElement(r,"folder").text = "testpart"
  e.SubElement(r,"filename").text = d[0]['image']
  for w in d[0]["annotations"]:
    object = e.SubElement(r,"object")
    e.SubElement(object,"name").text = "error"
    bndbox = e.SubElement(object,"bndbox")
    e.SubElement(bndbox,"xmin").text=str(w["coordinates"]["x"])
    e.SubElement(bndbox,"ymin").text=str(w["coordinates"]["y"])
    e.SubElement(bndbox,"xmax").text=str(w["coordinates"]["width"])
    e.SubElement(bndbox,"ymax").text=str(w["coordinates"]["height"])

  a=e.ElementTree(r)

  image_file_list_py2 = image_file_list_py[i].split(".")[0]

  print(image_file_list_py[i])

  a.write("/content/drive/MyDrive/json/" + image_file_list_py2)
