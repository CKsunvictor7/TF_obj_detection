TF_obj_detection
================
this repository is based on TensorFlow Object Detection API
https://github.com/tensorflow/models

## For Environment Setup
git clone https://github.com/tensorflow/models as TF_framework  
replace the codes with in TF_framework/research/object_detection

## In this repository 
* TFrecords_maker.py creates TFRecord by the annotations from YOLOv3
* config contains the networks and hyper-parameters of object detection networks
* label_maps contains the label_maps for each dataset
* TF_Obj_Detector.py detects objects and save the results as images