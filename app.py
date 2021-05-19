from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, send_file
import torch, torchvision
from pycocotools.coco import COCO
import numpy as np
import pylab
import random
import os
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image
from scipy.spatial import distance
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import matplotlib.pyplot as plt
import skimage.io as io
from detectron2.data.datasets import register_coco_instances
import urllib.request
from werkzeug.utils import secure_filename

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
plt.rcParams["figure.figsize"] = [16,9]

app = Flask(__name__)

app.secret_key = "caircocoders-ednalan"
 
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DISPLAY_FOLDER = 'static/display/'
app.config['DISPLAY_FOLDER'] = DISPLAY_FOLDER
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

PATH = "entire_model.pt"
PATH_mul = "entire_model_mul.pt"
model1 = torch.load(PATH)
model2 = torch.load(PATH_mul)
damage_predictor = DefaultPredictor(model1)
part_predictor = DefaultPredictor(model2)

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/detect',methods=['POST','GET'])
def detect():
    def detect_damage_part(damage_dict, parts_dict):
            try:
              max_distance = 1e9
              stack = []
              assert len(damage_dict)>0, "AssertError: damage_dict should have atleast one damage"
              assert len(parts_dict)>0, "AssertError: parts_dict should have atleast one part"
              max_distance_dict = dict(zip(damage_dict.keys(),[max_distance]*len(damage_dict)))
              part_name = dict(zip(damage_dict.keys(),['']*len(damage_dict)))

              for y in parts_dict.keys():
                  for x in damage_dict.keys():
                    dis = distance.euclidean(damage_dict[x], parts_dict[y])
                    
                    if dis <= max_distance_dict[x]:
                      part_name[x] = y.rsplit('_',1)[0]
                      stack.append(part_name[x])
              
              return list(set(stack))
            except Exception as e:
              print(e)

    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files[]')
    
     
    errors = {}
    success = False
     
    for file in files:      
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
            
        else:
            errors[file.filename] = 'File type is not allowed'
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
      
    
    im = io.imread(os.path.join(app.config['UPLOAD_FOLDER']) + '/' +filename)
    
    damage_class_map= {0:'damage'}
    parts_class_map={0:'headlamp',1:'rear_bumper', 2:'door', 3:'hood', 4: 'front_bumper'}



#damage inference
    damage_outputs = damage_predictor(im)
    damage_v = Visualizer(im[:, :, ::-1],metadata=MetadataCatalog.get("car_dataset_val"), scale=0.5, instance_mode=ColorMode.IMAGE_BW )
    damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

  #part inference
    parts_outputs = part_predictor(im)
    parts_v = Visualizer(im[:, :, ::-1],metadata=MetadataCatalog.get("car_mul_dataset_val"), scale=0.5,instance_mode=ColorMode.IMAGE_BW )
    parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))


    damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
    damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
    damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))

    parts_prediction_classes = [ parts_class_map[el] + "_" + str(indx) for indx,el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
    parts_polygon_centers =  parts_outputs["instances"].pred_boxes.get_centers().tolist()

      #Remove centers which lie in beyond 800 units
    parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
    parts_dict = dict(zip(parts_prediction_classes,parts_polygon_centers_filtered))


  #cv2.imshow('Damages',damage_out.get_image()[:, :, ::-1])
  #cv2.waitKey(0)
    def detect_damage_img(dimage):
          
        try:
          #cv2.imshow('Damages',damage_out.get_image()[:, :, ::-1])
          #cv2.waitKey(0)

          return dimage

        except Exception as e:
          print(e)

      #cv2.imshow('image', detect_damage_img(damage_out.get_image()[:, :, ::-1]))
      #cv2.waitKey(0)
      #print("Damaged Parts: ",detect_damage_part(damage_dict,parts_dict))
    
    #return detect_damage_img(damage_out.get_image()[:, :, ::-1]),detect_damage_part(damage_dict,parts_dict)
    #a = detect_damage_img(damage_out.get_image()[:, :, ::-1])
    a = cv2.imwrite(app.config['DISPLAY_FOLDER']+ filename, detect_damage_img(damage_out.get_image()[:, :, ::-1]))
    imagepath = os.path.join(app.config['DISPLAY_FOLDER'], filename)
    return render_template('index.html', 
    prediction_text='Damaged Parts are  {}'.format(detect_damage_part(damage_dict,parts_dict)),imagep=send_from_directory(app.config['DISPLAY_FOLDER'],filename, mimetype='image/jpg',as_attachment=False)
    )

#im = io.imread("/home/sandaruwan/Downloads/dataset2/img/t8.jpg")
#cv2.imshow('test',detect(im)[0])
#cv2.waitKey(0)
#print("Damaged Parts: ",detect(im)[1])


if __name__=="__main__":
    app.run(debug=True)