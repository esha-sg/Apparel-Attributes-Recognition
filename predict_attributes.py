#Importing tensorflow and numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pickle
 
import tensorflow.python.platform
from tensorflow.python.platform import gfile

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

import os
import re
import sys


def getFullyConnectedLayer(no_of_categories):
    nn = Sequential()
    nn.add(Dense(256,input_dim=2048,activation = 'relu'))
    nn.add(Dropout(0.5))
    nn.add(Dense(output_dim=no_of_categories, activation='softmax'))
    nn.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    return nn

#This function is from classify_image.py which creates graph from saved GraphDef file and returns a saver
def create_graph():
  
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#Function to generate bottleneck values from classifyimage.py
def get_bottleneck_values(images):  # modifying name of function
  
  # Creates graph from saved GraphDef.
  create_graph()
  feature_vector_size = 2048 # pool_3:0 contains a float description vector of size 2048  
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0') #changing from softmax:0 to pool_3:0
    feature_vectors = np.empty([len(images),2048])
    image_names = []
    for i,image in enumerate(images): # Iterating through images 
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        feature_vector = sess.run(bottleneck_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        feature_vector = np.squeeze(feature_vector)
        image_names.append(image)
        feature_vectors[i,:] = feature_vector 
        #if(i % 10 == 0): #Print out just to see the function is processing 
         #   print("Processing image %d  %s"%(i,image))
    return feature_vectors,image_names

def predictAttributes(image):
    
    sleve_len_nn = getFullyConnectedLayer(4);
    sleve_len_nn.load_weights("sleeve_weights.h5")
    sleeve_prediction = sleve_len_nn.predict(image)
    sleeve_labels = ["Not sure, need to get some brain tweaking","Sleeveless","Half Sleeves", "Full sleeves"]
    printPredictions(sleeve_prediction, sleeve_labels)
    
    white_color_nn = getFullyConnectedLayer(3)
    #white_color_nn = trainFullyConnectedLayer(white_color_nn,feature_vectors_sorted,white_color)
    white_color_nn.load_weights('white_col_weights.h5')
    white_prediction = white_color_nn.predict(image)
    white_color_labels = ["Pretty confusing to tell if its white Sorry","Not white", "White"]
    printPredictions(white_prediction, white_color_labels)
    
    pattern_stripe_nn = getFullyConnectedLayer(3)
    #pattern_stripe_nn = trainFullyConnectedLayer(pattern_stripe_nn,feature_vectors_sorted,pattern_stripe)
    pattern_stripe_nn.load_weights('pattern_stripe_weights.h5')
    pattern_strip_prediction = pattern_stripe_nn.predict(image)
    pattern_strip_labels = ["Not sure buddy", "Striped", "No Stripes"]
    printPredictions(pattern_strip_prediction, pattern_strip_labels)
    
    pattern_graphics_nn = getFullyConnectedLayer(3)
    #pattern_graphics_nn = trainFullyConnectedLayer(pattern_graphics_nn,feature_vectors_sorted,pattern_graphics)
    pattern_graphics_nn.load_weights('pattern_graphics_weight.nn')
    pattern_graphics_prediction = pattern_graphics_nn.predict(image)
    pattern_graphics_labels = ["Not sure buddy", "No Graphics", "Ohhh Graphics"]
    printPredictions(pattern_graphics_prediction, pattern_graphics_labels)
    
    yellow_color_nn = getFullyConnectedLayer(3)
    #yellow_color_nn = trainFullyConnectedLayer(yellow_color_nn,feature_vectors_sorted,yellow_color)
    yellow_color_nn.load_weights('yellow_color.h5')
    yellow_prediction = yellow_color_nn.predict(image)
    yellow_color_labels = ["Pretty confusing to tell if its yellow Sorry","Not yellow", "Yellow yellow dirty fellow"]
    printPredictions(yellow_prediction, yellow_color_labels)
    
    skin_exp_nn = getFullyConnectedLayer(3)
    #skin_exp_nn = trainFullyConnectedLayer(skin_exp_nn,feature_vectors_sorted,skin_exposure)
    skin_exp_nn.load_weights('skin_exp_weights.h5')
    skin_ex_prediction = white_color_nn.predict(image)
    skin_ex_labels = ["I cant tell","You are covered", "Might get tanned"]
    printPredictions(skin_ex_prediction, skin_ex_labels)
    
    collor_nn = getFullyConnectedLayer(3)
    #collor_nn = trainFullyConnectedLayer(collor_nn,feature_vectors_sorted,collar)
    collor_nn.load_weights('collor_weights.h5')
    collar_prediction = white_color_nn.predict(image)
    collar_labels = ["I need more data boy","No Collar", "Collar"]
    printPredictions(collar_prediction, collar_labels)

def printPredictions(prediction, labels):
    top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
    check_match_for_No = re.match('No',labels[top_k[0]])
    s = str(check_match_for_No)
    if(top_k[0]!=0 and s == 'None'):
        tags.append(labels[top_k[0]])
    
#Tags of image
tags = []
model_dir = "imagenet"
im_name = str(sys.argv[1])
name = [im_name]
bottleneck_values, jk = get_bottleneck_values(name[0:1])
predictAttributes(bottleneck_values)
print (tags)
