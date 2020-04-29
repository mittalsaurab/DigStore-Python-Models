# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import keras
# import json
# import pickle
from keras.applications import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

label2name = {"0" : 'alai_darwaza',"1" : 'alai_minar',"2" : 'qutub_minar',"3" : 'iron_pillar',"4" : 'jamali_kamali_tomb'}

config = tensorflow.ConfigProto(
device_count={'GPU': 1},
intra_op_parallelism_threads=1,
allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tensorflow.Session(config=config)
keras.backend.set_session(session)

model  = load_model('model.02.hdf5')
model._make_predict_function()

# model_temp = InceptionV3(weigths='imagenet')
# model_inception = Model(model_temp.input,model_temp.layers[-2].output)



def Predict(img):
	
	img = image.load_img(img,target_size=(299,299,3))
	img = image.img_to_array(img)
	img = np.expand_dims(img,axis=0)
	o = model.predict(img)[0]
	ind = np.argmax(o)
	return label2name[str(ind)]



