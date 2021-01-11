
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import math


 
#function to randomly pick an image 
# and convert it to a numpy array
def random_image2array(patology,dir_covid):

  patology=patology.upper()
  idx_set=np.random.randint(3)
  sets=['train','test','validation']
  dir_image=os.path.join(dir_covid,sets[idx_set],patology)
  idx_image=np.random.randint(len(os.listdir(dir_image)))
  random_image=os.path.join(dir_image,os.listdir(dir_image)[idx_image])

  return plt.imread(random_image)




#image counting function
def count(path):
  dir_covid19=os.path.join(path,'COVID19')
  dir_normal=os.path.join(path,'NORMAL')
  dir_pneumonia=os.path.join(path,'PNEUMONIA')

  len_covid19=len(os.listdir(dir_covid19))
  len_normal=len(os.listdir(dir_normal))
  len_pneumonia=len(os.listdir(dir_pneumonia))
  
  num_images=len_covid19+len_normal+len_pneumonia

  return num_images,len_covid19,len_normal,len_pneumonia




def create_covidResnet(base_model):

  #create top layers to customize ResNet50 
  # to a network for Covid-19 detection
  input=tf.keras.Input(shape=(256,256,3))
  preprocess_input=tf.keras.applications.resnet.preprocess_input(input)
  model_base=base_model(preprocess_input)
  global_average_layer=tf.keras.layers.GlobalAveragePooling2D()(model_base)
  drop_out_1=tf.keras.layers.Dropout(0.4)(global_average_layer)
  dense_layer=tf.keras.layers.Dense(4096,activation='relu')(drop_out_1)
  drop_out_2=tf.keras.layers.Dropout(0.4)(dense_layer)
  output_model=tf.keras.layers.Dense(3,activation='softmax')(drop_out_2)

  #create the model
  model=tf.keras.models.Model( inputs=input,outputs=output_model)

  return model




#Function to create model checkpoints
def create_checkpoint(path,checkpoints_dir,save_freq='epoch'):
  path_ck=os.path.join( checkpoints_dir,path)
  path_dir=os.path.dirname(path_ck)
  checkpoint=tf.keras.callbacks.ModelCheckpoint( filepath=path_ck,save_freq=save_freq,save_weights_only=True)
  return checkpoint,path_dir




#function to return a complete set of data from the generator 
# to be used for validation and testing metrics
def total_batch(generator):
 
  batches=math.ceil(generator.samples/generator.batch_size)
  
  for batch in range(batches):
    
    if batch==0:
      x,y=next(generator)
    else:
      x_batch,y_batch=next(generator)
      x=np.vstack((x,x_batch))
      y=np.vstack((y,y_batch))

  labels=np.argmax(y,axis=-1)
  class_indices=generator.class_indices
  return x,labels,class_indices




#Plot 'history' of the model - loss and accuracy
def plot_history(hist):
  
  metrics=hist
  metrics_list=list(metrics.keys())
  epochs=range(len(metrics[metrics_list[0]]))
  
  fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4))
  list_ax=[ax1,ax2]
  
  
  for (ind,ax) in enumerate(list_ax):
    ax.plot(epochs,metrics[metrics_list[ind]],label='train')
    ax.plot(epochs,metrics[metrics_list[ind+2]],label='val')
    ax.set_ylabel(metrics_list[ind])
    ax.set_xlabel('epochs')
    ax.legend()

    
    
    
#Plot 'metrics'- relevant metrics for 
# multiclass and unbalanced datasets
def plot_metrics(metrics):
  """ metrics: Keras custom callback"""
  metrics_list=list(metrics.keys()) 
  metrics_values=np.array(list(metrics.values())[0:2])
  epochs=range(metrics_values.shape[1])
  
  fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,4))
  list_ax=[ax1,ax2]
  for (idx,ax) in enumerate(list_ax):
    ax.plot(epochs,metrics_values[idx,:,0],label='covid19')
    ax.plot(epochs,metrics_values[idx,:,1],label='normal')
    ax.plot(epochs,metrics_values[idx,:,2],label='pneumonia')
    ax.set_ylabel(metrics_list[idx])
    ax.set_xlabel('epochs')
    ax.legend()

    
    
    
#function to plot the confusion matrix
def plot_confusion_matrix(y_test,y_pred,labels):
  cm=confusion_matrix(y_test,y_pred,labels=labels)
  plt.figure(figsize=(8,8))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix ')
  plt.ylabel('True label')
  plt.xlabel('Predictions')