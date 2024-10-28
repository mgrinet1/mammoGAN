import albumentations as A
import segmentation_models as sm
import pandas as pd
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers
from keras_vision_transformer import utils
import distutils.dir_util
import json
import os
import shutil
import sys
from zipfile import ZipFile

import requests


print(f'Start Training for {model_type} {IMG_HEIGHT}x{IMG_WIDTH}')
print(f'Tensorflow version: {tf.__version__}')

DATA_DIR = "./dataset/mammoSEG_dataset"
BATCH_SIZE = 8
CLASSES = [['mass'], ['mass', 'breast'], ['mass', 'fibroglandular', 'adipose']]
LR = 0.0001
EPOCHS = 40
        
IMG_WIDTH_ARRAY = [128, 128, 256, 256, 512, 512, 1024, 1024]
IMG_HEIGHT_ARRAY = [128, 256, 256, 512, 512, 1024, 1024, 2048]

one_hot = False
custom_loss = False
models_list = ['UNet', 'DeepLabV3', 'Attention_UNet', 'ResUNet', 'FCN_8', 'DeepLabV3_scratch', 'SwimT'] 

N_CLASSES = [1,2,3]

for N_CLASS in N_CLASSES:
  for model_type in models_list:
    for IMG_WIDTH, IMG_HEIGHT in zip(IMG_WIDTH_ARRAY, IMG_HEIGHT_ARRAY):
        import tensorflow as tf

        
        print(f'Start Training for {model_type} {IMG_HEIGHT}x{IMG_WIDTH}')
        print(f'Tensorflow version: {tf.__version__}')
        
        if one_hot:
            model_type = 'one_hot_'+model_type
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            
        if custom_loss:
            model_type = 'custom_loss_'+model_type
            dice_loss = sm.losses.DiceLoss(class_weights=np.array([1.0, 10.0, 3.0, 2.0])) 
            focal_loss = sm.losses.BinaryFocalLoss() if NUM_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
            loss = dice_loss + (1 * focal_loss)

        #IMAGE_SIZE = 512
        BATCH_SIZE = 8#4
        NUM_CLASSES = N_CLASS
        DATA_DIR = "./dataset/mammoSEG_dataset"
        NUM_TRAIN_IMAGES = 1131
        NUM_VAL_IMAGES = 100
        EPOCHS = 100
        train_images = sorted(glob(os.path.join(DATA_DIR, "Declutter_Mass-Training/*")))[:NUM_TRAIN_IMAGES]
        train_masks = sorted(glob(os.path.join(DATA_DIR, "Declutter_Category_ids_3_classes/*")))[:NUM_TRAIN_IMAGES]
        val_images = sorted(glob(os.path.join(DATA_DIR, "Declutter_Mass-Training/*")))[
            NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
        ]
        val_masks = sorted(glob(os.path.join(DATA_DIR, "Declutter_Category_ids_3_classes/*")))[
            NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
        ]
        test_images = sorted(glob(os.path.join(DATA_DIR, "Declutter_Mass-Test/*")))
        
        test_masks = sorted(glob(os.path.join(DATA_DIR, "Declutter_Test_Category_ids_3_classes/*")))
        
        def resize(input_image, real_image, height, width):
          input_image = tf.image.resize(input_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          real_image = tf.image.resize(real_image, [height, width],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
          return input_image, real_image

        
        #
       

        train_dataset = data_generator(train_images, train_masks)
        val_dataset = data_generator(val_images, val_masks)
        test_dataset = test_data_generator(test_images)

        print("Train Dataset:", train_dataset)
        print("Val Dataset:", val_dataset)
        print("Test Dataset:", test_dataset)
        
        

        if (model_type == 'DeepLabV3')|(model_type == 'one_hot_DeepLabV3'):
            model = DeeplabV3Plus(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES, weights=True)
        if (model_type=='DeepLabV3_scratch')|(model_type == 'one_hot_DeepLabV3_scratch'):
            model = DeeplabV3Plus(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES, weights=False)
        if model_type == 'UNet':
            #model = UNet_Generator(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES)
            model = model = sm.Unet('vgg16', classes=NUM_CLASSES) #, activation='softmax'
        if (model_type == 'ResUNet')|(model_type == 'one_hot_ResUNet'):
            model = ResUNet_Generator(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES)
        if (model_type == 'Attention_UNet')|(model_type == 'one_hot_Attention_UNet'):
            model = Attention_UNet_Generator(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES)
        if model_type == 'FCN_8':
            model = FCN_8_Generator(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES)
        if model_type =='SwimT':
            model = SwimT_Generator(image_size=(IMG_HEIGHT, IMG_WIDTH), num_classes=NUM_CLASSES)    
        
        
        
        colormap = loadmat("./dataset/human_colormap.mat")["colormap"]
        colormap = colormap * 100
        colormap = colormap.astype(np.uint8)
        
        
               
                
        figfolder = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}' 
        if not os.path.exists(figfolder):
            os.makedirs(figfolder)
            
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=300, to_file=f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/segmentation_model_{IMG_HEIGHT}x{IMG_WIDTH}.png')
        log_dir=f"../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir, save_weights_only=False, monitor='loss', mode='min', save_best_only=True)

        figfolder = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/results_figures' 
        if not os.path.exists(figfolder):
            os.makedirs(figfolder)
    
        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/real_source'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)
    
        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/real_target'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)
    
        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_mask'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)  

        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_overlay'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)  

        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_colormap'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)       
        
        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/saved_segmentation_model'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)
            
        testoutput = f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/saved_segmentation_model'
        if not os.path.exists(testoutput):
            os.makedirs(testoutput)
        
        initial_learning_rate = 3e-4#0.001
        #Nth step * (NUM_TRAIN_IMAGES / BATCH_SIZE)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=20*(NUM_TRAIN_IMAGES /BATCH_SIZE),decay_rate=0.96,staircase=True)
        
        
        #dice_loss = sm.losses.DiceLoss(class_weights=np.array([1.0, 10.0, 3.0, 2.0])) 
        #focal_loss = sm.losses.BinaryFocalLoss() if NUM_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
        #total_loss = dice_loss + (1 * focal_loss)
        
        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
        
        metrics = ["accuracy", sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        
        
        #, tf.keras.metrics.MeanIoU(name='IoU', num_classes=NUM_CLASSES)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001), #lr_schedule
            loss=loss,
            metrics=metrics
        )
        
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        #####################
        #      Classes      #
        # 1 - mass          #
        # 2 - dense         #
        # 3 - fatty         #
        # 0 - background    #
        #####################
        class_weights = {0: 1.0,
                         1: 5.0,
                         2: 3.0,
                         3: 2.0}
        
        
        if one_hot:
            history = model.fit(train_dataset,
                                validation_data=val_dataset,
                                epochs=EPOCHS,
                                callbacks=[model_checkpoint_callback, CustomCallback(), keras.callbacks.ReduceLROnPlateau()])
        else:
            history = model.fit(train_dataset.map(add_sample_weights), 
                                validation_data=val_dataset.map(add_sample_weights),
                                epochs=EPOCHS,
                                callbacks=[model_checkpoint_callback, CustomCallback(), keras.callbacks.ReduceLROnPlateau()])#
         
        plt.plot(history.history["loss"])
        plt.title("Training Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.show()
        
        plt.plot(history.history["iou_score"])
        plt.title("Training IoU")
        plt.ylabel("IoU")
        plt.xlabel("epoch")
        plt.show()        

        plt.plot(history.history["accuracy"])
        plt.title("Training Accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.show()

        plt.plot(history.history["val_loss"])
        plt.title("Validation Loss")
        plt.ylabel("val_loss")
        plt.xlabel("epoch")
        plt.show()
        
        plt.plot(history.history["val_iou_score"])
        plt.title("Validation IoU")
        plt.ylabel("val_IoU")
        plt.xlabel("epoch")
        plt.show()    

        plt.plot(history.history["val_accuracy"])
        plt.title("Validation Accuracy")
        plt.ylabel("val_accuracy")
        plt.xlabel("epoch")
        plt.show()
        
        pd.DataFrame(history.history).to_csv(f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/history.csv')
        
        
        checkpoint = tf.train.Checkpoint(model)
        # Restore the checkpointed values to the `model` object.
        checkpoint.restore(tf.train.latest_checkpoint(log_dir))       

        plot_predictions(train_images[:2], colormap, model=model)
        plot_predictions(val_images[:2], colormap, model=model)
        plot_test_results(test_images, colormap, model=model)
            
        model.save(f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/saved_segmentation_model/{model_type}_segmentation_model_{IMG_HEIGHT}x{IMG_WIDTH}')
        model.summary()
def start_training_process():
    print('Training process started...')

if __name__ == "__main__":
    if len(sys.argv) == 0:
        raise Exception("Input Params: Model Type, Number of Classes, Image Width, Image Height")
    print(f"Modle Type: {sys.argv[0]}, Number of Classes: {sys.argv[1]}")
    start_training_process()