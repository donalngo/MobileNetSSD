import numpy as np
from array import array
import cv2
from math import ceil
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
import glob

from keras_ssd_pretrained_224 import build_model
from keras_ssd_pretrained_224 import SSDLoss
from keras_ssd_pretrained_224 import datagen
from keras_ssd_pretrained_224 import SSDInputEncoder
from keras_ssd_pretrained_224 import decode_detections
from keras_ssd_pretrained_224 import xml_to_csv
from keras_ssd_pretrained_224 import read_csv
from keras_ssd_pretrained_224 import image_augmentation
#%%
#from tensorflow.keras.applications import MobileNetV2
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Model
#Xin = Input(shape=(224, 224, 3))
#base_model = MobileNetV2()(Xin)
#model = Model(inputs=Xin, outputs=base_model)
#
#
#model.summary()
#%% Block 1: Building Model

loadweights = True
weightsfile = "pretrain_ssd224.h5"


aspect_ratio = [1, 2, 3, 0.5, 0.33]
n_classes = 2
img_height = 224
img_width = 224
img_channels = 3
min_scale = 0.1
max_scale = 0.9
normalize_coords = False

model = build_model((img_height,img_width,img_channels),
                n_classes,
                l2_reg=0.0,
                min_scale=min_scale,
                max_scale=max_scale,
                aspect_ratios=aspect_ratio,
                normalize_coords=normalize_coords)

#%% Block 2: Creating a new model

if loadweights is False:
    model.load_weights(weightsfile)
    print("weights Loaded")

#model.load_weights(args.weight_file, by_name=True,skip_mismatch=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
model.summary()


#%% Block 3: Generator for Training

# Directories
train_image_dir = 'data/img_train/'
val_image_dir = 'data/img_val/'
batch_size = 20

predictor_sizes = [model.get_layer('classes1').output_shape[1:3],
                   model.get_layer('classes2').output_shape[1:3],
                   model.get_layer('classes3').output_shape[1:3],
                   model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3]]


label_encoder = SSDInputEncoder(img_height,
                                img_width,
                                n_classes,
                                predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                aspect_ratios=aspect_ratio,
                                pos_iou_threshold=0.5,
                                neg_iou_limit=0.3,
                                normalize_coords=normalize_coords,
                                background_id=0)

train_dataset, train_imgs = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=224, label_encoder=label_encoder,
                                                   translate=0, rotate=0, scale=1, shear=0, hor_flip=True, ver_flip=False)
#train_dataset, train_imgs = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=224, label_encoder=label_encoder,
                                                   # translate=0, rotate=0, scale=1, shear=0, hor_flip=False,
                                                   # ver_flip=False)
val_dataset, val_imgs = datagen.data_generator(img_dir = val_image_dir, xml_dir = val_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=224, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=False, ver_flip=False)

print("train images : ", train_imgs)
print("validation images : ", val_imgs)

#%% Block 4
# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.0005
    elif epoch < 100:
        return 0.0005
    elif epoch < 200:
        return 0.0004
    elif epoch < 300:
        return 0.0004
    elif epoch < 500:
        return 0.0003
    elif epoch < 600:
        return 0.0002
    elif epoch < 800:
        return 0.0001
    else:
        return 0.00005
    
# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='pretrain_ssd224.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best = 

csv_logger = CSVLogger(filename='pretrain_ssd224+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler]
             #terminate_on_nan]

#callbacks = [learning_rate_scheduler,
#             terminate_on_nan]

#%% Block 5: Training

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 1000
train_steps_per_epoch = ceil(train_imgs/batch_size)
val_steps_per_epoch = ceil(val_imgs/batch_size)
val_dataset_size = 10

history = model.fit_generator(generator=train_dataset,
                              validation_data=val_dataset,
                              validation_steps=val_steps_per_epoch,
                              steps_per_epoch=train_steps_per_epoch,
                              callbacks = callbacks,
                              epochs=final_epoch)
# #%%
# for layer in model.layers:
#     print(layer, layer.trainable)
#
# #%% Block 6a: Predictions on Test Image (With labels)
#
# confidence_thresh=0.2,
# iou_threshold=0.5
#
# test_image_dir     = 'img_test/'
# ImgList = glob.glob("img_test/*.jpg")
#
# csv_path = xml_to_csv(test_image_dir)
# data_csv = read_csv(test_image_dir, "labels.csv")
#
#
# y_pred = []
# test_shape =[]
# prediction = []
# total_loss = 0
#
# for i in range(len(ImgList)):
# #for i in range(2):
#
#     # Getting Ground Truth Y Label
#     image_aug, y_truth = image_augmentation(ImgList[i][9:], img_dir='img_test/', data_csv=data_csv)
#     y_truth = y_truth.reshape(1, y_truth.shape[0], y_truth.shape[1])
#     y_truth = label_encoder(y_truth)
#
#     # Getting Predicted Y
#     img = cv2.imread(ImgList[i])
#     CurrentShape = [img.shape[0], img.shape[1]]
#     test_shape.append(CurrentShape)
#
#     img = cv2.resize(img,(224,224))
#     img =img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
#     Current_pred = model.predict(img)
#     prediction.append(Current_pred)
#     y_pred_decoded = decode_detections(Current_pred,
#                                     confidence_thresh=confidence_thresh,
#                                     iou_threshold=iou_threshold,
#                                     top_k=200)
#     y_pred.append(y_pred_decoded)
#
#     #Computing Loss
#     total_loss = tf.add(ssd_loss.compute_loss(y_truth, Current_pred), total_loss)
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print('The total loss for the entire test set is %.2f' % total_loss.eval())
#
# print("To see the actual img and the bounding box of the test set, please proceed to the next section.")
#
# # Converting Prediction to actual Image Size
# for i in range(len(y_pred)):
#     for j in range(len(y_pred[i][0])):
#         # x axis with img rows
#         y_pred[i][0][j][2] = y_pred[i][0][j][2]/img_width * test_shape[i][0]
#         y_pred[i][0][j][4] = y_pred[i][0][j][4]/img_width * test_shape[i][0]
#
#         # y axis with img cols
#         y_pred[i][0][j][3] = y_pred[i][0][j][3]/img_height * test_shape[i][1]
#         y_pred[i][0][j][5] = y_pred[i][0][j][5]/img_height * test_shape[i][1]
#
# #%% Block 7a: Plotting Predictions
# # Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
# classes = ['background',
#             'Fruit', 'Milk']
#
# # Select the test image to display
# CurrentSelect = 0
#
# img = cv2.imread(ImgList[CurrentSelect])
# plt.figure(figsize=(20,12))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
# current_axis = plt.gca()
#
# np.shape(y_predcopy[0])
#
# for box in y_predcopy[CurrentSelect][0]:
#      xmin = box[2]
#      ymin = box[3]
#      xmax = box[4]
#      ymax = box[5]
#      color = colors[int(box[0])]
#      label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#      current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#      current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
#




