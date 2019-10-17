import numpy as np
from array import array
import cv2
from math import ceil
from keras import backend as K
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
import glob

from keras_ssd import build_model
from keras_ssd import SSDLoss
from keras_ssd import datagen
from keras_ssd import SSDInputEncoder
from keras_ssd import decode_detections
from keras_ssd import xml_to_csv
from keras_ssd import test_generator
from keras_ssd import read_csv

#%% Block 1: Building Model

aspect_ratio = [0.5, 1.0, 2.0]
n_classes = 2
img_height = 300
img_width = 300

model = build_model((300,300,3),
                n_classes,
                l2_regularization=0.0,
                min_scale=0.1,
                max_scale=0.9,
                aspect_ratios=aspect_ratio,
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None)

#%% Block 2: Creating a new model 

#model.load_weights(args.weight_file, by_name=True,skip_mismatch=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
model.summary()


#%% Block 3: 

# Directories
#train_image_dir     = '/Teerapong/3. Github/1. NUS Masters/CA/MobileNetSSD/models/img_train/'
train_image_dir     = 'img_train/'
val_image_dir     = 'img_val/'



batch_size = 20

predictor_sizes = [model.get_layer('classes1').output_shape[1:3],
                   model.get_layer('classes2').output_shape[1:3],
                   model.get_layer('classes3').output_shape[1:3],
                   model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3]]

# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]]


label_encoder = SSDInputEncoder(img_height,
                                img_width,
                                n_classes,
                                predictor_sizes,
                                min_scale=0.1,
                                max_scale=0.9,
                                aspect_ratios=aspect_ratio,
                                #aspect_ratios=aspect_ratios,
                                pos_iou_threshold=0.5,
                                neg_iou_limit=0.3,
                                border_pixels='half',
                                normalize_coords=True,
                                background_id=0)

train_dataset, train_imgs = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=True, ver_flip=False)
val_dataset, val_imgs = datagen.data_generator(img_dir = val_image_dir, xml_dir = val_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=False, ver_flip=False)

print("train images : ", train_imgs)
print("validation images : ", val_imgs)

#%% Block 4
# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001
    
# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
#model_checkpoint = ModelCheckpoint(filepath='ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
#                                   monitor='val_loss',
#                                   verbose=1,
#                                   save_best_only=True,
#                                   save_weights_only=False,
#                                   mode='auto',
#                                   period=1)
#model_checkpoint.best = 

#csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
#                       separator=',',
#                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

#callbacks = [model_checkpoint,
#             csv_logger,
#             learning_rate_scheduler,
#             terminate_on_nan]

callbacks = [learning_rate_scheduler,
             terminate_on_nan]

#%% Block 5: Training

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 1
train_steps_per_epoch = ceil(train_imgs/batch_size)
val_steps_per_epoch = ceil(val_imgs/batch_size)
val_dataset_size = 10

history = model.fit_generator(generator=train_dataset,
                              validation_data=val_dataset,
                              validation_steps=val_steps_per_epoch,
                              steps_per_epoch=train_steps_per_epoch,
                              epochs=final_epoch)

#history = model.fit_generator(generator=train_dataset,
#                              steps_per_epoch=steps_per_epoch,
#                              epochs=final_epoch,
#                              callbacks=callbacks,
#                              validation_data=val_generator,
#                              validation_steps=ceil(val_dataset_size/batch_size),
#                              initial_epoch=initial_epoch)

 #%% Block 7a: Predictions on Test Image (With labels)

confidence_thresh=0.4,
iou_threshold=0.1

test_image_dir     = 'img_test/'

NumberofTest = len(glob.glob1(test_image_dir,"*.jpg"))
ImgList = glob.glob("img_test/*.jpg")
#csv_path = xml_to_csv(test_image_dir)
#data_csv = read_csv(test_image_dir, "labels.csv")
y_pred = []
test_shape =[]
prediction = []

for i in range(len(ImgList)):   
    img = cv2.imread(ImgList[i])
    CurrentShape = [img.shape[0], img.shape[1]]
    test_shape.append(CurrentShape)
    
    img = cv2.resize(img,(300,300))
    img =img.reshape(1, img.shape[0], img.shape[1],img.shape[2])
    Current_pred = model.predict(img)
    prediction.append(Current_pred)
    y_pred_decoded = decode_detections(Current_pred,
                                    confidence_thresh=confidence_thresh,
                                    iou_threshold=iou_threshold,
                                    top_k=200)
    y_pred.append(y_pred_decoded)

y_predcopy = y_pred.copy()
 
for i in range(len(y_predcopy)):
    for j in range(len(y_predcopy[i][0])):
        # x axis        
        y_predcopy[i][0][j][2] = y_predcopy[i][0][j][2]/300 * test_shape[i][0]
        y_predcopy[i][0][j][4] = y_predcopy[i][0][j][4]/300 * test_shape[i][0]

        # y axis        
        y_predcopy[i][0][j][3] = y_predcopy[i][0][j][3]/300 * test_shape[i][1]
        y_predcopy[i][0][j][5] = y_predcopy[i][0][j][5]/300 * test_shape[i][1]

#print(y_pred[0][0][0][5])  
#print(y_predcopy[0][0][0][5])
#
#print(test_shape[0][1])
#print(len(y_pred))
#print(np.shape(y_pred[0]))
#
#print(y_pred[0])
#print(y_pred[0][0][0])

#y_test = test_generator(test_image_dir, data_csv)
#
#y_pred = model.predict(y_test)

#test_dataset, test_imgs = datagen.data_generator(img_dir = test_image_dir, xml_dir = test_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
#                       translate=0, rotate=0, scale=1, shear=0, hor_flip=False, ver_flip=False)
#
#
#batch_images = next(test_dataset)
#
#y_pred = model.predict(batch_images[0])

#%% Block 9: Plotting Predictions
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['background',
            'Fruit', 'Milk']

CurrentSelect = 0
print(ImgList[i])



img = cv2.imread(ImgList[CurrentSelect])
plt.figure(figsize=(20,12))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

current_axis = plt.gca()

np.shape(y_predcopy[0])
 
#for box in batch_original_labels[i]:
#    xmin = box[1]
#    ymin = box[2]
#    xmax = box[3]
#    ymax = box[4]
#    label = '{}'.format(classes[int(box[0])])
#    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
#    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_predcopy[CurrentSelect][0]:
     xmin = box[2]
     ymin = box[3]
     xmax = box[4]
     ymax = box[5]
     color = colors[int(box[0])]
     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})


#%%
     
imgfile = 'img_test\2.jpg'
img = cv2.imread(imgfile)

#%%
np.shape(prediction[0])
prediction[0][0][0][:-4]
#%%
yy=[]
img = cv2.imread(ImgList[0])
x = [img.shape[0], img.shape[1]]
yy.append(x)
yy.append(x)
yy[1]
 

#%%
#batch_images[1].shape
y_pred.shape
 #img_height = 300
 #img_width = 300
 #n_classes = 3
 #
 #train_dataset = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=10, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
 #                       translate=0, rotate=0, scale=0, shear=0, hor_flip=True, ver_flip=False)
 #
 #predict_generator = val_dataset.generate(batch_size=1,
 #                                         shuffle=True,
 #                                         transformations=[convert_to_3_channels,
 #                                                          resize],
 #                                         label_encoder=None,
 #                                         returns={'processed_images',
 #                                                  'filenames',
 #                                                  'inverse_transform',
 #                                                  'original_images',
 #                                                  'original_labels'},
 #                                         keep_images_without_gt=False)

 #y_pred = model.predict(test_image[0])

 #%% Block 7b: Predictions on Test Image (Without labels)

 # Directories
 test_image     = '/img_test/'

 img_height = 300
 img_width = 300
 n_classes = 3



#%%
y_pred[0][3]
#%% Block 8: Decoding Predictions

y_pred_decoded = decode_detections(y_pred,
                                    confidence_thresh=0.001,
                                    iou_threshold=0.001,
                                    top_k=200)

print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[0])
 #y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

 #         if return_inverter:
 #             def inverter(labels):
 #                 labels = np.copy(labels)
 #                 labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
 #                 labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
 #                 return labels
 #
 # np.set_printoptions(precision=2, suppress=True, linewidth=90)
 # print("Predicted boxes:\n")
 # print('   class   conf xmin   ymin   xmax   ymax')
 # print(y_pred_decoded_inv[i])




