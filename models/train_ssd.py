from math import ceil
from keras import backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
import glob

from keras_ssd import build_model
from keras_ssd import SSDLoss
from keras_ssd import datagen
from keras_ssd import SSDInputEncoder

aspect_ratios = [0.5, 1.0, 2.0]
model = build_model((300,300,3),
                2,
                l2_regularization=0.0,
                min_scale=0.2,
                max_scale=0.9,
                aspect_ratios=aspect_ratios,
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
a = '/Teerapong/3. Github/1. NUS Masters/CA1 MobileNet SSD/MobileNetSSD/models/img_train/*'

#%% Block 3: 

# Directories
#train_image_dir     = '/Teerapong/3. Github/1. NUS Masters/CA/MobileNetSSD/models/img_train/'
train_image_dir     = 'img_train/'
val_image_dir     = 'img_val/'

img_height = 300
img_width = 300
n_classes = 3
batch_size = 5

predictor_sizes = [model.get_layer('classes1').output_shape[1:3],
                   model.get_layer('classes2').output_shape[1:3],
                   model.get_layer('classes3').output_shape[1:3],
                   model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3]]

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]


label_encoder = SSDInputEncoder(img_height,
                                img_width,
                                n_classes,
                                predictor_sizes,
                                min_scale=0.1,
                                max_scale=0.9,
                                aspect_ratios=aspect_ratios,
                                pos_iou_threshold=0.5,
                                neg_iou_limit=0.3,
                                border_pixels='half',
                                normalize_coords=True,
                                background_id=0)

train_dataset = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=True, ver_flip=False)
val_dataset = datagen.data_generator(img_dir = val_image_dir, xml_dir = val_image_dir, batch_size=batch_size, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=False, ver_flip=False)


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
final_epoch     = 120
steps_per_epoch = 1000
val_dataset_size = 10

history = model.fit_generator(generator=train_dataset,
                              validation_data=val_dataset,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch)

#history = model.fit_generator(generator=train_dataset,
#                              steps_per_epoch=steps_per_epoch,
#                              epochs=final_epoch,
#                              callbacks=callbacks,
#                              validation_data=val_generator,
#                              validation_steps=ceil(val_dataset_size/batch_size),
#                              initial_epoch=initial_epoch)

#%%
for l in range(len(model.layers)):
    print(l)
    print(model.layers[l])
    print(model.layers[l].name)
    print(model.layers[l].output)

#%% Block 7a: Predictions on Test Image (With labels)

# Directories
test_image     = 'will get from his generator'

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

y_pred = model.predict(test_image[0])

#%% Block 7b: Predictions on Test Image (Without labels)

# Directories
test_image     = '/img_test/'

img_height = 300
img_width = 300
n_classes = 3

#%% Block 8: Decoding Predictions
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

#y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
                return labels

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])




#%% Block 9: Plotting Predictions
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})