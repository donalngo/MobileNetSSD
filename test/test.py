from keras_ssd import datagen
from keras_ssd import SSDInputEncoder
train_image_dir ="MnF/"

predictor_sz = [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)]
label_encoder = SSDInputEncoder(300,300,3,predictor_sz)

train_dataset = datagen.data_generator(img_dir = train_image_dir, xml_dir = train_image_dir, batch_size=10, steps_per_epoch=None, img_sz=300, label_encoder=label_encoder,
                       translate=0, rotate=0, scale=1, shear=0, hor_flip=True, ver_flip=False)

x,y = next(train_dataset)

print(x.shape , y.shape)