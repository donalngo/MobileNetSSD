from keras_ssd import build_model
from keras_ssd import SSDLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from keras_loss_function.keras_ssd_loss import SSDLoss

model = build_model((300,300,3),
                10,
                l2_regularization=0.0,
                min_scale=0.2,
                max_scale=0.9,
                aspect_ratios=[1,2,3,0.5,0.33],
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
model.summary()