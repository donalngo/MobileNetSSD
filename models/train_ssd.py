from keras_ssd import build_model

model = build_model((300,300,3),
                10,
                l2_regularization=0.0,
                min_scale=0.2,
                max_scale=0.9,
                aspect_ratios=[1,2,3,0.5,0.33],
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None)

model.summary()