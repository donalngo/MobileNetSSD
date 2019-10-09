'''
A small 7-layer Keras model with SSD architecture. Also serves as a template to build arbitrary network architectures.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, DepthwiseConv2D, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.nn import relu6
import tensorflow.keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
#from keras_layers.keras_layer_DecodeDetections import DecodeDetections
#from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def build_model(image_size,
                n_classes,
                l2_regularization=0.0,
                min_scale=0.2,
                max_scale=0.9,
                aspect_ratios=[1,2,3,0.5,0.33],
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None):
    '''
    Build a Keras model with SSD architecture, see references.

    The model consists of convolutional feature layers and a number of convolutional
    predictor layers that take their input from different feature layers.
    The model is fully convolutional.

    The implementation found here is a smaller version of the original architecture
    used in the paper (where the base network consists of a modified VGG-16 extended
    by a few convolutional feature layers), but of course it could easily be changed to
    an arbitrarily large SSD architecture by following the general design pattern used here.
    This implementation has 7 convolutional layers and 4 convolutional predictor
    layers that take their input from layers 4, 5, 6, and 7, respectively.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Training currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all predictor layers. The original implementation uses more aspect ratios
            for some predictor layers and fewer for others. If you want to do that, too, then use the next argument instead.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each predictor layer.
            This allows you to set the aspect ratios for each predictor layer individually. If a list is passed,
            it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size,
            which is also the recommended setting.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''
    two_boxes_for_ar1 = True
    coords = 'centroids'
    variances = [1.0, 1.0, 1.0, 1.0]
    n_predictor_layers = 6 # The number of predictor conv layers in the network
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    clip_boxes=False

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios is None :
        raise ValueError("`aspect_ratios' cannot both be None. ")
    if (min_scale is None or max_scale is None):
        raise ValueError("`min_scale` and `max_scale` need to be specified.")
    scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    variances = np.array(variances)


    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.

    aspect_ratios = [aspect_ratios] * n_predictor_layers
    n_boxes = len(aspect_ratios) + 1
    n_boxes = [n_boxes] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    ############################################################################
    # Continue MobileNetV2
    ############################################################################
    def _conv_block(inputs, filters, kernel, strides):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        return Activation(relu6)(x)

    def _bottleneck(inputs, filters, kernel, t, alpha=1, s=1, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            alpha: Integer, width multiplier.
            r: Boolean, Whether to use the residuals.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        # Depth
        tchannel = K.int_shape(inputs)[channel_axis] * t
        # Width
        cchannel = int(filters * alpha)

        x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(relu6)(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def _inverted_residual_block(inputs, filters, kernel, t, alpha=1, strides=1, n=1):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            alpha: Integer, width multiplier.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

        for i in range(1, n):
            x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

        return x

    ############################################################################
    # Build the network.
    ############################################################################

    Xin = Input(shape=(img_height, img_width, img_channels))
    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(Xin)
    if not (subtract_mean is None):
        x = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x)
    if not (divide_by_stddev is None):
        x = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x)


    x = _conv_block(x, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1,  strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6,  strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6,  strides=2, n=3)
    conv1 = x
    x = _inverted_residual_block(x, 64, (3, 3), t=6,  strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6,  strides=1, n=3)
    conv2 = x
    x = _inverted_residual_block(x, 160, (3, 3), t=6,  strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6,  strides=1, n=1)

    conv3 = x

    conv4 = _inverted_residual_block(conv3, 480, (3, 3), t=6,  strides=2, n=2)
    conv5 = _inverted_residual_block(conv4, 640, (3, 3), t=6,  strides=2, n=2)
    conv6 = _inverted_residual_block(conv5, 800, (3, 3), t=6,  strides=2, n=2)


    # The next part is to add the convolutional predictor layers on top of the base network
    # that we defined above. Note that I use the term "base network" differently than the paper does.
    # To me, the base network is everything that is not convolutional predictor layers or anchor
    # box layers. In this case we'll have four predictor layers, but of course you could
    # easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
    # predictor layers on top of the base network by simply following the pattern shown here.

    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes1 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes1')(conv1)
    classes2 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes2')(conv2)
    classes3 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes3')(conv3)
    classes4 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
    classes5 = Conv2D(n_boxes[4] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
    classes6 = Conv2D(n_boxes[5] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)

    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes1 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes1')(conv1)
    boxes2 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes2')(conv2)
    boxes3 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes3')(conv3)
    boxes4 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
    boxes5 = Conv2D(n_boxes[4] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
    boxes6 = Conv2D(n_boxes[5] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)


    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    print(conv1.shape)
    print(boxes1.shape)
    anchors1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors1')(boxes1)
    anchors2 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors2')(boxes2)
    anchors3 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors3')(boxes3)
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes1_reshaped = Reshape((-1, n_classes), name='classes1_reshape')(classes1)
    classes2_reshaped = Reshape((-1, n_classes), name='classes2_reshape')(classes2)
    classes3_reshaped = Reshape((-1, n_classes), name='classes3_reshape')(classes3)
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes1_reshaped = Reshape((-1, 4), name='boxes1_reshape')(boxes1)
    boxes2_reshaped = Reshape((-1, 4), name='boxes2_reshape')(boxes2)
    boxes3_reshaped = Reshape((-1, 4), name='boxes3_reshape')(boxes3)
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors1_reshaped = Reshape((-1, 8), name='anchors1_reshape')(anchors1)
    anchors2_reshaped = Reshape((-1, 8), name='anchors2_reshape')(anchors2)
    anchors3_reshaped = Reshape((-1, 8), name='anchors3_reshape')(anchors3)
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes1_reshaped,
                                                                 classes2_reshaped,
                                                                 classes3_reshaped,
                                                                 classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes1_reshaped,
                                                             boxes2_reshaped,
                                                             boxes3_reshaped,
                                                             boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors1_reshaped,
                                                                 anchors2_reshaped,
                                                                 anchors3_reshaped,
                                                                 anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])


    model = Model(inputs=Xin, outputs=predictions)


    return model



