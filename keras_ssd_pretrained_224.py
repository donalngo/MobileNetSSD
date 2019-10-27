from __future__ import division
# Tensorflow Libraries
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Activation, DepthwiseConv2D, Add, ZeroPadding2D, Reshape, Concatenate,InputSpec, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.nn import relu6

import numpy as np
import matplotlib
import random
# Image Libraries
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# File Libraries
import csv
import xml.etree.ElementTree as ET
import glob

########################################################
### Bounding Boxes Utilities
########################################################

def convert_coordinates(tensor, start_index, conversion):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'corners2centroids', 'centroids2corners'.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    d = 0
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)

    if conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 2]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 1] + tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind + 2] = tensor[..., ind + 2] - tensor[..., ind] + d  # Set w
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set xmin
        tensor1[..., ind + 1] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind + 2] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set ymax
    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are 'corners2centroids', 'centroids2corners'. ")

    return tensor1


def intersection_area_(boxes1, boxes2, mode='outer_product'):
    '''
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    '''
    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    d = 0

    # Compute the intersection areas.
    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product'):
    '''
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.


    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError(
        "All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
            boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError(
        "`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.

    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1, boxes2, mode=mode)

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the union areas.
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3
    d = 0

    if mode == 'outer_product':

        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def match_bipartite_greedy(weight_matrix):
    '''
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''
    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))  # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1)  # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)  # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches


def match_multi(weight_matrix, threshold):
    '''
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    '''
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))  # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0)  # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]  # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met


class SSDInputEncoder:
    '''
    Transform ground truth labels from list of ndarrays into the shape required
    by the model i.e. (batch_size, n_boxes, #classes + 12). The anchor boxes are matched using a 
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 background_id=0):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            min_scale (float): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be >0.
            max_scale (float): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last predictor layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if `two_boxes_for_ar1` is `True`. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be greater than or equal to `min_scale`.
            aspect_ratios (list): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Note that you should set the aspect ratios such
                that the resulting anchor box shapes roughly correspond to the shapes of the objects you are trying to detect.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box.
            neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.

        '''
        variances = [1.0, 1.0, 1.0, 1.0]
        two_boxes_for_ar1 = True,
        predictor_sizes = np.array(predictor_sizes)
        coords = 'centroids'
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # Handle exceptions.
        ##################################################################################

        if aspect_ratios is None:
            raise ValueError("`aspect_ratios' cannot both be None. ")
        if (min_scale is None or max_scale is None):
            raise ValueError("`min_scale` and `max_scale` need to be specified.")

        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1  # + 1 for the background class
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)
        self.aspect_ratios = [aspect_ratios] * predictor_sizes.shape[0]
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.coords = coords
        self.background_id = background_id

        self.boxes_list = []  # Anchor boxes for each predictor layer.

        # Iterate over all predictor layers and compute the anchor boxes (feature_map_height, feature_map_width, n_boxes, 4) for each one.
        for i in range(len(self.predictor_sizes)):
            boxes = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                         aspect_ratios=self.aspect_ratios[i],
                                                         this_scale=self.scales[i],
                                                         next_scale=self.scales[i + 1])
            self.boxes_list.append(boxes)

    def __call__(self, ground_truth_labels):
        '''
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''
        # 1. Initialization
        # Column indices for the ground truth ndarray(n=2)
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4
        batch_size = len(ground_truth_labels)
        if batch_size == 0:
            print('No ground truth labels passed to encoder')
        # Generate the template for y_encoded using the anchor boxes generated in constructor
        y_encoded = self.generate_encoding_template(batch_size=batch_size)  # shape (batch_size, n_boxes, #classes + 12)
        y_encoded[:, :, self.background_id] = 1  # Initialize all boxes to background class
        n_boxes = y_encoded.shape[1]  # The total number of boxes that the model predicts per batch item
        class_vectors = np.eye(self.n_classes)  # An identity matrix used as one-hot class vectors

        # 2. Match Ground Truth boxes to anchor boxes. Boxes with IOU > pos_iou_threshold will be positive,
        # those with IOU < neg_iou_limit will be negative, while others will be neutral.

        for i in range(batch_size):
            if ground_truth_labels[i].size == 0: continue  # If there is no ground truth for this batch item, there is nothing to match.
            
            labels = ground_truth_labels[i].astype(np.float)  # The labels for this batch item. shape(k,5)

            # convert the box coordinate format
            labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids')

            classes_one_hot = class_vectors[labels[:, class_id].astype(
                np.int)]  # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]],
                                            axis=-1)  # The one-hot version of the labels for this batch item

            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12:-8], coords=self.coords,
                               mode='outer_product')

            # 2.1 For each ground truth box, get the anchor box with highest IOU even if less than pos_iou_threshold.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # 2.2 Get all matches that satisfy the IoU threshold.
            matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)
            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]
            similarities[:, matches[1]] = 0

            # 2.3 Assign 0 to background class for boxes with IOU >= neg_iou_limit
            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################
        y_encoded[:, :, [-12, -11]] -= y_encoded[:, :, [-8, -7]]  # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        y_encoded[:, :, [-12, -11]] /= y_encoded[:, :, [-6, -5]] * y_encoded[:, :, [-4,-3]]  # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[:, :, [-10, -9]] /= y_encoded[:, :, [-6, -5]]  # w(gt) / w(anchor), h(gt) / h(anchor)
        y_encoded[:, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encoded[:, :, [-2,-1]]  # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale):
        '''
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        step_height = self.img_height / feature_map_size[0]
        step_width = self.img_width / feature_map_size[1]

        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        offset_height = 0.5
        offset_width = 0.5
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # clip the coordinates to lie within the image boundaries
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= self.img_width] = self.img_width - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords
        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= self.img_height] = self.img_height - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords

        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids')

        return boxes_tensor


    def generate_encoding_template(self, batch_size):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Arguments:
            batch_size (int): The batch size.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` to match the model output.
            The last 12 values represent 4 predicted box coordinate offsets the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        return y_encoding_template


class AnchorBoxes(Layer):
    '''
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError(
                "This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(
                    K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError(
                "`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                    this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.variances = variances
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
                or `(batch, height, width, channels)` if `dim_ordering = 'tf'`. The input for this
                layer must be the output of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor

        batch_size, feature_map_height, feature_map_width, feature_map_channels = K.int_shape(x)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # clip the coordinates to lie within the image boundaries
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= self.img_width] = self.img_width - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords
        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= self.img_height] = self.img_height - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords


        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids')

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(
            boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == 'tf':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else:  # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'variances': list(self.variances),
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


########################################################
### Loss Functions
########################################################

class SSDLoss:
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio: Maximum ratio of number of negative to positive ground truth boxes to include in losss calculation
            n_neg_min: Minimum number of negative ground truth boxes to include in loss calculation
            alpha: weight for localisation loss as a fraction of classification loss
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss.

        Arguments:
            y_true (nD tensor): Ground Truth TensorFlow tensor of shape (batch_size, #boxes, 4)
                contains '(xmin, xmax, ymin, ymax)'.
            y_pred (nD tensor): Preciction TensorFlow tensor of identical structure to `y_true`
        Returns:
            The smooth L1 loss, a 2D Tensorflow tensor of shape (batch, n_boxes_total).
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A Ground Truth TensorFlow tensor of shape (batch_size, #boxes, #classes)
                contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a  Tensorflow tensor of shape (batch, n_boxes_total).
        '''
        # Make sure that  (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)  # `y_pred` should not contain any zeros before applying log func
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (nD tensor): A Ground Truth TensorFlow tensor of shape (batch_size, #boxes, #classes + 12)
                where `#boxes` is the total number of boxes that the model predicts
                contains the ground truth bounding box categories. The last axis `#classes + 12` contains
                [classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
            neg_pos_ratio: Maximum ration of number of negative to positive ground truth boxes to include in losss calculation
            n_neg_min: Minimum number of negative ground truth boxes to include in loss calculation
            alpha: weight for localisation loss as a fraction of classification loss

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        neg_pos_ratio = 3
        n_neg_min = 0
        alpha = 1.0
        neg_pos_ratio = tf.constant(neg_pos_ratio)
        n_neg_min = tf.constant(n_neg_min)
        alpha = tf.constant(alpha)

        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]  # total number of boxes per image

        # 1: Clasification and Localisation Loss for every box
        classification_loss = tf.to_float(
            self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(
            self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

        # 2: Classification losses for the positive and negative targets.

        # 2.a Masks for the positive and negative ground truth classes.
        negatives = y_true[:, :, 0]  # shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # shape (batch_size, n_boxes)

        n_positive = tf.reduce_sum(positives)  # number of positive boxes in ground truth across the whole batch

        # 2.b Positive class loss
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # shape (batch_size,)

        # 2.c Negatve class loss
        neg_class_loss_all = classification_loss * negatives  # shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                        dtype=tf.int32)
        n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.to_int32(n_positive), n_neg_min), n_neg_losses)

        # n_neg_losses will be 0 when either there are no negative ground truth boxes at all or
        # classification loss for all negative boxes is zero. Unlikely but return zero as the `neg_class_loss` in that case.
        def zero_loss():
            return tf.zeros([batch_size])

        # Otherwise compute the top-k negative loss.
        def topk_loss():
            # Reshape `neg_class_loss_all` to 1D.
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # shape (batch_size * n_boxes,)
            # Get top-k indices
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)
            # Negative Keep Mask
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))  # shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(
                tf.reshape(negatives_keep, [batch_size, n_boxes]))  # shape (batch_size, n_boxes)
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)  # shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), zero_loss, topk_loss)
        # 2.d Total class loss
        class_loss = pos_class_loss + neg_class_loss  # shape (batch_size,)

        # 3: Localisation Loss for Positive boxes
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # shape (batch_size,)

        # 4: Total Loss
        total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)
        # Keras divides the loss by the batch size but the relevant criterion to average our loss over is the number of positive boxes in the batch
        # not the batch size. So in order to revert Keras' averaging over the batch size, multiply by batch_size
        total_loss = total_loss * tf.to_float(batch_size)
        return total_loss


def build_model(image_size,
                n_classes,
                l2_reg=0.0,
                min_scale=0.2,
                max_scale=0.9,
                aspect_ratios=[1, 2, 3, 0.5, 0.33]):
    '''
    Build a Keras model with SSD architecture.

    The model consists of convolutional feature layers and a number of convolutional
    predictor layers that take their input from different feature layers.
    The model is fully convolutional.

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        l2_reg (float, optional): The L2-regularization rate. Applies to all convolutional layers.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated.
        aspect_ratios (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all predictor layers. The original implementation uses more aspect ratios
            for some predictor layers and fewer for others.

    Returns:
        model: The Keras SSD model.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''
    two_boxes_for_ar1 = True
    variances = [1.0, 1.0, 1.0, 1.0] # variance is catered but will not be used for our training
    n_predictor_layers = 6  # The number of predictor conv layers in the network
    n_classes += 1  # Account for the background class.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios is None:
        raise ValueError("`aspect_ratios' cannot both be None. ")
    if (min_scale is None or max_scale is None):
        raise ValueError("`min_scale` and `max_scale` need to be specified.")
    scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    variances = np.array(variances)

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    n_boxes = len(aspect_ratios) + 1
    aspect_ratios = [aspect_ratios] * n_predictor_layers
    n_boxes = [n_boxes] * n_predictor_layers

    ############################################################################
    # Build the network.
    ############################################################################
    # use MobileNetV2 without top as base model
    base_model = MobileNetV2(input_shape=(img_height, img_width, img_channels), include_top=False)

    #Freeze weights for base models.
    for layer in base_model.layers[:]:
    	layer.trainable=False
    
    # SSD: VGG Conv4_3 equivalent (28x28)
    conv4_3 = base_model.get_layer('block_4_add').output

    # SSD: VGG fc6 equivalent (14x14) This layer not used for SSD.
    fc6 = base_model.get_layer('block_12_add').output

    # SSD: VGG fc7 equivalent (14x14)
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)
    
    # SSD: VGG conv8_2 equivalent (7x7)
    conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(fc7)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(conv8_1)
    conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    # SSD: VGG conv9_2 equivalent (4x4)
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # SSD: VGG conv10_2 equivalent (2x2)
    conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2')(conv10_1)
    
    # SSD: VGG conv11_2 equivalent (1x1)
    conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_1')(conv10_2)
    conv11_2 = Conv2D(256, (2, 2), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_2')(conv11_1)

    # Build the convolutional predictor layers on top of the base network
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 6 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 6`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes1 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes1')(conv4_3)
    classes2 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes2')(fc7)
    classes3 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes3')(conv8_2)
    classes4 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes4')(conv9_2)
    classes5 = Conv2D(n_boxes[4] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes5')(conv10_2)
    classes6 = Conv2D(n_boxes[5] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes6')(conv11_2)

    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes1 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes1')(conv4_3)
    boxes2 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes2')(fc7)
    boxes3 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes3')(conv8_2)
    boxes4 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes4')(conv9_2)
    boxes5 = Conv2D(n_boxes[4] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes5')(conv10_2)
    boxes6 = Conv2D(n_boxes[5] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes6')(conv11_2)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                            variances=variances, name='anchors1')(boxes1)
    anchors2 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                           aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                            variances=variances, name='anchors2')(boxes2)
    anchors3 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                           aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                            variances=variances, name='anchors3')(boxes3)
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                           aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           variances=variances, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                           aspect_ratios=aspect_ratios[4],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                           variances=variances, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                           aspect_ratios=aspect_ratios[5],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=None, this_offsets=None,
                            variances=variances, name='anchors6')(boxes6)

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

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def read_csv(xml_path, filename):
    """Read CSV
    This function reads CSV File and returns a list
    # Arguments
        inputs: CSV file directory as string
    # Returns
        List of values [Img Filename, Img width, Img height, Img Chn, xmin, xmax, ymin, ymax, label]
    """
    with open(xml_path + filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def xml_to_csv(xml_directory):
    """xml to csv file converter
    converts all xml files into a single csv in same directory
    input:
            xml_directory (string) : directory of xml files for image labels
    returns:
            labels.csv at xml directory provided
            file_location(string)

    """
    xml = glob.glob(xml_directory + "*.xml")
    image_dict = []
    for image in xml:
        root = ET.parse(image).getroot()
        img_name = root.find('filename').text

        for sz in root.iter('size'):
            img_w = sz.find('width').text
            img_h = sz.find('height').text
            img_d = sz.find('depth').text

        for obj in root.iter('object'):
            for coordinate in obj.iter('bndbox'):
                x_min = coordinate.find('xmin').text
                y_min = coordinate.find('ymin').text
                x_max = coordinate.find('xmax').text
                y_max = coordinate.find('ymax').text
                label = obj.find('name').text
                image_dict.append([img_name, img_w, img_h, img_d, x_min, x_max, y_min, y_max, label])

    with open(xml_directory + "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(image_dict)

    print("Successfully created labels.csv file at xml directory.")
    file_location = "label.csv"
    return file_location


def image_augmentation(filename, data_csv=None, img_dir=None, img_sz=224, translate=0, rotate=0, scale=1, shear=0,
                       hor_flip=False,
                       ver_flip=False):
    """Image Augmentation
    This function checks for multiple classes/bounding box in an image, resizes and augments images and
    returns augmented images and bounding box co-ordinates
    # Arguments
		data_csv (List)	=  [Img Filename, Img width, Img height, Img Chn, xmin, xmax, ymin, ymax, label]
		img_dir  (Str)	= Root directory of images
		filename (Str)	= filename of image
		img_sz (int)	= dimensions to be resized
						(eg. 224 if image is (224,224))
		translate (float, optional)= percentage to translate image
									(eg. 0.2, x and y will randomly translate from -20% to 20%)
		rotate (int, optional)= angle to rotate image (0-45)
		scale (int, optional)= factor to scale image values (0-5)
								(eg. 5 will randomly scale image from 1/5 to 5)
		shear (int, optional)= shear angle of image (0-45)
		hor_flip (bool, optional)= True to allow generator to randomly flip images horizontally
		ver_flip (bool, optional) = True to allow generator to randomly flip images vertically

    #Returns
        image_aug : augmented image
        y   : labels with augmented bounding box coordinates [label, x1, y1, x2, y2]
    """
    bounding_boxes = []
    clsses = []
    labels = []
    img = cv2.imread(img_dir + filename)

    # sets variable for 90deg rotation
    if hor_flip is True and ver_flip is True:
        flip = (0, 4)
    elif hor_flip is True and ver_flip is False:
        flip = [0, 1, 3]
    elif hor_flip is False and ver_flip is True:
        flip = [4]
    else:
        flip = 0

    if data_csv is not None and img_dir is not None:
        # collect bounding boxes on the same image
        indexes = [j for j, x in enumerate(data_csv) if filename in x]

        for j in indexes:
        	#---------- NEED SOME WAY TO DYNAMICALLY HANDLE THIS -----------------------
            if int(data_csv[j][8]) == 3:
                label = 1
            elif int(data_csv[j][8]) == 4:
                label = 2
            # collect bounding boxes and classes and append them into a single list
            bounding_box = BoundingBox(x1=int(data_csv[j][4]),
                                       y1=int(data_csv[j][6]),
                                       x2=int(data_csv[j][5]),
                                       y2=int(data_csv[j][7]),
                                       label = label)

            bounding_boxes.append(bounding_box)
    try:
        bbs = BoundingBoxesOnImage(bounding_boxes, shape=img.shape)
    except:
        print ("error",filename)

    # use image aug to perform augmentation on image and bounding boxes
    seq = iaa.Sequential([
        iaa.Resize({"height": img_sz, "width": img_sz}),
        # iaa.Rot90(flip),
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))), #Gaussian Blur
        iaa.ContrastNormalization((0.75, 1.5)), # Change Contrast of Image
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5), # Add Gaussian Noise
        iaa.Multiply((0.8, 1.2), per_channel=0.2), #Make images brighter or darker based on Channel
        # iaa.Affine(
        #     translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
        #     rotate=(-rotate, rotate),
        #     scale=(1 / scale, scale),
        #     shear=(-shear, shear)
        # )
    ])
    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    image_aug = image_aug / 255
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()


    for i in range(len(bbs_aug.bounding_boxes)):
        boxes = bbs_aug.bounding_boxes[i]
        labels.append([boxes.label, boxes.x1, boxes.y1, boxes.x2, boxes.y2])

    y = np.asarray(labels)

    return image_aug, y


def image_batch_generator(img_dir, csv_data, steps_per_epoch, batch_size, label_encoder, img_sz=224, translate=0,
                          rotate=0,
                          scale=0, shear=0, hor_flip=False, ver_flip=False):
    """Batch Generator for Training Images
    Generator which returns batches of numpy array consisting of 2 numpy arrays for training

    # Arguments
        img_dir : Root directory of images
        csv_data : List [Img Filename, Img width, Img height, Img Chn, xmin, xmax, ymin, ymax, label]
        steps_per_epoch : takes either batch_size or steps_per_epoch
        batch_size : takes either steps_per_epoch or batch_size
        img_sz (int): dimensions to be resized
                (eg. 224 if image is (224,224))
        translate (float): percentage to translate image
                    (eg. 0.2, x and y will randomly translate from -20% to 20%)
        rotate (int): angle to rotate image (0-45)
        scale (int): factor to scale image values (0-5)
                (eg. 5 will randomly scale image from 1/5 to 5)
        shear (int): shear angle of image (0-45)
        hor_flip (bool): True to allow generator to randomly flip images horizontally
        ver_flip (bool) : True to allow generator to randomly flip images vertically
    #Returns
        X : [batch,h,w,c]
        Y : [batch,n_boxes,no_of_classes,12]
    """
    batch_data = []
    # get the full list of images
    images = list(set([csv_data[i][0] for i in range(len(csv_data))]))
    cur_step = 0
    # organise images into batches
    while True:
    	# Check for remaining images on the last step of the epoch, resets counter on step count on last epoch
        if cur_step == steps_per_epoch-1:
            start = cur_step * batch_size
            end = len(images)
            batch_data = images[start:end]
            cur_step = 0

        else:
            start = cur_step * batch_size
            end = cur_step * batch_size + batch_size
            batch_data = images[start:end]
            cur_step += 1

        X = []
        Y = []
        for j in batch_data:
            # do image augmentation and returns augmented image and bounding boxes
            x_image, y_boxes = image_augmentation(j,
                                                  csv_data,
                                                  img_dir,
                                                  img_sz=img_sz,
                                                  translate=translate,
                                                  rotate=rotate,
                                                  scale=scale,
                                                  shear=shear,
                                                  hor_flip=hor_flip,
                                                  ver_flip=ver_flip)
            X.append(x_image)
            Y.append(y_boxes)

        X = np.array(X)
        if len(Y) == 0:
            print(Y)
        Y = label_encoder(Y)


        yield X, Y
        # clear list after every yield
        X = []
        y_boxes = []
        Y = []

def data_generator(img_dir, xml_dir, label_encoder, batch_size=None, steps_per_epoch=None, img_sz=224,
                   translate=0, rotate=0, scale=0, shear=0, hor_flip=False, ver_flip=False):
    """Data Generator
    Generate batches of tensor image data with real-time data augmentation from image and xml directory. The data will be looped over (in batches).
    Arguments:
    --- Generator Settings ---
    img_dir(string): Set the target image directory
    xml_dir(string): Set the target xml directory
    batch_size(int, optional): Set the batch size for each generator, auto-tabulates the steps_per_epoch. Either batch_size or steps_per_epoch must be specified.
    steps_per_epoch(int, optional): Set the steps per epoch for each generator, auto-tabulates the batch_size. Either batch_size or steps_per_epoch must be specified.
    img_sz(int): set the width and height dimensions to be resized. (input size to the model)
    --- Augmentation Settings --- 
    translate(int): percentage to translate image (eg. 0.2, x and y will randomly translate from -20% to 20%)
    rotate(int):angle to rotate image (0-45)
    scale(int):factor to scale image values (0-5) (eg. 5 will randomly scale image from 1/5 to 5)
    shear(int): shear angle of image (0-45)
    hor_flip(bool): True to allow generator to randomly flip images horizontally
    ver_flip(bool): True to allow generator to randomly flip images vertically
    :return: Generators for training, validation and testing
    """
    # Exceptions
    # Check File Directory
    if glob.glob(img_dir) == [] or glob.glob(xml_dir) == []:
    	raise ValueError(
    		"Invalid image or xml directory.")

    # User must input either batch_size or steps_per_epoch
    if batch_size is None and steps_per_epoch is None:
        raise ValueError(
            "'batch_size' or `steps_per_epoch` have not been set yet. You need to pass them as arguments.")
    if batch_size is not None and steps_per_epoch is not None:
        raise ValueError("'batch_size' and `steps_per_epoch` has been set. You can only pass either one arguments.")
    # Converts directory xml files to CSV and stores in same directory
    csv_path = xml_to_csv(xml_dir)
    # Read CSV File
    data_csv = read_csv(xml_dir, "labels.csv")
    images = list(set([data_csv[i][0] for i in range(len(data_csv))]))

    # Calculate batch_size and steps_per_epoch
    if batch_size is not None and steps_per_epoch is None:
        steps_per_epoch = int((len(images) + 1) / batch_size)
    if batch_size is None and steps_per_epoch is not None:
        batch_size = int((len(images) + 1) / steps_per_epoch)


    train_batch_gen = image_batch_generator(img_dir,
                                            data_csv,
                                            steps_per_epoch,
                                            batch_size,
                                            label_encoder=label_encoder,
                                            img_sz=img_sz,
                                            translate=translate,
                                            rotate=rotate,
                                            scale=scale,
                                            shear=shear,
                                            hor_flip=hor_flip,
                                            ver_flip=ver_flip)

    total_img = len(images)

    return train_batch_gen, total_img


########################################################
### Decoder
########################################################
   
def _greedy_nms(predictions, iou_threshold=0.45, coords='corners'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords, mode='element-wise') # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)

def decode_detections(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      img_height=224,
                      img_width=224):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).
    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.
    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`


    y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
    y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
    y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
    y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
    y_pred_decoded = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners') # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded