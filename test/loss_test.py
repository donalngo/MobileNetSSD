import tensorflow as tf

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
        y_pred = tf.maximum(y_pred, 1e-15) # `y_pred` should not contain any zeros before applying log func
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