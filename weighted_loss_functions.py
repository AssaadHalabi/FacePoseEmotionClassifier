import tensorflow as tf

def weighted_sparse_categorical_crossentropy(class_weights):
    """
    Return a class weighted version of the sparse categorical crossentropy loss function.

    Arguments:
        class_weights (numpy.array): An array of each classes weight.
    
    Returns:
        loss (function): A class weighted version of the sparse categorical crossentropy loss function.
    """
    def loss(y_obs, y_pred):
        y_obs = tf.dtypes.cast(y_obs, tf.int32)
        one_hot = tf.one_hot(tf.reshape(y_obs, [-1]), depth=len(class_weights))
        weight = tf.math.multiply(class_weights.astype("float32"), one_hot)
        weight = tf.reduce_sum(weight, axis=-1)
        losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_obs,
                                                                  logits=y_pred,
                                                                  weights=weight)
        return losses
    return loss