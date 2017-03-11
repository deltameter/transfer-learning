import tensorflow as tf
import math

# images: a tensor of images
# dfs: features of the dataset
# dropout_probs: probability of keeping the layers

def images_to_tensor(images, dfs):
    images = tf.reshape(images, [-1, dfs['NUM_CHANNELS'], dfs['IMAGE_SIZE'], dfs['IMAGE_SIZE']])
    images = tf.transpose(images, perm=[0, 2, 3, 1])
    return images

def load_images(images, dfs):
    images = images_to_tensor(images, dfs)

    offset = int(dfs['IMAGE_SIZE'] - dfs['DESIRED_SIZE'])

    # center crop to desired size
    images = tf.map_fn(lambda image: tf.image.crop_to_bounding_box(image, 
        offset, offset, dfs['DESIRED_SIZE'], dfs['DESIRED_SIZE']), images)

    return images

def augment_images(images, dfs):
    images = images_to_tensor(images, dfs)
    images = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), images)

    # random crop to desired size
    images = tf.map_fn(lambda image: tf.random_crop(image, [dfs['DESIRED_SIZE'], dfs['DESIRED_SIZE'], 3]), images)

    return images

def inference(images, dfs, dropout_probs):

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(input_layer, kernel_shape, dropout_prob, should_pool=True):
        # Set the standard deviation approximately to sqrt(2.0/N) (got this from Stanford notes)
        stddev = math.sqrt(1.0 / (int(kernel_shape[2]) * int(input_layer.shape[1]) ** 2))

        # shape of kernel, shape of kernel, depth of previous layer, desired depth
        shape = [kernel_shape[0], kernel_shape[1], input_layer.shape[3], kernel_shape[2]]

        weights = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable("biases", [int(shape[3])], initializer=tf.constant_initializer(0.01))

        conv = conv2d(input_layer, weights) + biases
        relu = tf.nn.relu(conv)
        dropout = tf.nn.dropout(relu, dropout_prob)

        if not should_pool:
            return dropout

        pooled = max_pool_2x2(dropout)
        return pooled

    def fc_layer(input_layer, num_next_neurons, is_output=False):
        num_prev_neurons = int(input_layer.shape[1])
        shape = [num_prev_neurons, num_next_neurons]

        weights = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable("biases", [int(shape[1])], initializer=tf.constant_initializer(0.01))

        dot = tf.matmul(input_layer, weights) + biases

        if is_output:
            return dot

        relu = tf.nn.relu(dot)
        dropout = tf.nn.dropout(relu, dropout_probs['fc'])
        return dropout


    # show images in tensorboard
    tf.summary.image('images', images, max_outputs=3)

    images = tf.nn.dropout(images, dropout_probs["input"])

    with tf.variable_scope('conv1'):
        kernel_shape = [3, 3, 64]
        conv1 = conv_layer(images, kernel_shape, dropout_probs["conv"], should_pool=False)

    with tf.variable_scope('conv2'):
        kernel_shape = [3, 3, 96]
        conv2 = conv_layer(conv1, kernel_shape, dropout_probs["conv"])

    with tf.variable_scope('conv3'):
        kernel_shape = [3, 3, 128]
        conv3 = conv_layer(conv2, kernel_shape, dropout_probs["fc"])

    with tf.variable_scope('fc1'):
        output3_flat = tf.reshape(conv3, [-1, int(conv3.shape[1]) ** 2 * 128])
        fc1 = fc_layer(output3_flat, 2048)

    with tf.variable_scope('fc2'):
        fc2 = fc_layer(fc1, 2048)

    with tf.variable_scope('softmax'):
        logits = fc_layer(fc2, dfs['NUM_LABELS'], is_output=True)

    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="xentropy")
    return tf.reduce_mean(cross_entropy, name="xentropy_mean")

def training(loss, learning_rate):
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
