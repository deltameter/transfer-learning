import tensorflow as tf

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
    images =  tf.map_fn(lambda image: tf.random_crop(image, [dfs['DESIRED_SIZE'], dfs['DESIRED_SIZE'], 3]), images)

    return images

def inference(images, dfs, dropout_probs):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_3x3_overlap(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    # show images in tensorboard
    tf.summary.image('images', images, max_outputs=3)

    images = tf.nn.dropout(images, dropout_probs["input"])

    with tf.name_scope('conv1'):
        weights = weight_variable([3, 3, dfs['NUM_CHANNELS'], 64])
        biases = bias_variable([64])
        
        hidden_conv_1 = tf.nn.relu(conv2d(images, weights) + biases)
        hidden_conv_1 = tf.nn.dropout(hidden_conv_1, dropout_probs["conv"])

    with tf.name_scope('conv2'):
        weights = weight_variable([3, 3, 64, 96])
        biases = bias_variable([96])
        
        hidden_conv_2 = tf.nn.relu(conv2d(hidden_conv_1, weights) + biases)
        hidden_conv_2 = tf.nn.dropout(hidden_conv_2, dropout_probs["conv"])
        hidden_pooled_2 = max_pool_2x2(hidden_conv_2)

    with tf.name_scope('conv3'):
        weights = weight_variable([5, 5, 96, 128])
        biases = bias_variable([128])
        
        hidden_conv_3 = tf.nn.relu(conv2d(hidden_pooled_2, weights) + biases)
        hidden_conv_3 = tf.nn.dropout(hidden_conv_3, dropout_probs["fc"])
        hidden_pooled_3 = max_pool_2x2(hidden_conv_3)

    with tf.name_scope('fc1'):
        hidden_pooled_3_flat = tf.reshape(hidden_pooled_3, [-1, int(hidden_pooled_3.shape[1]) ** 2 * 128])
        weights = weight_variable([int(hidden_pooled_3.shape[1]) ** 2 * 128, 2048])
        biases = bias_variable([2048])

        hidden_fc_1 = tf.nn.relu(tf.matmul(hidden_pooled_3_flat, weights) + biases)
        hidden_fc_1 = tf.nn.dropout(hidden_fc_1, dropout_probs["fc"])

    with tf.name_scope('fc2'):
        weights = weight_variable([2048, 2048])
        biases = bias_variable([2048])

        hidden_fc_2 = tf.nn.relu(tf.matmul(hidden_fc_1, weights) + biases)
        hidden_fc_2 = tf.nn.dropout(hidden_fc_2, dropout_probs["fc"])

    with tf.name_scope('softmax'):
        weights = weight_variable([2048, dfs['NUM_LABELS']])
        biases = bias_variable([dfs['NUM_LABELS']])

        logits = tf.matmul(hidden_fc_2, weights) + biases

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
