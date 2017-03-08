import tensorflow as tf


def inference(images, dfs, dropout_probs):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv_2d_large_stride(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    images = tf.reshape(images, [-1, dfs['NUM_CHANNELS'], dfs['IMAGE_SIZE'], dfs['IMAGE_SIZE']])
    images = tf.transpose(images, perm=[0, 2, 3, 1])

    # show images in tensorboard
    tf.summary.image('images', images, max_outputs=1)

    images = tf.nn.dropout(images, dropout_probs["input"])

    with tf.name_scope('hidden1'):
        weights = weight_variable([3, 3, dfs['NUM_CHANNELS'], 64])
        biases = bias_variable([64])
        
        hidden_conv_1 = tf.nn.relu(conv2d(images, weights) + biases)
        hidden_conv_1 = tf.nn.dropout(hidden_conv_1, dropout_probs["conv"])

    with tf.name_scope('hidden2'):
        weights = weight_variable([3, 3, 64, 96])
        biases = bias_variable([96])
        
        hidden_conv_2 = tf.nn.relu(conv2d(hidden_conv_1, weights) + biases)
        hidden_conv_2 = tf.nn.dropout(hidden_conv_2, dropout_probs["conv"])
        hidden_pooled_2 = max_pool_2x2(hidden_conv_2)

    with tf.name_scope('hidden3'):
        weights = weight_variable([3, 3, 96, 128])
        biases = bias_variable([128])
        
        hidden_conv_3 = tf.nn.relu(conv2d(hidden_pooled_2, weights) + biases)
        hidden_conv_3 = tf.nn.dropout(hidden_conv_3, dropout_probs["conv"])

    with tf.name_scope('hidden4'):
        weights = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        
        hidden_conv_4 = tf.nn.relu(conv_2d_large_stride(hidden_conv_3, weights) + biases)
        hidden_conv_4 = tf.nn.dropout(hidden_conv_4, dropout_probs["fc"])
        hidden_pool_4 = max_pool_2x2(hidden_conv_4)

    with tf.name_scope('fc1'):
        hidden_pool_4_flat = tf.reshape(hidden_pool_4, [-1, (int(dfs['IMAGE_SIZE'] / 8)) ** 2 * 128])
        weights = weight_variable([int((dfs['IMAGE_SIZE'] / 8)) ** 2 * 128, 1024])
        biases = bias_variable([1024])

        hidden_fc_1 = tf.nn.relu(tf.matmul(hidden_pool_4_flat, weights) + biases)
        hidden_fc_1 = tf.nn.dropout(hidden_fc_1, dropout_probs["fc"])

    with tf.name_scope('fc1'):
        weights = weight_variable([1024, 512])
        biases = bias_variable([512])

        hidden_fc_2 = tf.nn.relu(tf.matmul(hidden_fc_1, weights) + biases)
        hidden_fc_2 = tf.nn.dropout(hidden_fc_2, dropout_probs["fc"])

    with tf.name_scope('softmax'):
        weights = weight_variable([512, dfs['NUM_LABELS']])
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