import tensorflow as tf
import transfer_model
from dataset import Dataset, ToyDataset
import argparse
import shutil
import numpy as np

def train():
	dataset = Dataset()
	test_images, test_labels = dataset.get_test_data()

	with tf.Graph().as_default():
		images_placeholder = tf.placeholder(tf.float32, shape=(None, 
			dataset.FEATURES['IMAGE_SIZE'] ** 2 * dataset.FEATURES['NUM_CHANNELS']))

		labels_placeholder = tf.placeholder(tf.int64, shape=(None))

		input_drop = tf.placeholder(tf.float32)
		conv_drop = tf.placeholder(tf.float32)
		fc_drop = tf.placeholder(tf.float32)

		dropout_probs = {
			'input': input_drop,
			'conv': conv_drop,
			'fc': fc_drop
		}

		logits = transfer_model.inference(images_placeholder, dataset.FEATURES, dropout_probs)

		loss = transfer_model.loss(logits, labels_placeholder)

		train = transfer_model.training(loss, FLAGS.learning_rate)

		evaluation = transfer_model.evaluation(logits, labels_placeholder)

		summary = tf.summary.merge_all()

		writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())

		init = tf.global_variables_initializer()

		saver = tf.train.Saver()

		sess = tf.Session()

		sess.run(init)

		for step in range(FLAGS.steps):
			images, labels = dataset.get_minibatch(FLAGS.batch_size)

			if FLAGS.skip_dropout:
				feed_dict =  {
					images_placeholder: images, 
					labels_placeholder: labels,
					input_drop: 1.0,
					conv_drop: 1.0,
					fc_drop: 1.0
				}
			else:
				feed_dict = { 
					images_placeholder: images, 
					labels_placeholder: labels,
					input_drop: FLAGS.input_dropout,
					conv_drop: FLAGS.conv_dropout,
					fc_drop: FLAGS.fc_dropout
				}

			_, loss_val = sess.run([train, loss], feed_dict=feed_dict)

			if step % 100 == 0:
				print('Step {0} and loss {1}'.format(step, loss_val))
				summary_str = sess.run(summary, feed_dict=feed_dict)
				writer.add_summary(summary_str, step)
				writer.flush()

			if step % 1000 == 0:
				num_correct = 0

				for start in range(0, len(test_labels), FLAGS.batch_size):
					end = start + FLAGS.batch_size
					num_correct += sess.run(evaluation, 
						feed_dict = { 
						images_placeholder: test_images[start:end], 
						labels_placeholder: test_labels[start:end],
						input_drop: 1.0,
						conv_drop: 1.0,
						fc_drop: 1.0
						})

				val_accuracy_sum = tf.Summary(value=[
				    tf.Summary.Value(tag="summary_tag", simple_value=num_correct/len(test_images)), 
				])

				writer.add_summary(val_accuracy_sum, step)
				writer.flush()

def main(_):
	if FLAGS.wipe_logs:
		shutil.rmtree('./logs')

	train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.0001,
		help='Initial learning rate.'
	)

	parser.add_argument(
		'--batch_size',
		type=int,
		default=100,
		help='Batch size.'
	)

	parser.add_argument(
		'--steps',
		type=int,
		default=10000,
		help='How many minibatch SGDs to run.'
	)

	parser.add_argument(
		'--input_dropout',
		type=float,
		default=0.95,
		help='Dropout probability for input layer'
	)

	parser.add_argument(
		'--conv_dropout',
		type=float,
		default=0.85,
		help='Dropout probability for convolutional layers'
	)

	parser.add_argument(
		'--fc_dropout',
		type=float,
		default=0.6,
		help='Dropout probability for fully connected layers'
	)

	parser.add_argument(
		'--skip_dropout',
		default=False,
		action='store_true',
		help='Use dropout or not'
	)

	parser.add_argument(
		'--log_dir',
		type=str,
		default='./logs',
		help='Dropout probability for input layer'
	)

	parser.add_argument(
		'--wipe_logs',
		default=False,
		action='store_true',
		help='Delete all logs.'
	)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=unparsed)