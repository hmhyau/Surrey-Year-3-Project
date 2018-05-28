import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', default = "train")
parser.add_argument('-gpu', '--gpu', default = "0")
parser.add_argument('-i', '--iterations', default = 100000)
parser.add_argument('-dir', '--dir', type = str)
parser.add_argument('-inference', '--inference_img', type=str)
parser.add_argument('-o', '--output', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
#from Model import DDSRCNN
from Model_div2k import DDSRCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("epochs", args.iterations, "Number of epochs")
flags.DEFINE_integer("img_size", 112, "Size of image used [112]")
flags.DEFINE_float("l_rate", 1e-4, "Learning rate of the network [1e-4]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint_" + args.dir + '/', "Checkpoint directory")
flags.DEFINE_string("saver_dir", "./saver_" + args.dir + '/', "Saver directory")
flags.DEFINE_string("mode", args.mode, "Mode to invoke")
flags.DEFINE_string("inference_img", args.inference_img, "Path to inference image")
flags.DEFINE_string("output_path", args.output, "Path to store inference image")
FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def main(argv):
	if args.mode is "train":
		if not os.path.exists(FLAGS.checkpoint_dir):
			os.makedirs(FLAGS.checkpoint_dir)
		if not os.path.exists(FLAGS.saver_dir):
			os.makedirs(FLAGS.saver_dir)
			
	with tf.Session(config=config) as sess:
		Model = DDSRCNN(sess, FLAGS)
		Model.train()
		# print(Model.layers)

if __name__ == "__main__":
	tf.app.run()
