import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import time

import argparse
import os

from div2k import *
# import vgg16


def PSNR(a, b, name="PSNR"):
    with tf.name_scope(name, "PSNR", [a, b]):
        mse = tf.reduce_mean(tf.squared_difference(a, b), [-3, -2, -1])
        psnr_val = tf.subtract(20 * tf.log(1.0) / tf.log(10.0), np.float32(10 / np.log(10)) * tf.log(mse), name='psnr')
        return tf.reduce_mean(psnr_val)

    return psnr


def Conv2D(input_tensor, kernel_size, stride, input_feature, output_feature, act_fn, scope):
    with tf.variable_scope(scope):
        filter = tf.get_variable("w", [kernel_size, kernel_size, input_feature, output_feature],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("b", [output_feature], initializer=tf.zeros_initializer())
        conv = act_fn(tf.nn.conv2d(input_tensor, filter, strides=[1, stride, stride, 1], padding='SAME') + conv_b)
    # conv = tf.layers.conv2d(input_tensor, output_feature, stride, kernel_initializer = tf.contrib.layers.xavier_initializer(), padding = 'SAME', activation = act_fn)
    return conv


def Conv2DTranspose(input_tensor, kernel_size, stride, input_feature, output_feature, act_fn, scope):
    with tf.variable_scope(scope):
        filter = tf.get_variable("w", [kernel_size, kernel_size, output_feature, input_feature],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("b", [output_feature], initializer=tf.zeros_initializer())
        input_tensor_shape = tf.shape(input_tensor)
        output_shape = [input_tensor_shape[0], input_tensor_shape[1] * stride, input_tensor_shape[2] * stride,
                        output_feature]
        deconv = act_fn(
            tf.nn.conv2d_transpose(input_tensor, filter, output_shape, [1, stride, stride, 1], padding='SAME') + conv_b)
    # deconv = tf.layers.conv2d_transpose(input_tensor, output_feature, [kernel_size, kernel_size], strides = [stride, stride], kernel_initializer = tf.contrib.layers.xavier_initializer(), bias_initializer = tf.constant_initializer(0.1), padding = 'SAME', activation = act_fn)
    return deconv


class DDSRCNN(object):

    def __init__(self, sess, flags):
        self.sess = sess

        # Hyperparameters
        self.epochs = flags.epochs
        self.l_rate = flags.l_rate

        # Directories
        self.checkpoint_dir = flags.checkpoint_dir
        self.saver_dir = flags.saver_dir
        # self.sample_dir = flags.sample_dir

        self.iterator = get_iterator()
        self.eval_iterator = get_eval_iterator()
        self.inference_img_name = flags.inference_img
        self.inference_iterator = get_inference_iterator(self.inference_img_name)
        self.inference_img = mpimg.imread(self.inference_img_name)[:, :, 0:3]
        self.inference_img_name = os.path.splitext(os.path.basename(self.inference_img_name))[0]
        self.output_path = flags.output_path
        
        self.kernel = 3
        self.mode = flags.mode

        # self.vgg_ground = vgg16.Vgg16()
        # self.vgg_output = vgg16.Vgg16()
        self.model()

    def model(self):
        self.is_training = tf.placeholder(tf.bool)
        # images, ground_truth = self.get_next_element()
        if (self.mode == "inference"):
            images = self.get_next_infer()
            ground_truth = tf.image.resize_images(images, (self.img_size, self.img_size), method=0, align_corners=True)
        elif (self.mode == "train"):
            images, ground_truth = tf.cond(tf.equal(self.is_training, True), lambda: self.get_next_element(),
                                           lambda: self.get_next_eval())

        _images = tf.placeholder_with_default(images, shape=[1, self.img_size / 4, self.img_size / 4, 3], name='images')
        _ground_truth = tf.placeholder_with_default(ground_truth, shape=[1, self.img_size, self.img_size, 3],
                                                    name='ground_truth')

        with tf.name_scope("Resize_Convolution"):
            input = tf.image.resize_images(images, (112, 112), method=0, align_corners=True)
            input = Conv2D(input, 3, 1, 3, 3, tf.nn.relu, "input")

        conv_1 = Conv2D(input, 3, 1, 3, 64, tf.nn.relu, "conv_1")
        conv_2 = Conv2D(conv_1, 3, 1, 64, 64, tf.nn.relu, "conv_2")
        conv_3 = Conv2D(conv_2, 3, 1, 64, 128, tf.nn.relu, "conv_3")
        conv_4 = Conv2D(conv_3, 3, 1, 128, 128, tf.nn.relu, "conv_4")
        conv_5 = Conv2D(conv_4, 3, 1, 128, 128, tf.nn.relu, "conv_5")
        conv_6 = Conv2D(conv_5, 3, 1, 128, 128, tf.nn.relu, "conv_6")
        conv_7 = Conv2D(conv_6, 3, 1, 128, 256, tf.nn.relu, "conv_7")
        conv_8 = Conv2D(conv_7, 3, 1, 256, 256, tf.nn.relu, "conv_8")
        conv_9 = Conv2D(conv_8, 3, 1, 256, 256, tf.nn.relu, "conv_9")
        conv_10 = Conv2D(conv_9, 3, 1, 256, 256, tf.nn.relu, "conv_10")
        deconv_1 = Conv2DTranspose(conv_10, 3, 1, 256, 256, tf.nn.relu, "deconv_1")
        deconv_2 = Conv2DTranspose(deconv_1, 3, 1, 256, 256, tf.nn.relu, "deconv_2") + conv_9
        deconv_3 = Conv2DTranspose(deconv_2, 3, 1, 256, 256, tf.nn.relu, "deconv_3")
        deconv_4 = Conv2DTranspose(deconv_3, 3, 1, 256, 256, tf.nn.relu, "deconv_4") + conv_7
        deconv_5 = Conv2DTranspose(deconv_4, 3, 1, 256, 128, tf.nn.relu, "deconv_5")
        deconv_6 = Conv2DTranspose(deconv_5, 3, 1, 128, 128, tf.nn.relu, "deconv_6") + conv_5
        deconv_7 = Conv2DTranspose(deconv_6, 3, 1, 128, 128, tf.nn.relu, "deconv_7")
        deconv_8 = Conv2DTranspose(deconv_7, 3, 1, 128, 128, tf.nn.relu, "deconv_8") + conv_3
        deconv_9 = Conv2DTranspose(deconv_8, 3, 1, 128, 64, tf.nn.relu, "deconv_9")
        deconv_10 = Conv2DTranspose(deconv_9, 3, 1, 64, 3, tf.identity, "deconv_10")

        output = Conv2D(deconv_10, 3, 1, 3, 3, tf.nn.sigmoid, "output")
        loss_l2 = tf.losses.mean_squared_error(ground_truth, output)
        # self.vgg_ground.build(ground_truth)
        # self.vgg_output.build(output)
        # ground_percep = self.vgg_ground.conv2_2/255.
        # output_percep = self.vgg_output.conv2_2/255.
        # loss_percep = tf.losses.mean_squared_error(ground_percep, output_percep)
        
        self.loss = loss_l2
        self.output = output
        self.images = images
        self.groundtruth = ground_truth

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.l_rate).minimize(self.loss, global_step=self.global_step)

    # Training Function
    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        Saver = tf.train.Saver()
        
        if self.mode == "inference":
            print("Restoring Model...")
            Saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            graph = tf.get_default_graph()
            print("Super-resolving image...")
            height, width, _ = self.inference_img.shape
            numWidth = width // 28
            numHeight = height // 28
            print(numWidth, numHeight)
            srimg = np.zeros([numHeight*112, numWidth*112, 3])
            print (srimg.shape)
            start_time = time.clock()
            count = 0
            for i in range(0, numHeight):
                for j in range(0, numWidth):
                    target = self.sess.run([self.output])
                    print(target[0].shape)
                    srimg[112 * i: 112 * (i+1), 112*j:112*(j+1), :] += np.squeeze(target[0])
                    count = count + 1
                    print (count)
            end_time = time.clock()
            print("Super-Resolve Complete! Time Elapsed: {}".format(end_time - start_time))
            mpimg.imsave(self.output_path + "_SR.png", srimg)
            print(srimg.shape)
            print(self.inference_img.shape)

        else:
            summary_train = tf.summary.FileWriter(self.saver_dir + '/train/', self.sess.graph)
            summary_eval = tf.summary.FileWriter(self.saver_dir + '/eval/', self.sess.graph)

            loss_summary = tf.summary.scalar("loss", self.loss)
            psnr_summary = tf.summary.scalar("PSNR", PSNR(self.output, self.groundtruth))
            recon_summary = tf.summary.image('Reconstruction', self.output, 2)
            gt_summary = tf.summary.image('Groundtruth', self.groundtruth, 2)

            train_summary_op = tf.summary.merge([loss_summary, psnr_summary])
            eval_summary_op = tf.summary.merge([loss_summary, psnr_summary, recon_summary, gt_summary])

            eval_iter = 1
            _eval = True
            self.sess.run(self.iterator.initializer)
            for iter in range(1, self.epochs):
                if (iter == self.epochs):
                    break
                while True:
                    if (_eval == True):
                        self.sess.run(self.eval_iterator.initializer)
                        for i in range(10):
                            start_time = time.time()
                            eval_loss, summary_str = self.sess.run([self.loss, eval_summary_op],
                                                                   feed_dict={self.is_training: False})
                            summary_eval.add_summary(summary_str, global_step=iter)
                            end_time = time.time()
                            print("Iterations {}/{}: \t Evaluation loss: {:.8f} \t Time Elapsed: {}".format(iter,
                                                                                                            self.epochs,
                                                                                                            eval_loss,
                                                                                                            end_time - start_time))
                            eval_iter += 1
                        _eval = False
                    else:
                        start_time = time.time()
                        _, loss, summary_str = self.sess.run([self.optimizer, self.loss, train_summary_op],
                                                             feed_dict={self.is_training: True})
                        summary_train.add_summary(summary_str,
                                                  global_step=tf.train.global_step(self.sess, self.global_step))
                        end_time = time.time()
                        print("Iterations {}/{}: \t Training loss: {:.8f} \t Time Elapsed: {}".format(iter, self.epochs,
                                                                                                      loss,
                                                                                                      end_time - start_time))

                        if (iter % 100 == 0):
                            Saver.save(self.sess, self.checkpoint_dir, global_step=self.global_step)
                            _eval = True
                            break
                        break

    def get_next_element(self):
        next_element = self.iterator.get_next()
        return next_element

    def get_next_eval(self):
        next_element = self.eval_iterator.get_next()
        return next_element

    def get_next_infer(self):
        image = self.inference_iterator.get_next()
        return image

    # Load the latest checkpoint for training
    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model.checkpoint_path)
        pass
