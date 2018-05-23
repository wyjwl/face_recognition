# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import align.detect_face
import cv2
import facenet
import numpy as np
import tensorflow as tf
from keras.models import load_model
from scipy import misc

gpu_memory_fraction = 1
facenet_model_checkpoint = "/Users/yjwang/Desktop/ML/facenet/src/models/20180402-114759"
classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)

        return faces


def process_data(wuhan_data, embedding):
    data = np.zeros(shape=(len(wuhan_data), 512))
    for i in range(0, len(wuhan_data)):
        data[i] = np.square(np.subtract(wuhan_data[i], embedding))
    return data


def getMaxIndex(predict):
    maxIndex = 0
    for i in range(1, len(predict)):
        if predict[i] > predict[maxIndex]:
            maxIndex = i
    return maxIndex


def getSecondMaxIndex(predict, max_index):
    second_max_index = 0
    for i in range(1, len(predict)):
        if predict[i] > predict[second_max_index] and second_max_index != max_index:
            second_max_index = i
    return second_max_index


def getDif(data_after_process):
    dif = np.zeros(shape=(len(data_after_process), 1))
    for i in range(0, len(data_after_process)):
        dif[i] = 0
        for j in range(0, 512):
            dif[i] = dif[i] + data_after_process[i][j]
    return dif


def findMinIndex(data):
    minIndex = 0
    for i in range(1, len(data)):
        if data[i] < data[minIndex]:
            minIndex = i
    return minIndex


class Identifier:
    def __init__(self):
        self.model = load_model('my_model.h5')
        self.wuhan_data = np.loadtxt('../data/tw_wuhan/emb_tw_wuhan.txt')
        name_list_file = open("../data/tw_wuhan/emb_tw_wuhan_name_list.txt")
        name_list = name_list_file.readline()
        self.name_list = name_list[1:len(name_list) - 1].split(",")

    def identify(self, face):
        if face.embedding is not None:
            data_after_process = process_data(self.wuhan_data, face.embedding)
            predict = self.model.predict(data_after_process)
            max_index = getMaxIndex(predict)
            if predict[max_index] <= 0.8:
                return "unknown " + str(predict[max_index])
            return self.name_list[max_index] + " " + str(predict[max_index])


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces
