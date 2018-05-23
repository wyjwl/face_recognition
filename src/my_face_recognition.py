# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
import argparse
import sys
# MIT License
#
# Copyright (c) 2017 François Gervais
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
import time
import os
import cv2
import myface
import numpy as np


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(img=frame, text=face.name, org=(face_bb[0] - 10, face_bb[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))


def main(args):
    frame_interval = 5  # Number of frames after which to run face detection
    fps_display_interval = 3  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = myface.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        myface.debug = True

    while True:

        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            if embed_new_face(face_recognition):
                face_recognition.identifier = myface.Identifier()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def contain_image_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if isImage(os.path.splitext(file_path)[1]):
            return True
    return False


def embed_new_face(face_recognition, path="../data/to_be_embed"):
    if len(os.listdir(path)) > 0 and contain_image_file(path):
        print("detect new faces to be embeded")
        origin_data = np.loadtxt('../data/tw_wuhan/emb_tw_wuhan.txt')
        name_list_file = open("../data/tw_wuhan/emb_tw_wuhan_name_list.txt")
        name_list = name_list_file.readline()
        name_list = name_list[1:len(name_list) - 1]
        name_list = name_list.replace("'","")
        origin_name_list = name_list.split(", ")
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if isImage(os.path.splitext(file_path)[1]):
                img = cv2.imread(file_path)
                new_face = face_recognition.identify(img)
                if len(new_face) > 0:
                    origin_data = np.vstack((origin_data, new_face[0].embedding))
                    new_name = os.path.splitext(file_path)[0].split("/")[3]
                    origin_name_list.append(new_name)

        np.savetxt(fname="../data/tw_wuhan/emb_tw_wuhan.txt", X=origin_data)
        file = open("../data/tw_wuhan/emb_tw_wuhan_name_list.txt", 'w+')
        file.write(str(origin_name_list))
        file.close()
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            os.remove(file_path)
        print("finish embed new face and delete images")
        return True
    return False


def isImage(str):
    if str == '.jpg' or str == '.png' or str == '.jpeg':
        return True
    return False


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
