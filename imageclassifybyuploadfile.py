#!/usr/bin/env python
# encoding: utf-8

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys


import numpy as np

import tensorflow as tf

import os

from flask import Flask, render_template, request,  url_for, redirect
from werkzeug import secure_filename
UPLOAD_FOLDER = '/home/pi/Public'     #服务器上的任意目录 '/var/www/uploads'

ALLOWED_EXTENSIONS = set(['bmp', 'jpg', 'png', 'jpeg', 'gif'])




FLAGS = None

MODEL_DIR = '/home/pi/Public'
IMAGE_FILE = 'image.jpg'
TOP_PREDICTIONS = 5


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return '''<H1>Hello, this is a image classifying tool for you.
              <p> just upload a image, i can tell you what are in this image 
              <p>
              <a href="./imageclassify">Go</a> 
           '''
    

@app.route('/calsquare')
def response_request():
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  a = tf.placeholder(tf.int32, shape=(), name="input")
  asquare = tf.multiply(a, a, name="output")
  sess = tf.Session()
  num = request.args.get('num')
  for i in range (100):
      ret = sess.run([asquare], feed_dict={a: num})  
  return str(ret)
    




@app.route('/imageclassify', methods=['GET'])
def imageclassify_form():
	return '''
	<H2>
		<p> <H2>Please click the button below and select a image file, then upload, then wait a moment
		<p> <H2>only support JPG, JPEG, BMP, GIF and PNG files
		<p>
	  <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" style="width: 100px; height: 30px; font-size: 20px;" />
        <p>
        <input type="submit" value="Start..." style="width: 100px; height: 30px; font-size: 20px;" />
    </form>	
    '''



@app.route('/upload', methods=['POST'])
def imageclassify():
	f = request.files['file']
	if f and allowed_file(f.filename):
		
		filename = f.filename #secure_filename(f)
		
		print(filename)
		
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		fextension=os.path.splitext(filename)
		os.rename('/home/pi/Public/'+filename,'/home/pi/Public/demo'+fextension[1])		
		return run_inference_on_image('/home/pi/Public/demo'+fextension[1])
	else:
		return '<H1>The image file is incorrect, please check and try again'




@app.route('/image')
def image():
	return run_inference_on_image(IMAGE_FILE)

    
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)


  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-TOP_PREDICTIONS:][::-1]
    #top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    retval=''
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      retval='<p><H3>'+retval + human_string + ':  '+ str(score)+'<br>'
    return retval



def main(_):

  run_inference_on_image(IMAGE_FILE)


#if __name__ == '__main__':
#  tf.app.run()
  
#if __name__ == "__main__":
#    server = wsgi.WSGIServer(('0.0.0.0', 5000), app)
#    server.serve_forever()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
