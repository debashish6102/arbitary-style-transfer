import os

import PIL
import numpy as np
from flask import Flask, render_template, request, Blueprint, url_for
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import random
import tensorflow as tf
from werkzeug.utils import redirect
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

views = Blueprint('views', __name__)
######################################
loaded_model = tf.keras.models.load_model("project/my_model")


def preprocess(image):
    # rgb to bgr
    image = tf.reverse(image, axis=[-1])

    return tf.keras.applications.vgg19.preprocess_input(image)


def postprocess(image):
    return tf.clip_by_value(image, 0., 255.)


def tensor_to_image(tensor):
    # tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


@views.route('/', methods=['GET', 'POST'])
def home_get():
    if request.method == 'POST':
        # imagefile_1 = request.files['style']
        imagefile_2 = request.files['content']

        # image_path_1 = imagefile_1.filename
        # image_path_2 = imagefile_1.filename
        # filename_1 = secure_filename(imagefile_1.filename)
        # imagefile_1.save(f"project\static\images\style.jpeg")

        # filename_2 = secure_filename(imagefile_2.filename)
        imagefile_2.save(f"project\static\images\content.jpeg")

        # imagefile_1.save(f"/project/static/style/{secure_filename(imagefile_1.filename)}")
        # imagefile_2.save(f"/project/static/content/{secure_filename(imagefile_2.filename)}")
        style_image = image.load_img(f"project\static\images\style.jpeg", target_size=(256, 256))
        content_image = image.load_img(f"project\static\images\content.jpeg", target_size=(256, 256))

        content_image = img_to_array(content_image)
        style_image = img_to_array(style_image)
        content_image = content_image.reshape(
            (1, content_image.shape[0], content_image.shape[1], content_image.shape[2]))
        style_image = style_image.reshape((1, style_image.shape[0], style_image.shape[1], style_image.shape[2]))

        # content_image = preprocess_input(content_image)
        # style_image = preprocess_input(style_image)

        style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
        content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)

        style_tensor = preprocess(style_tensor)
        content_tensor = preprocess(content_tensor)
        # style_tensor=tf.convert_to_tensor(style_image, dtype=tf.float32)
        # content_tensor=tf.convert_to_tensor(content_image, dtype=tf.float32)
        # s1=tf.reshape(style_tensor,[None,256,256,3])
        # c1 = tf.reshape(content_tensor, [None, 256, 256, 3])
        # data=tf.stack(style_tensor,content_tensor)
        # style_tensor=tf.expand_dims(style_tensor, 0)
        # content_tensor = tf.expand_dims(content_tensor, 0)

        output_tensor = loaded_model([content_tensor, style_tensor])
        output_tensor = postprocess(output_tensor)
        plt.switch_backend('agg')
        # a = output_tensor.numpy()
        # a = a.reshape((256, 256, 3))
        # a = tf.reverse(a, axis=[-1])
        # plt.imshow(a / 255)
        # plt.savefig('project\static\images\out.jpeg')
        image_output = tensor_to_image(output_tensor)
        image_output.save('project\static\images\out.jpeg')
        return redirect(url_for('views.choose_style'))
    return render_template('index.html')


######################################
@views.route('/choose_style', methods=['GET', 'POST'])
def choose_style():
    required_image='content'
    style_images_list = ['style', 'style1','style2','style3','style4','style5','style6',]
    if request.method == 'POST':
        imagefile_1 = request.files['style']
        # imagefile_2 = request.files['content']

        image_path_1 = imagefile_1.filename
        # image_path_2 = imagefile_1.filename
        imagefile_1.save(f"project\static\images\style.jpeg")
        # imagefile_2.save(f"project\static\images\content.jpeg")
        style_image = image.load_img(f"project\static\images\style.jpeg", target_size=(256, 256))
        content_image = image.load_img(f"project\static\images\content.jpeg", target_size=(256, 256))

        content_image = img_to_array(content_image)
        style_image = img_to_array(style_image)
        content_image = content_image.reshape(
            (1, content_image.shape[0], content_image.shape[1], content_image.shape[2]))
        style_image = style_image.reshape((1, style_image.shape[0], style_image.shape[1], style_image.shape[2]))

        style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
        content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)

        style_tensor = preprocess(style_tensor)
        content_tensor = preprocess(content_tensor)

        output_tensor = loaded_model([content_tensor, style_tensor])
        output_tensor = postprocess(output_tensor)
        plt.switch_backend('agg')
        image_output = tensor_to_image(output_tensor)
        image_output.save('project\static\images\out.jpeg')

        return render_template('stylising_image.html',required_image=required_image, style_image=style_images_list)
    return render_template('stylising_image.html',required_image=required_image, style_image=style_images_list)


@views.route('/stylise', methods=['GET', 'POST'])
def stylise():
    style_images_list = ['style', 'style1','style2','style3','style4','style5','style6',]
    selectedImage = request.args.get('selectedImage')
    style_image = image.load_img(f"project\static\images\{selectedImage}.jpeg", target_size=(256, 256))
    content_image = image.load_img(f"project\static\images\content.jpeg", target_size=(256, 256))

    content_image = img_to_array(content_image)
    style_image = img_to_array(style_image)
    content_image = content_image.reshape(
        (1, content_image.shape[0], content_image.shape[1], content_image.shape[2]))
    style_image = style_image.reshape((1, style_image.shape[0], style_image.shape[1], style_image.shape[2]))

    style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
    content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)

    style_tensor = preprocess(style_tensor)
    content_tensor = preprocess(content_tensor)

    output_tensor = loaded_model([content_tensor, style_tensor])
    output_tensor = postprocess(output_tensor)
    plt.switch_backend('agg')
    image_output = tensor_to_image(output_tensor)
    image_output.save('project\static\images\out.jpeg')
    required_image='out'
    return render_template('stylising_image.html',required_image=required_image,style_image=style_images_list)
