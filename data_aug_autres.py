import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from skimage import data, filters
import cv2
from tensorflow.keras import layers


def visualize(original, augmented):  # visualiser l'image et son image augmentée
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented, cmap='gray')
    plt.show()

# image = tf.io.decode_raw('CDStent.raw')
# print(tf.shape(image))


im = np.fromfile('CDStent.raw', dtype=np.uint16)
im = im.reshape(1024, 1024, -1)

# plt.imshow(img,cmap='gray')
# plt.show()

img = Image.open('fleurs_grises.jpg')
print(img)
# im.show()

# ci-dessous, les différentes augentaions que l'on peut effectuer
adjusted_brightness = tf.image.adjust_brightness(im, delta=0.7)
visualize(im, adjusted_brightness)

adjusted_r_brightness = tf.image.random_brightness(im, 0.4, seed=42)
visualize(im, adjusted_r_brightness)

flipped_lr = tf.image.flip_left_right(im)  # left-right
visualize(im, flipped_lr)

random_contrasted = tf.image.random_contrast(im, 0.2, 0.5, seed=42)
visualize(im, random_contrasted)

flipped_ud = tf.image.flip_up_down(im)  # up-down
visualize(im, flipped_ud)

# saturated = tf.image.adjust_saturation(im, 3)  #pas pertinent pour des images grises
# visualize(im, saturated)

cropped = tf.image.central_crop(im, central_fraction=0.5)
visualize(im, cropped)

rotated = tf.image.rot90(im)
visualize(im, rotated)


# numpy_image = np.asarray(image_chat2)
# image_chat = 'chat_mignon.jpg'
# plt.imshow(image_chat)
# plt.imshow(numpy_image)

# coins = data.coins()
# plt.imshow(coins, cmap='gray')
# plt.show()

# definir le IMG_SIZE
# resize_and_rescale = tf.keras.Sequential([
#   layers.Resizing(IMG_SIZE, IMG_SIZE),
#   layers.Rescaling(1./255)
# ])

# resize_and_rescale(image)

# def resize_rescale(image, IMG_SIZE):
#     resize_and_rescale = tf.keras.Sequential([
#   layers.Resizing(IMG_SIZE, IMG_SIZE),
#   layers.Rescaling(1./255)])
#     result = resize_and_rescale(image)
#     plt.imshow(result)
