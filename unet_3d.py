{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "unet_3d.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "metadata": {
        "id": "loT_7foe9t6m",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout, ReLU, Cropping3D, Dropout\n",
        "from keras.models import Model\n",
        "from keras.optimizers import Adam\n",
        "from keras import regularizers \n",
        "from keras.layers.normalization import BatchNormalization as bn"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "9NGq6rR28m29",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def conv_3d(input_tensor, n_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1), activation = \"relu\", padding = \"valid\", batch_norm = True, dropout = True):\n",
        "  \n",
        "  conv = Conv3D(n_filters, kernel_size, padding = padding, strides = strides)(input_tensor)\n",
        "  \n",
        "  if batch_norm:\n",
        "    conv = bn()(conv)\n",
        "    \n",
        "  if activation.lower() == \"relu\":\n",
        "    conv = ReLU()(conv)\n",
        "  \n",
        "  if dropout:\n",
        "    conv = Dropout(0.3)(conv)\n",
        "    \n",
        "  return conv"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "AAFN9JwJuI-k",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def upconv_and_concat(tensor_to_upconv, tensor_to_concat, upconv_n_filters, kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = \"valid\"):\n",
        "  upconv = Conv3DTranspose(upconv_n_filters, kernel_size, strides = strides, padding = padding)(tensor_to_upconv)\n",
        "  \n",
        "  crop_size = (int(tensor_to_concat.shape[1]) - int(tensor_to_upconv.shape[1])*2) // 2\n",
        "  cropped = Cropping3D((crop_size, crop_size, crop_size))(tensor_to_concat)\n",
        "  concat = concatenate([upconv, cropped], axis = 4)\n",
        "  \n",
        "  return concat"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Wo9qIla5-d97",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def unet_3d(input_shape, n_classes, loss, metrics, n_gpus = 1, optimizer = \"adam\", lr = 0.0001, batch_norm = True, activation = \"relu\", pool_size = (2, 2, 2)):\n",
        "  \n",
        "  # Encoder\n",
        "  input = Input(input_shape)\n",
        "  conv1_1 = conv_3d(input, 32, batch_norm = batch_norm, activation = activation)\n",
        "  conv1_2 = conv_3d(conv1_1, 32, batch_norm = batch_norm, activation = activation)\n",
        "  pool_1 = MaxPooling3D(pool_size)(conv1_2)\n",
        "  \n",
        "\n",
        "  conv2_1 = conv_3d(pool_1, 64, batch_norm = batch_norm, activation = activation)\n",
        "  conv2_2 = conv_3d(conv2_1, 64, batch_norm = batch_norm, activation = activation)\n",
        "  pool_2 = MaxPooling3D(pool_size)(conv2_2)\n",
        "  \n",
        "  conv3_1 = conv_3d(pool_2, 128, batch_norm = batch_norm, activation = activation)\n",
        "  conv3_2 = conv_3d(conv3_1, 128, batch_norm = batch_norm, activation = activation)\n",
        "  pool_3 = MaxPooling3D(pool_size)(conv3_2)\n",
        "  \n",
        "  conv4_1 = conv_3d(pool_3, 256, batch_norm = batch_norm, activation = activation)\n",
        "  conv4_2 = conv_3d(conv4_1, 128, batch_norm = batch_norm, activation = activation)\n",
        "  \n",
        "  \n",
        "  # Decoder\n",
        "  upconv_5 = upconv_and_concat(conv4_2, conv3_2, 128)\n",
        "  conv5_1 = conv_3d(upconv_5, 128, batch_norm = batch_norm, activation = activation)\n",
        "  conv5_2 = conv_3d(conv5_1, 64, batch_norm = batch_norm, activation = activation)\n",
        "  \n",
        "  upconv_6 = upconv_and_concat(conv5_2, conv2_2, 64)\n",
        "  conv6_1 = conv_3d(upconv_6, 64, batch_norm = batch_norm, activation = activation)\n",
        "  conv6_2 = conv_3d(conv6_1, 32, batch_norm = batch_norm, activation = activation)\n",
        "  \n",
        "  upconv_7 = upconv_and_concat(conv6_2, conv1_2, 32)\n",
        "  conv7_1 = conv_3d(upconv_7, 32, batch_norm = batch_norm, activation = activation)\n",
        "  conv7_2 = conv_3d(conv7_1, 32, batch_norm = batch_norm, activation = activation)\n",
        "  \n",
        "  final_conv = Conv3D(n_classes, kernel_size = (1, 1, 1), padding = \"same\")(conv7_2)\n",
        "  \n",
        "  \n",
        "  model = Model(input, final_conv)\n",
        "  \n",
        "  if optimizer == \"adam\":\n",
        "    adam = Adam(lr = lr)\n",
        "    \n",
        "    model.compile(optimizer = adam, loss = loss, metrics = metrics)\n",
        "  \n",
        "  else:\n",
        "    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)\n",
        "  \n",
        "  return model"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}