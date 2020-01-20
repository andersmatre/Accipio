
# Copyright 2020 Anders Matre
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


import numpy as np
import PIL.Image
import accipio
import os


def get_files(path):
    """Gets files from folder.

    :param path: (str) Path to folder.
    :return: (list) Path to files.
    """
    files = []
    for _path in os.listdir(path):
        full_path = os.path.join(path, _path)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files


def split_data(data, labels, split=0.2, shuffle=True, normalize=False):
    """Splits data and labels into training and testing.

    Takes the given data and labels and splits
    them into taining data, training labels,
    test data and test labels.

    The size of the test data and labels (and in return
    the training data and labels) are based on
    the size of the split parameter; a split value of
    0.2 means that 20% of the data gets returned as
    testing and 80% gets returned as training.


    :param data: The data to be split.
    :param labels: The labels that match the data.
    :param split: (int) Value between 0-1 to represent the %
                  of data to be returned as testing (and in return
                  the training data and labels). E.g.: 0.2 = 20%.
                  It is also worth noting that the split
                  gets rounded so that the data will always
                  be divisible by the split.
    :param shuffle: (bool) If True the data will be randomly
                    shuffled to give a more balanced output when
                    later looking at testcases. The data and labels
                    will get shuffled the same way to keep the order.
    :param normalize: (bool) If True the data will be
                      divided by 255 to get values between
                      0 and 1, which is what most AI models requires.
                      This is mainly done when working with images
                      as pixel values are represented as integers
                      between 0 and 255.
    :return: Train data, train labels, test data, test labels
    """
    if len(data) == len(labels):
        if shuffle:
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            _data = data[indices]
            _labels = labels[indices]
        else:
            _data = data
            _labels = labels
        _split = round(len(data) * split)
        train_data, test_data = _data[_split:], _data[:_split]
        train_labels, test_labels = _labels[_split:], _labels[:_split]
        if normalize:
            train_data = train_data / 255
            test_data = test_data / 255
        return train_data, train_labels, test_data, test_labels
    else:
        raise ValueError(f'{accipio.utils.pcolors.WARNING}'
                         f'The length of data and labels must be the same. '
                         f'Len of data: {len(data)} and len of labels: {len(labels)}'
                         f'{accipio.utils.pcolors.ENDC}')


# TODO, add option to pass list of folders instead of top folder
def load_image_data(path, resize=None):
    """Gets all images from the given local path.

    Gets all images form the given local path and
    opens them as a numpy array and splits
    the filenames into an int category
    aswell as storing all unique class names.

    :param path: (str) Path to folder containing folders
                 of images. The names of these subfolders will be
                 used as class names to classify the data.
    :param resize: ((int, int)) If given it will resize the images
                   before returning them. This is done as (width, height)
                   in pixels.
    :return: Numpy array containing image data,
             numpy array containing integers corresponding to a class name and
             a list of unique class names. (images, index, class_names).
    """
    image_list = []
    label_list = []
    class_names = []
    shape = None

    for folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, folder)):

            impath = os.path.join(path, folder, image)
            im = PIL.Image.open(impath)

            if folder not in class_names:
                class_names.append(folder)

            if resize:
                im = accipio.images.resize_images(im, resize)[0]

            im = np.array(im)

            if not shape:
                shape = im.shape

            if im.shape == shape:
                image_list.append(im)
                label_list.append(len(class_names) - 1)
            else:
                print(f'{accipio.utils.pcolors.WARNING}'
                      f'WARNING: Skipped image with incosisten shape: {im.shape} at {image}'
                      f'{accipio.utils.pcolors.ENDC}')

    images = np.array(image_list)
    labels = np.array(label_list)
    return images, labels, class_names
