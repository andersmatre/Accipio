
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


import matplotlib.pyplot as plt
import numpy as np


def image_grid(predictions, test_images, test_labels, class_names, cols=4, rows=4, correct_color='blue', wrong_color='red'):
    """Creates a figure showing AI predictions.

    Creates a grid of images and
    labels each image with the corresponding
    prediciton, prediction certainty and
    the actual answer. The label is formatted
    as 'prediction certainty% -- (answer). The
    grid also contains a header with the format
    'correct predictions / total predictions'.

    It is also possible to specify the size
    of the grid by passing in the cols and rows
    paramaters (default is 4x4). If there are too
    few test_images to fill up the grid it will
    shrink to the largest possible size using those images.

    :param predictions: (list) Predictions returned by TensorFlow's
                        model.predict().
    :param test_images: (list) Images that was tested using TensorFlow's
                        model.predict().
    :param test_labels: (list) Corresponding labels to the test_images.
    :param class_names: (list) Strings that correspond with the test_labels.
    :param cols: (int) Number of coulmns in the grid.
    :param rows: (int) Number of rows in the grid.
    :param correct_color: (str) Color of the label when it's correct.
    :param wrong_color: (str) Color of the label when it's wrong.
    :return: None.
    """
    num_cols = min(round(len(test_images) / cols), cols)
    num_rows = min(round(len(test_images) / rows), rows)
    total = num_cols * num_rows
    correct = total
    plt.figure(figsize=(8, 8))

    for i in range(1, total + 1):
        prediction = class_names[np.argmax(predictions[i])]
        answer = class_names[test_labels[i]]
        color = correct_color
        if prediction != answer:
            color = wrong_color
            correct -= 1
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(test_images[i])
        plt.text(x=0.0, y=-0.1, s=f'{prediction} {round(int(100*np.max(predictions[i])))}% -- ({answer})', transform=plt.gca().transAxes, color=color)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f'{correct} / {total} predicted correctly')
    plt.show()
