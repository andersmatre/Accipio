
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


from tensorflow import keras


def image_classification(train_images, train_labels, class_names,
                         epochs=40, batch_size=300):
    """Simple image classification model

    Creates, compiles and trains
    a simple image classification model using
    the Keras module from TensorFlow.

    :param train_images: (list) Images to train with.
    :param train_labels: (list) Corresponding labels to the images.
    :param class_names: (list) String names of the train_labels.
                        Used to determine the output layer size,
                        e.g.: len(class_names).
    :param epochs: (int) Hyperparameter that determines the number
                   of times the algorithm will work through the entire
                   training dataset. One epoch means that each sample
                   in the training dataset has had an oppurtunity to
                   update the internal model parameters.
    :param batch_size: (int) Hyperparameter that determines the number
                       of training examples used each iteration.
                       What batch size to use largely depends on the
                       data you are passing to the AI and may require
                       some tweaking.
    :return: Trained keras model.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=train_images[0].shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels,
              epochs=epochs, batch_size=batch_size)

    return model
