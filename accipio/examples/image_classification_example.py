
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
"""Example on how to use the image_classification().

This script gives you an idea on how you could use
the accipio module to classifiy images from
a local folder.
"""


import accipio


if __name__ == '__main__':

    # Load images from a local folder
    x, y, class_names = accipio.data.load_image_data(path='images', resize=(64, 64))

    # Split the images into training and testing
    train_images, train_labels, test_images, test_labels = accipio.data.split_data(x, y, split=0.2, shuffle=True, normalize=True)

    # Create and train the model
    model = accipio.ai.image_classification(train_images, train_labels, class_names, epochs=10, batch_size=10)

    # Predict the test images
    predictions = model.predict(test_images)

    # Give some example predictions
    accipio.visuals.image_grid(predictions, test_images, test_labels, class_names, cols=5, rows=5)
