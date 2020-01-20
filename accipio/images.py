
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


import json
import os
import requests
import io
import PIL.Image
import accipio


def resize_images(images, resize):
    """Resizes images.

    Resizes a list of paths to images or
    a byte type object of an image.

    :param images: (list) Images to resize.
    :param resize: ((int, int)) Desired size (width, height) in pixels.
    :return: (list) Images after resizing.
    """
    image_list = []
    if type(images) == list:
        for img in images:
            open_img = PIL.Image.open(img)
            open_img = open_img.resize(resize, PIL.Image.ANTIALIAS)
            image_list.append(open_img)
            open_img.save(img)
    else:
        image = images.resize(resize, PIL.Image.ANTIALIAS)
        image_list.append(image)
    return image_list


def save_google_images(keywords, path=f'{os.getcwd()}\\images\\', count=100, resize=None):
    """Collects and saves images from www.google.com.

    Gets images based on keywords from www.google.com.
    Google only loads 100 image to begin with and therefor
    this function can't save more than that at a time.
    If given a count it will limit the number of images to save.

    :param keywords: (list) Words to search. Each word
                     in the list will get its own search.
    :param path: (str) Path to output folder.
    :param count: (int) Number of images to search for before returning.
    :param resize: ((int, int)) If given it will resize the images before
                 saving them (width, height) in pixels.
    :return: None.
    """
    print(f'{accipio.utils.pcolors.OKGREEN}'
          f'Starting to save images, this may take a couple of minutes.'
          f'{accipio.utils.pcolors.ENDC}')

    images_total = 0
    for search in keywords:

        _path = f'{path}\\{search}'
        if not os.path.exists(_path):
            os.makedirs(_path)

        url = accipio.utils.get_url(search)
        soup = accipio.utils.get_soup(url)

        images = []
        soup_total = 0
        for a in soup.find_all('div', {'class': 'rg_meta'}):
            if count > soup_total:
                link, type = json.loads(a.text)['ou'], json.loads(a.text)['ity']
                images.append((link, type))
                soup_total += 1

        for i, img in enumerate(images):
            if img[1] == 'jpg':
                file_name = f'{search}_{i}.jpg'
                try:
                    image = PIL.Image.open(io.BytesIO(requests.get(img[0]).content))
                except (PIL.UnidentifiedImageError, requests.exceptions.RequestException):
                    continue
                if resize:
                    image = resize_images(image, resize)
                    image[0].save(f'{_path}\\{file_name}')
                    images_total += 1
                else:
                    image.save(f'{_path}\\{file_name}')
                    images_total += 1

    print(f'{accipio.utils.pcolors.OKGREEN}'
          f'Process finished, successfully saved {images_total} images.'
          f'{accipio.utils.pcolors.ENDC}')
