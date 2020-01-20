
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


from bs4 import BeautifulSoup
import urllib.request as urllib_req


class pcolors:
    """Colorcodes to color console prints."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_url(keyword):
    """Gets the Google image search URL with the given keyword.

    :param keyword: (str) String to use as search parameter
                    when getting str.
    :return: (str) URL to google images with given keyword correctly formated.
    """
    _keyword = keyword.replace(' ', '%20').replace(',', '')
    image_url = f'https://google.com/search?q={_keyword}&tbm=isch'
    return image_url


def get_soup(url):
    """Gets the soup of the given URL.

    :param url: (str) URL the get the soup from.
    :return: Soup of given URL.
    """
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'}
    return BeautifulSoup(urllib_req.urlopen(urllib_req.Request(url, headers=header)), 'html.parser')
