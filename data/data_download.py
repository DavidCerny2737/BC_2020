import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import urllib.request


class UrlDownloader:

    def __init__(self, url, params, folder):
        self.url = url
        self.params = params
        self.folder = folder

    def download_data(self):
        response = requests.get(self.url, params=self.params)
        soap = BeautifulSoup(response.text, 'html.parser')
        file_url = ''
        for link in soap.find_all('a', href=True):
            file_url = link.get('href')
            if file_url.endswith('.mid'):
                urllib.request.urlretrieve(self.url + file_url, self.folder + file_url + '.mid')


'''class VgMusicParser(AbstractClasses.Parser):

    def __init__(self, folder, data):
        super(folder, data)

    def parse(self):
        links = self.data.find_all('a', href=True)
        links_clean = []
        [links_clean.append(link) for link in links if "Comments" not in link]
        self.save(links_clean)

    def save(self, parsed_data):
        with(urllib.URLOpener()) as opener:
'''

downloader = UrlDownloader('https://www.vgmusic.com/music/other/miscellaneous/piano/', None,
                           'C:\\My Documents\\BC\\data\\game\\orig\\')
downloader.download_data()
