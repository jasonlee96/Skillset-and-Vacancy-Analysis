import requests
from bs4 import BeautifulSoup
from model import Job
import time


def is_captcha(soup):
    # return soup.find("div", id="cf-wrapper") is not None
    return soup.find("title").get_text() == "Sorry, unable to access page..." or soup.find("div", id="cf-wrapper") is not None


class Extraction:
    def __init__(self, url, header, key=None, location=None):
        # Request Session for sending request to HTML
        # Until certain limit of requests the Captcha will be pop out, so new Session is required when Captcha pop out
        self.req = requests.Session()
        self.url = url
        self.header = header
        self.payload = {'key': '', 'location': '', 'pg': ''}
        self.list_data = []
        if key is not None:
            self.payload['key'] = key
        if location is not None:
            self.payload['location'] = location
        print(self.payload)

    def set_key(self, key):
        self.payload['key'] = key

    def set_page(self, page):
        self.payload['pg'] = page

    def set_location(self, location):
        self.payload['location'] = location

    def get_html(self):
        html = self.req.get("https://www.jobstreet.com.my/en/job-search/job-vacancy.php",
                            headers=self.header, params=self.payload)
        soup = BeautifulSoup(html.text, 'html.parser')
        return soup

    def get_job_titles(self):
        for i in range(1, 10):
            print("current page: ", i, sep=" ")
            self.set_page(i)
            soup = self.get_html()
            links = soup.find_all('a', {'class': 'position-title-link'})
            for link in links:
                if link.get('href') == 'https://www.jobstreet.com.my/en/job/1':
                    continue
                job = Job(title=link.get('data-job-title'), href=link.get('href'), key=self.payload['key'])
                self.get_job_description(job, self.header)
                self.list_data.append(job)
        return self.list_data

    def get_job_description(self, job, header):
        # get html response by request and parse it into bs4 format
        r = self.req.get(job.href, headers=header)
        s = BeautifulSoup(r.text, 'html.parser')
        # If "are you robot?" popped out then we have to reset session
        while is_captcha(s):
            self.req = requests.Session()
            print("reset session ")
            r = self.req.get(job.href, headers=header)
            s = BeautifulSoup(r.text, 'html.parser')

        # get relevant job description
        description = s.find('div', {'id': 'job_description'})

        # eliminate the HTML tag to retrieve inner text by placing whitespace between the tags
        job.add_description(description.get_text(" ").replace("\n", "").strip())


