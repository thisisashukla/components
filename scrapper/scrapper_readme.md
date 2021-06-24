# Web Scrapper

A flexible web scrapper in Python to faciliate data mining from open web pages. The Scrapper class provides an interface using which you can write a child class, focusing only on scrapping logic while the base classes handles the parallelization, exceptions and logging.

## Sample Child Class

```python
from web_scrapper.scrapper import Scrapper

class ChildScrapper(Scrapper):
    
    def __init__(self, init_url, output_path):
        
        super(ChildScrapper, self).__init__(init_url, output_path)
        
        self.base_url = 'http://url_string'
        
        self.initialize()
    
    def init_routine(self):
        {your code}
    
    def extract_content(self, url):
        {your code}

    def get_content(self, url):
        {your code}
    
    def get_name(self, url):
        {your code}
    
    def read_content_log(self, log):
        {your code}
```

The base scrapper provides code to do the following:

- Read or create an initial log file to create a list of base URLs
- Runs the initial URLs to collect actual URLs for scrapping data
- Runs the data collection process in parallel to reduce execution time

This base classes makes sure you don't have to worry about writing data read/write and parallelization code. It is also fault tolerant, that is it maintains a log of completed queries so that if you interrupt the process, it does not have to waist time doing everything again.

# TODOs

- [ ] Add logger support for multiprocessing 
- [ ] Further reduce need of code from users
- [ ] Improve logging and robustness
- [ ] Add unit tests

# Contribution

Raise a PR with any of the above issues to contribute to the repo.
Reach out to work.ankurshukla@gmail.com or [Twitter](http://twitter.com/thisisashukla)/[Linkedin](http://linkedin.com/thisisashukla) for any query or suggestons on how can we take this forward.


