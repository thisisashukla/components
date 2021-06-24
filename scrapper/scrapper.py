import sys
import logging
import requests
import json, time
import pandas as pd
from tqdm import tqdm
import os, re, string, ast
import multiprocessing as mp
from datetime import datetime
from bs4 import BeautifulSoup
from functools import partial
from abc import ABCMeta, abstractmethod
from multiprocessing_logging import install_mp_handler

class Scrapper(metaclass=ABCMeta):
    
    """
    Instantiates an object of Scrapper class which does the heavy lifting to 
    scrap data from a collection webpage based on logic defined in the child class.

    Attributes
    ----------
        init_url : str
            the url used to collect links for the target pages
        log_line : int
            integer maintaning index of the current log line
        output_path : str
            string containing path where downloaded data is to be stored.
        log_path : str
            path to store the log file
    """

    def __init__(self, init_url, output_path = './/Data'):

        """
        constructor
        
        Parameters
        ----------
            init_url : str
                the url used to collect links for the target pages
            output_path : str
                string containing path where downloaded data is to be stored.
        """
        
        self.init_url = init_url
        self.log_line = 1
        self.output_path = output_path
        self.log_path = os.path.join(self.output_path, 'logs')
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            os.mkdir(self.log_path)
            print('Created log folder')
        else:
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            print('Log folder already exists')
        
        logging.basicConfig(filename=f"{self.log_path}/log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
                            level=logging.DEBUG)
    
    def log(self, message, verbose = 1):
        
        """
        function to log messages to a log file
        
        Parameters
        ----------
            message : str
                message to be logged
            verbose : int
                prints the message on console if verbose = 1 else not
        """
        
        message = f"{self.log_line} - {message}"
        try:
            if verbose:
                print(message)
            logging.debug(message)
            self.log_line+=1
        except Exception as e:
            print(f"Exception {e} occurred while logging.")
        

    def page_bs(self, url):

        """
        function to get the beautiful soup representation of a page
        
        Parameters
        ----------
            url : str
                url of a page
        
        Returns
        -------
            soup : BeautifulSoup
                beautiful soup object for the page
        """
        
        page = requests.get(url)
        self.log(f"{url} returned status code {page.status_code}")
        
        if page.status_code == 406:
            page = requests.get(url, headers = {"User-Agent": "XY"})
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

    @abstractmethod
    def init_routine(self):

        """
        Abstract method to be implemented by the child class to create the 
        initial file required for crawling the data.
        """

        pass

    def initialize(self,  override = False):

        """
        Initializes the process of data scrapping. Creates a folder with the 
        given name and creates folders for content and Log.This method also 
        creates or reads the file required to crawl a website (index of content). 
        The contents are read from cache if it is available. 
        Cache can be overriden by using the override argument.

        Parameters
        ----------
            override : bool
                boolean denoting whether to over write cached init_log file or not

        """
        
        try:
            if override:
                raise Exception('Override.')
                
            with open(os.path.join(self.output_path, 'init_url_log.txt'), 'r') as file:
                self.init_content = file.readlines()
                self.init_content = [re.sub('\n', '', line) for line in self.init_content]
                if len(self.init_content) == 0:
                    raise Exception('Empty File Found.')
                self.log('Using init cache.')
                
        except Exception as e:
            self.log(f"Exception {e} occured while fetching from webpage.")
            self.init_content = self.init_routine()
            with open(os.path.join(self.output_path, 'init_url_log.txt'), 'w') as file:
                if isinstance(self.init_content, dict):
                    file.write(str(self.init_content))
                elif isinstance(self.init_content, list):
                    self.init_content = list(set([c for c in self.init_content if len(c) > 0]))
                    file.write('\n'.join(self.init_content))

    @abstractmethod
    def extract_content(self, url):

        """
        Abstract method to be implemented by child class. Used to extract relevant content from the webpage.

        Parameters
        ----------
            url : str
                a string containing the URL of the webpage

        """

        pass

    def get_content(self, url):

        """
        Method to call and handle exceptions in the extractContent method

        Parameters
        ----------
            url : str
                a string containing the URL of the webpage

        """

        success = True
        comment = 'NE'
        try:
            content = self.extract_content(url)
        except Exception as e:
            self.log(f"Exception {e} occured while extracting content from {url}")
            content = ''
            success = False
            comment = e
            
        return content, success, re.sub(',', '', str(comment))

    def write_file(self, file_path, content):

        """
        Writes the given string to a file at the given file path.

        Parameters
        ----------
            file_path : str
                a string containing the path of the file
            content : str
                content to be written to the file

        """
        
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            self.log(f"Successfully written at {file_path}", 0)
        except Exception as e:
            self.log(f"Exception {e} occurred while writing at {file_path}", 0)
        

    @abstractmethod
    def get_name(self, url):

        """
        Abstract method to be implemented by child class. 
        Used to get name of the file to be saved from a URL.

        Parameters
        ----------
            url : str
                a string containing the URL of the webpage
        
        Returns
        -------
            name : str
                string containing name of the file for the url as derived from
                the get_name function
            success : bool
                boolean denoting whether the url fetch was a success or not
            comment : str
                empty string if the url fetch was successful else it contains
                the exception which caused it to fail.
        """

        pass

    def save_content(self, url):

        """
        Save content of a webpage based on getContent function

        Parameters
        ----------
            url : str
                string containing page URL
                
        Returns
        -------
            name : str
                string containing name of the file for the url as derived from
                the get_name function
            success : bool
                boolean denoting whether the url fetch was a success or not
            comment : str
                empty string if the url fetch was successful else it contains
                the exception which caused it to fail.
        """
        
        name = self.get_name(url)
        content, success, comment = self.get_content(url)
        
        self.write_file(os.path.join(self.output_path, f"{name}.txt"), content)
        time.sleep(0.1)

        return name, success, comment
    
    @abstractmethod
    def read_content_log(self, log):

        """
        Abstract method to be implemented by child class. Used to read the content log.

        Parameters
        ----------
            file : open
                file containing the log.
        """

        pass

    def execute_process(self, queries = None, parallelize = True):

        """
        Executes the process of crawling the website for different queries.
        Stores the log in a log file and results as files at the give output path

        Parameters
        ----------
            queries : list
                list of queries. Queries can be different combination of URL 
                parameters to parse different pages.
            parallelize : bool
                boolean for parallel execution
        """
        
        if queries is None:
            queries = self.init_content
        
        self.log(f"{len(queries)} current queries.")
        
        init_log_path = f"{self.log_path}//init_log.txt"
        file_mode = 'a'

        if os.path.exists(init_log_path):
            self.log('Found existing log.')
            try:
                log = pd.read_csv(init_log_path, header = 0, names = ['datetime', 'filename', 'status', 'comment'])
            except Exception as e:
                self.log(f"pandas failed reading file due {e}. Reading using open")
                ff = open(init_log_path, 'r').readlines()
                ff = [re.sub('\n', '', line).strip() for line in ff]
                lines = [line.split(',') for line in ff]
                log = pd.DataFrame(lines, columns = ['datetime', 'filename', 'status', 'comment'])
            log['datetime'] = pd.to_datetime(log['datetime'], format = "%Y%m%d-%H%M%S")
            log['status'] = log['status'].str.strip()
            log = log.sort_values(['filename', 'datetime', 'status'])
            completed = self.read_content_log(log)
            self.log(f"Found {len(completed)} completed queries.")
        else:
            file_mode = 'w'
            completed = set([])
        
        queries = set([q for q in queries if len(q) > 0])
        queries_pending = set(sorted(queries - completed))
        self.log(f"{len(queries_pending)} out of {len(queries)} pending. Now crawling.")
        
        success_count = 0
        if parallelize:
            install_mp_handler()
            pool = mp.Pool(8) 
            with open(init_log_path, file_mode) as file:
                for name, success, comment in tqdm(pool.imap_unordered(self.save_content, queries_pending), total = len(queries_pending)):
                    file.write('{}, {}, {}, {} \n'.format(time.strftime("%Y%m%d-%H%M%S"), name, success, comment))
                    if success:
                        success_count+=1
            pool.close()
            pool.join()
        else:
            with open(init_log_path, file_mode) as file:
                for params in tqdm(queries_pending, total = len(queries_pending)):
                    name, success, comment = self.save_content(params)
                    file.write('{}, {}, {}, {} \n'.format(time.strftime("%Y%m%d-%H%M%S"), name, success, comment))
                    if success:
                        success_count+=1
        
        self.log('{} attemps successful out of {}'.format(success_count, len(queries_pending)))
