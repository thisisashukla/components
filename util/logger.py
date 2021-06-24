import os
import logging
from datetime import datetime

class Logger():

    def __init__(self, log_path = 'logs'):

        self.log_path = log_path

        if os.path.exists(self.log_path):
            print(f"Log path {self.log_path} exists")
        else:
            os.mkdir(self.log_path)
        
        logging.basicConfig(f"log_{datetime.now().strftime("%Y_%M_%D_%h_%m_%s")}.log", 
                            level = logginer.INFO)

        self.logger = logging
        self.log_id = 1

        self.log(f"LOGGER => Logger setup")

    def log(self, msg):

        msg = f"{self.log_id}: {msg}"

        self.log_id+=1

        self.logger.INFO(msg)
        print(msg)
    

