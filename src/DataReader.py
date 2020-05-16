import os
import pandas as pd
from src.comman import Constant
from src.Log import Logger



Log = Logger()


class DataLoder:
    def __init__(self, dir_path, file_name):
        self.data_path = '{}/{}'.format(Constant.DATA_PATH, Constant.DATA_FILE_NAME)

    def load(self):
        Log.logger.info('Try to loading data')
        try:
            data = pd.read_excel(self.data_path,encoding='utf-8', errors='ignore')
        except(Exception , ValueError) as e:
            Log.logger.error('Data loading taken error = {}'.format(e))
            return
        Log.logger.info('Succefuly data loaded')
        return data
