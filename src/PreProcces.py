from hazm import *
import re
from pandas import DataFrame, ExcelWriter
from src.Log import Logger
from hazm import word_tokenize, sent_tokenize
import pandas as pd
from src.comman import Constant
from PersianStemmer import PersianStemmer

ps= PersianStemmer()

Log = Logger()
normalizer = Normalizer()
stemmer = Stemmer()
lemmatizer = Lemmatizer()

class PreProccesser:
    def __init__(self, data_frame: DataFrame):
        self.data = data_frame

    def process(self):
        Log.logger.info('Start processing ...')

        data = self.data.values
        new_data =[]
        for i in range(data.shape[0]):
            func_result = self.run_funcs(data[i][0])
            if func_result != None:
                new_data.append([func_result,data[i][1],data[i][2]])

        self.data = pd.DataFrame(new_data)
        Log.logger.info('Processing finshed')

        return new_data

    # @property
    # def data(self):
    #     return self.data

    def run_funcs(self,val):
        try:
            pass
            val = self.remove_url(val)
            val = self.remove_name(val)
            val = self.remove_special_character(val)

            # val = self.remove_stop_words(val)
            val = self.normalize(val)
            val = self.remove_stop_words(val)
            # val = self.lemma(val)
            # val = self.s_normal(val)
            val = self.singel_char(val)
            # val = self.stremme(val)

        except:
            return None
        return val

    @staticmethod
    def remove_url(val):
        Log.logger.info('All url removed')
        return re.sub(r'https?://[^ ]+', '', val)

    @staticmethod
    def remove_name(val):
        Log.logger.info('All name removed')

        val = re.sub(r'www.[^ ]+', '', val)
        return  re.sub(r'@[A-Za-z0-9_]+', '',val)

    @staticmethod
    def remove_special_character(val):
        Log.logger.info('All url special character')

        return re.sub(r"[a-zA-Z!$()&@0-9:\\#/|{}<>?؟=.\"\'…»«;,،]", "", val)

    @staticmethod
    def remove_stop_words(val):
        Log.logger.info('Stop words removed')
        stops = Constant.STOP_WORDS
        words = [[word for word in word_tokenize(sentence) if word not in stops] for sentence in sent_tokenize(val)]
        words = words[0]

        val = ' '.join(words)
        return val

    @staticmethod
    def s_normal(val):
        words = []
        for sentence in sent_tokenize(val):
            for word in word_tokenize(sentence):
                end = word.find('#')
                if end == -1:
                    end = len(word)
                words.append(word[:end])
        val = ' '.join(words)
        return val

    @staticmethod
    def singel_char(val):
        words = [[word for word in word_tokenize(sentence) if len(word)>1] for sentence in sent_tokenize(val)]
        words = words[0]
        val = ' '.join(words)
        return val

    @staticmethod
    def normalize(val):
        Log.logger.info('Data normalized by hazm package')
        # words = [[normalizer.normalize(word) for word in word_tokenize(sentence)] for sentence in sent_tokenize(val)]
        # words = words[0]
        # val = ' '.join(words)
        # return val
        return normalizer.normalize(val)

    @staticmethod
    def stremme(val):
        Log.logger.info('Data stemme by hazm package ')
        # words = [[stemmer.stem(word) for word in word_tokenize(sentence)] for sentence in sent_tokenize(val)]
        words = [[ps.run(word) for word in word_tokenize(sentence)] for sentence in sent_tokenize(val)]
        words = words[0]
        val = ' '.join(words)
        return val

    @staticmethod
    def lemma(val):
        words = [[lemmatizer.lemmatize(word) for word in word_tokenize(sentence)] for sentence in sent_tokenize(val)]
        words = words[0]
        val = ' '.join(words)
        return val

    def convert_to_excel(self,path=Constant.OUTPUT_PATH,file_name=Constant.DATA_OUTPUT_FILE_NAME,sheet_num=1):
        Log.logger.info('New data saved as {} in "{}" path'.format(file_name,path))
        writer = ExcelWriter('{}/{}'.format(path,file_name))
        self.data.to_excel(writer, 'Sheet{}'.format(sheet_num))
        writer.save()