from src.DataReader import DataLoder
from src.comman import Constant
from src.Log import Logger
from src.PreProcces import PreProccesser
from src.Classify import Classify


DATA_PATH = Constant.DATA_PATH
DATA_FILE_NAME = Constant.DATA_FILE_NAME



CLASSIFIERS = ['Ensemble','SVC', 'LinearSVC', 'NaiveBayes']
# CLASSIFIERS = ['Ensemble']

Log = Logger()


if __name__ =='__main__':

    Log.logger.info('Project starting ...')
    loader = DataLoder(DATA_PATH, DATA_FILE_NAME)
    data = loader.load()

    proc = PreProccesser(data)
    data = proc.process()
    proc.convert_to_excel()

    cls = Classify(data)

    acc, recall, precetion, f1 = cls.evaluate(CLASSIFIERS)
    print('acc',acc)
    print('recall ',recall)
    print('precetion ',precetion)
    print('f1', f1)
    # print(acc, recall, precetion, f1)

    Log.logger.info('Project end')


