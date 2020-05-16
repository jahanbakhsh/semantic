from src.comman import Constant
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif,f_regression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from src.Log import Logger


RANDOM_STATE = Constant.RANDOM_STATE
Log = Logger()
class Classify:
    # tuned_params = [{'C': [1, 10, 100, 1000]}]
    tuned_params = [{'C': [1, 56, 342, 987]}]
    # probability will be True OR False
    # kernel will be witch one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    # svc_tuned_params = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},]
    svc_tuned_params = [{'kernel': ['rbf'], 'C': [1, 89, 453, 876], 'gamma': [0.7878, 0.53456],},]

    def __init__(self, recordes):
        self.labels, self.scrams, self.doc= self.clean_data(recordes)


    def clean_data(self, recordes):
        labels = []
        scram = []
        docs = []
        for item in recordes:
            labels.append(item[1])
            scram.append(item[2])
            docs.append(item[0])

        return labels, scram, docs

    def split_data(self):
        yes_data = []
        no_data = []
        yes_label = []
        no_label = []

        for i in range(len(self.doc)):
            if Constant.SPLIT_DATA and self.scrams[i] == 'no':
                no_label.append(self.labels[i])
                no_data.append(self.doc[i])
            else:
                yes_data.append(self.doc[i])
                yes_label.append(self.labels[i])

        return yes_label, yes_data, no_label, no_data


    def create_train_test(self):
        vectorizer = TfidfVectorizer(lowercase=False, max_df=0.8)

        yes_label, yes_data, no_label, no_data = self.split_data()

        fs_no_train = []
        fs_no_test = []
        labels_no_train = []
        labels_no_test = []

        #''':create sparse matrix, [n_samples, n_features]
        #    Tf-idf-weighted document-term matrix.'''

        fs_yes = vectorizer.fit_transform(yes_data)

        #'Select features according to a percentile of the highest scores'
        selector = SelectPercentile(f_regression, percentile=15)

        #Run score function on (X, y) and get the appropriate features
        selector.fit(fs_yes, yes_label)

        #'''array of shape [n_samples, n_selected_features]
        #    The input samples with only the selected features'''
        fs_yes = selector.transform(fs_yes)

        fs_yes_train, fs_yes_test, labels_yes_train, labels_yes_test = train_test_split(
            fs_yes, yes_label, test_size=Constant.TEST_SIZE, random_state=RANDOM_STATE
        )
        if Constant.SPLIT_DATA:
            fs_no = vectorizer.fit_transform(no_data)
            selector = SelectPercentile(chi2, percentile=10)
            selector.fit(fs_no, no_label)
            fs_no = selector.transform(fs_no)
            fs_no_train, fs_no_test, labels_no_train, labels_no_test = train_test_split(
                fs_no, no_label, test_size=Constant.TEST_SIZE, random_state=RANDOM_STATE
            )

        return fs_no_train, fs_no_test, labels_no_train, labels_no_test,fs_yes_train, fs_yes_test, labels_yes_train, labels_yes_test

    @staticmethod
    def  my_ensemble( fs_train, fs_test, labels_train, clu='no'):
        clf_NaiveBayes = BernoulliNB()
        clf_LinearSVC = GridSearchCV(LinearSVC(), Classify.tuned_params, cv=5, scoring='accuracy')
        clf_SVC = GridSearchCV(SVC(), Classify.svc_tuned_params, cv=5, scoring='accuracy')

        clf_Boosting = GradientBoostingClassifier(n_estimators=5, random_state=RANDOM_STATE)

        clf_NaiveBayes.fit(fs_train, labels_train)
        clf_LinearSVC.fit(fs_train, labels_train)
        clf_SVC.fit(fs_train, labels_train)
        clf_Boosting.fit(fs_train, labels_train)


        pred = []
        if clu == 'yes':
            for item in fs_test:
                if clf_NaiveBayes.predict(item) ==0:
                    pred.append(0)
                else:
                    pred.append(clf_SVC.predict(item))
        else:
            for item in fs_test:
                out_maj = Classify.major_vote([
                    clf_SVC.predict(item)[0],
                    clf_NaiveBayes.predict(item)[0],
                    clf_LinearSVC.predict(item)[0]
                ])
                if clf_NaiveBayes.predict(item)[0] ==1:
                    pred.append(1)
                elif clf_LinearSVC.predict(item) == 0:
                    pred.append(0)
            #     out = clf_SVC.predict(item)
            #     if out in [0]:
            #         pred.append(out)
                else:
                    pred.append(out_maj)
        return pred

    @staticmethod
    def major_vote(votes):
        dic = dict()
        for item in set(votes):
            dic[item] = votes.count(item)

        return dic.popitem()[0]
    @staticmethod
    def create_classifier(CLASSIFIER, fs_train, fs_test, labels_train, clu='no'):
        clf = None
        pred = None

        if CLASSIFIER == 'NaiveBayes':
            clf = BernoulliNB()
        elif CLASSIFIER == 'LinearSVC':

            clf = GridSearchCV(LinearSVC(), Classify.tuned_params, cv=5, scoring='accuracy')

        elif CLASSIFIER == 'SVC':
            clf = GridSearchCV(SVC(), Classify.svc_tuned_params, cv=5, scoring='accuracy')

        elif CLASSIFIER == 'Ensemble':
            # clf = GradientBoostingClassifier(n_estimators=5, random_state=RANDOM_STATE)
            # fs_train = fs_train.toarray()
            # fs_test = fs_test.toarray()
            pred=Classify.my_ensemble(fs_train, fs_test, labels_train, clu)
            return pred

        clf.fit(fs_train, labels_train)
        pred = clf.predict(fs_test)

        return pred

    def evaluate(self, CLASSIFIERS):
        accuracy = []
        recall = []
        precision = []
        f1 = []

        fs_no_train, fs_no_test, labels_no_train, \
        labels_no_test, fs_yes_train, fs_yes_test, \
        labels_yes_train, labels_yes_test = self.create_train_test()

        print('NO-TRAIN')
        print(labels_no_train.count(-1))
        print(labels_no_train.count(0))
        print(labels_no_train.count(1))
        print(labels_no_train.count(2))
        print('NO-TEST')
        print(labels_no_test.count(-1))
        print(labels_no_test.count(0))
        print(labels_no_test.count(1))
        print(labels_no_test.count(2))
        print('YES-TRAIN')
        print(labels_yes_train.count(-1))
        print(labels_yes_train.count(0))
        print(labels_yes_train.count(1))
        print(labels_yes_train.count(2))
        print('YES-TEST')
        print(labels_yes_test.count(-1))
        print(labels_yes_test.count(0))
        print(labels_yes_test.count(1))
        print(labels_yes_test.count(2))

        acuracy_no = 0
        recall_no =0
        precision_no =0
        f1_no=0

        if Constant.SPLIT_DATA:
            yes_data_size = 0.5
        else:
            yes_data_size = len(labels_yes_train) / (len(labels_no_train) + len(labels_yes_train))
        for CLSS in CLASSIFIERS:

            if Constant.SPLIT_DATA:
                pred_no = self.create_classifier(CLSS,fs_no_train,fs_no_test,labels_no_train)

                acuracy_no = metrics.accuracy_score(labels_no_test, pred_no)
                recall_no = metrics.recall_score(labels_no_test, pred_no,average='macro')
                precision_no = metrics.precision_score(labels_no_test, pred_no,average='macro')
                f1_no = metrics.f1_score(labels_no_test, pred_no,average='macro')

            pred_yes = self.create_classifier(CLSS,fs_yes_train, fs_yes_test, labels_yes_train, clu='yes')

            acuracy_yes = metrics.accuracy_score(labels_yes_test,pred_yes)
            recall_yes = metrics.recall_score(labels_yes_test,pred_yes, average='macro')
            precision_yes = metrics.precision_score(labels_yes_test,pred_yes,average='macro')
            f1_yes = metrics.f1_score(labels_yes_test,pred_yes,average='macro')

            accuracy.append(round((1-yes_data_size)*acuracy_no + (yes_data_size *acuracy_yes),ndigits=2))
            recall.append(round((1-yes_data_size)*recall_no + (yes_data_size *recall_yes),ndigits=2))
            precision.append(round((1-yes_data_size)*precision_no + (yes_data_size *precision_yes),ndigits=2))
            f1.append(round((1-yes_data_size)*f1_no + (yes_data_size *f1_yes),ndigits=2))


            print(CLSS)

            print(metrics.confusion_matrix(labels_yes_test,pred_yes))
            if Constant.SPLIT_DATA:
                print(metrics.confusion_matrix(labels_no_test,pred_no))
        return accuracy, recall, precision, f1

