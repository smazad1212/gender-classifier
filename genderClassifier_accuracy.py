from sklearn import tree
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import svm
from sklearn import ensemble
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177,70,43], [160,60,38], [154, 54, 37],
     [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,55,37],
     [171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
    'male', 'female', 'male', 'female', 'male']

dtc = tree.DecisionTreeClassifier().fit(X,Y).predict([[190,90,44]])
dtc_accuracy = metrics.accuracy_score(dtc, ['male'])

gnb = naive_bayes.GaussianNB().fit(X,Y).predict([[190,90,44]])
gnb_accuracy = metrics.accuracy_score(gnb, ['male'])

linear_svc = svm.LinearSVC(tol=1e-6).fit(X,Y).predict([[190,90,44]])
linear_svc_accuracy = metrics.accuracy_score(linear_svc, ['male'])

abc = ensemble.AdaBoostClassifier(n_estimators=100).fit(X,Y).predict([[190,90,44]])
abc_accuracy = metrics.accuracy_score(abc, ['male'])

index = np.argmax([dtc_accuracy, gnb_accuracy, linear_svc_accuracy, abc_accuracy])
classfiers = {
     0: 'DecisionTreeClassifier',
     1: 'GaussianNB',
     2: 'LinearSVC',
     3: 'AdaBoostClassifier'
}

# print 'dtc:', str(dtc), '--> accuracy:', str(dtc_accuracy)
# print 'gnb:', str(gnb), '--> accuracy:', str(gnb_accuracy)
# print 'linear_svc:', str(linear_svc), '--> accuracy:', str(linear_svc_accuracy)
# print 'abc:', str(abc), '--> accuracy:', str(abc_accuracy)

print('Best gender classifier is {}'.format(classfiers[index]))