#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier,dump_classifier_and_data
from time import time

###  Select features
features_list = ['poi','salary', 'total_payments',
                 'bonus', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'shared_receipt_with_poi'] # You will need to use more features

#features_list = ['poi','bonus','deferred_income',]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###  Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
#for point in data:
#    salary = point[0]
#    bonus = point[1]
#    matplotlib.pyplot.scatter( salary, bonus )

#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()


###  Create new feature(s)
def computeFraction(poi_messages,all_messages):
    fraction = 0
    if all_messages == 'NaN':
        return fraction

    if poi_messages == 'NaN':
        poi_messages = 0

    fraction = 1.0*poi_messages/all_messages
    return fraction

for name in data_dict:
    to_messages = data_dict[name]['to_messages']
    from_messages = data_dict[name]['from_messages']
    from_poi_to_this_person = data_dict[name]['from_poi_to_this_person']
    from_this_person_to_poi = data_dict[name]['from_this_person_to_poi']

    fraction_from_poi = computeFraction(from_poi_to_this_person,to_messages)
    fraction_to_poi = computeFraction(from_this_person_to_poi,from_messages)

    data_dict[name]['fraction_from_poi'] = fraction_from_poi
    data_dict[name]['fraction_to_poi'] = fraction_to_poi

features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')


my_dataset = data_dict
### Store to my_dataset for easy export below.
### Extract features and labels from dataset for local testing


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###  Try a varity of classifiers

from sklearn.cross_validation import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
###use Pipeline
### LogisticRegression method
#clf = Pipeline([('sc',StandardScaler()),
#                 ('pca',PCA(n_components=4)),
#                 ('clf', LogisticRegression(random_state=1))])

#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#print('Test accuracy of LogisticRegression is : %f' % accuracy_score(pred,labels_test))


###  GaussianNB method
from sklearn.naive_bayes import GaussianNB
clf = Pipeline([('sc',StandardScaler()),
                 ('pca',PCA(n_components=4)),
                 ('nb', GaussianNB())])
t0 = time()
clf.fit(features_train,labels_train)
print("training time:",round(time()-t0,3),"s")
t0 = time()
pred = clf.predict(features_test)
print("prediting time:",round(time()-t0,3),"s")
print('Test accuracy of GaussianNB is : %f' % accuracy_score(pred,labels_test))

###  AdaBoostClassifier
#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators = 100)
#t0 = time()
#clf.fit(features_train,labels_train)
#print("training time:",round(time()-t0,3),"s")

#t0 = time()
#pred = clf.predict(features_test)
#print("prediting time:",round(time()-t0,3),"s")
#print(accuracy_score(pred,labels_test))

# Provided to give you a starting point. Try a variety of classifiers.


### DecisionTree method
#from sklearn.tree import DecisionTreeClassifier
#clf = Pipeline([('sc',StandardScaler()),
#                 ('pca',PCA(n_components=4)),
#                 ('tree', DecisionTreeClassifier())])

#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)
#print('Test accuracy of DecisionTree is : %f' % accuracy_score(pred,labels_test))


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import KFold

#kf = KFold(len(my_dataset),10)
#for train_indices,test_indices in kf:
    #make training and testing dataset
#    features_train= [features[ii] for ii in train_indices]
#    features_test= [features[ii] for ii in test_indices]
#    labels_train=[labels[ii] for ii in train_indices]
#    labels_test=[labels[ii] for ii in test_indices]


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
test_classifier(clf, my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)
