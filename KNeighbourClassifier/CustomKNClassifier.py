from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance



def euc(a,b):
    return distance.euclidean(a,b)

class KNN():
    def fit(self,features_train,labels_train):
        self.features_train=features_train
        self.labels_train=labels_train
    
    def predict(self,features_test):
        predictions=[]
        for item in features_test:
            label=self.closest(item)
            predictions.append(label)
        return predictions

    def closest(self,item):
        best_dist=euc(item,features_train[0])
        best_index=0
        for i in range(1,len(features_train)):
            dist=euc(item,self.features_train[i])
            if dist< best_dist:
                best_dist=dist
                best_index=i
        return self.labels_train[best_index]



iris=datasets.load_iris()
features=iris.data
labels=iris.target
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=.5)
my_classfier=KNN()
# KNeighborsClassifier()
my_classfier.fit(features_test,labels_test)
# prediction=my_classfier.predict(features_test)
# print(accuracy_score(labels_test,prediction))
iris_new=[[5.2,3.6,1.8,.3]]
iris_new_predict=my_classfier.predict(iris_new)

if iris_new_predict[0]==0:
    print("Setosa")
if iris_new_predict[0]==1:
    print("Versicolor")
if iris_new_predict[0]==2:
    print("Virginica")
# if iris_new_predict[0]==0:
#     print("Setosa")
# elif iris_new_predict[0]==1:
#     print("Versicolor")
# else:
#     print("	Virginica")


