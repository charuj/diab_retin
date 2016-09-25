import numpy as np
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier

X=  pickle.load( open( "file_images_array.p", "rb" ) )
Y=  pickle.load( open( "sampletarget_labels.p", "rb" ) )

Y= np.delete(Y, (2), axis=0)
Y= np.delete(Y, (2), axis=0)
print Y.shape

x_train= X[:-2]
x_test= X[-2:]

y_train= Y[:-2]
y_test= Y[-2:]

print x_train.shape, y_train.shape, x_test.shape, y_test.shape


#Random Forest Classifier
clf_rf= RandomForestClassifier()
clf_rf.fit(x_train,y_train)
y_pred_rf = clf_rf.predict(x_test)
#y_pred_rf= y_pred_rf.reshape([y_pred_rf.shape[0], 1])

acc_rf = clf_rf.score(y_test, y_pred_rf)
print "random forest accuracy: ",acc_rf

