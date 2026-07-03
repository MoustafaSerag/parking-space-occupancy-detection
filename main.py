#import
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


#preparedata
input_dir = 'clf-data'
categories = ['empty','not_empty']

data=[]
labels=[]

for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img, (15,15))  
        data.append(img.flatten())
        labels.append(category_idx)
data = np.asarray(data) #[[123, 255, ...], [...], ...]]
labels = np.asarray(labels)#[1, 0, 1, ...]

#train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels,shuffle=True)

#train classifier
classifier = SVC()

param_grid = [{
    'C': [ 1, 10,100,1000],
    'gamma': [0.01,0.001,0.0001]
}]
grid_search = GridSearchCV(classifier, param_grid)

grid_search.fit(X_train, y_train)

#evaluate
best_classifier = grid_search.best_estimator_

y_pred = best_classifier.predict(X_test)

score = accuracy_score(y_pred, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))


#save model 
pickle.dump(best_classifier,open('./model.p','wb'))
