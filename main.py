'''
""
================================
SVM Exercise
================================

A tutorial exercise for using different SVM kernels.
Problem Statment : Classification of three different varieties of wheat.
Data set Measurements of geometrical properties of kernels belonging to three different varieties of wheat. A soft X-ray technique and GRAINS package were used to construct all seven, real-valued attributes.
210 samples 7 features and 3 output classes
'''
print(__doc__)
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets, metrics
from matplotlib.colors import ListedColormap

inputs  = []
outputs = []

DS_TRAINING_RATIO = .7 # The ratio of the dataset that will be used in training
DS_TESTING_RATIO = 1 - DS_TRAINING_RATIO
NUMBER_OF_CLASSES = 3


with open('dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter ='\t')
    for row in csv_reader:
        tmp = []
        for i in range(len(row) - 1):
            if row[i]:
                tmp.append(float(row[i]))
        inputs.append(tmp)
        outputs.append(int(row[len(row) - 1]))


nTests = int(DS_TESTING_RATIO*len(inputs) / NUMBER_OF_CLASSES)
nEachTest = NUMBER_OF_CLASSES  * [nTests]

#print(nEachTest)
trainingInputs = []
trainingTargets = []
testingInputs = []
testingTargets = []

# separating training set and test set
for i in range(len(inputs)):
    tmp = NUMBER_OF_CLASSES * [0]
    if nEachTest[outputs[i] - 1] > 0:
        testingInputs.append(inputs[i])
        tmp[outputs[i] - 1] = 1
        testingTargets.append(tmp.index(1))
        nEachTest[outputs[i] - 1] -= 1
    else:
        trainingInputs.append(inputs[i])
        tmp[outputs[i] - 1] = 1
        trainingTargets.append(tmp.index(1))

trainingInputs = np.array(trainingInputs)
trainingTargets = np.array(trainingTargets)
testingInputs = np.array(testingInputs)
testingTargets = np.array(testingTargets)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainingInputs = sc.fit_transform(trainingInputs)
testingInputs = sc.transform(testingInputs)

#Create a svm Classifier
#change kernel (linear - poly - rbf) and adjust c & g
c , g , k = 1, 1, 'linear'
clf = svm.SVC(kernel= k, C = c, gamma = g)
#Train the model using the training sets
svc = clf.fit(trainingInputs, trainingTargets)
#Predict the response for test dataset
from sklearn.metrics import confusion_matrix

y_pred = clf.predict(testingInputs)
#confusion matrix
cm = confusion_matrix(testingTargets, y_pred)
#print(cm)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(testingTargets, y_pred) * 100)

# visualising the descision boundaries in 2d just pick 2 features from 0 to 6 (f1, f2)
f1, f2 = 0, 1
#change f1 and f2 between 0 and 6 to select the best 2 features to visualize calssification in the 2d space
# Visualising the results (training - testing)
def draw_results(X_set, y_set, *param):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, f1].min() - 1, stop = X_set[:, f1].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, f2].min() - 1, stop = X_set[:, f2].max() + 1, step = 0.01))
    # 2 features selectd and the rest 5 are default values of 0
    Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(5)]).T
    # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x7
    pred = clf.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
    plt.contourf(X1, X2, pred,
                 alpha = 0.75, cmap = ListedColormap(("red", "green", "blue")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        color = ['r' if i == 0 else 'g' if i == 1 else 'b']
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = color, label = 'class ' + str(j + 1))

    plt.title('SVM' + (' training ' if len(param) > 0 else ' testing ') + str(k) + ' Kernel c = ' + str(c) + ', gamma = ' + str(g))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
# drawing training results
draw_results(trainingInputs, trainingTargets, f1, f2, c, g)
# drawing testing results
draw_results(testingInputs, y_pred)

