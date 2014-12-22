print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=Y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone)

    axis.axis('off')
    axis.set_title(title)


# we create 20 points
np.random.seed(0)
# replace X by our data
features = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/features.dat','rb') as csvfile:
  feature_reader = csv.reader(csvfile, delimiter=',')
  for row in feature_reader:
    features.append(map(float, row))
    
X = np.r_[np.reshape(features,(40000,6))]
Y = 

# data used to train the svm
xx = []
yy = []

low_threshold = features[indices[int(len(indices)*0.1)]][-1]
high_threshold = features[indices[int(len(indices)*0.9)]][-1]
for row in features:
  if row[-1] < low_threshold: 
    y.append(1)
    count1 += 1
    xx.append(row)
    yy.append(0)
  elif row[-1] < high_threshold:
    y.append(20000)
    count2 += 1 
  else:
    y.append(40000)
    count3 += 1
    xx.append(row)
    yy.append(1)

f_mat = np.reshape(features,(40000,6))

x = f_mat
x = np.asarray(x)

xxx = np.asarray(xx)
yyy = np.asarray(yy)

pca = PCA(n_components=2)
pca.fit(x)
proj_x = np.transpose(np.dot(pca.components_,np.transpose(x)))
print "2"

x_min, x_max = proj_x[:,0].min(), proj_x[:,0].max()
y_min, y_max = proj_x[:,1].min(), proj_x[:,1].max()

#clf.fit(xxx, yyy)
X = xxx
Y = yyy
w = clf.coef_[0]
proj_w = np.transpose(np.dot(pca.components_,w))

a = -proj_w[0] / proj_w[1]
x_range = np.linspace(x_min,x_max)
y_range = a * x_range - clf.intercept_[0] / proj_w[1]
print "3"

# data used to train the weighted svm
xxw = []
yyw = []
weights = []

th1 = features[indices[int(len(indices)*0.1)]][-1]
th2 = features[indices[int(len(indices)*0.2)]][-1]
th3 = features[indices[int(len(indices)*0.3)]][-1]
th4 = features[indices[int(len(indices)*0.4)]][-1]
th6 = features[indices[int(len(indices)*0.6)]][-1]
th7 = features[indices[int(len(indices)*0.7)]][-1]
th8 = features[indices[int(len(indices)*0.8)]][-1]
th9 = features[indices[int(len(indices)*0.9)]][-1]

for row in features:
  if row[-1] < th1:
    xxw.append(row)
    yyw.append(0)
    weights.append(1.0)
  elif row[-1] < th2:
    xxw.append(row)
    yyw.append(0)
    weights.append(0.7)
  elif row[-1] < th3:
    xxw.append(row)
    yyw.append(0)
    weights.append(0.4)
  elif row[-1] < th4:
    xxw.append(row)
    yyw.append(0)
    weights.append(0.1)
  elif row[-1] < th6:
    # do nothing
    continue
  elif row[-1] < th7:
    xxw.append(row)
    yyw.append(1)
    weights.append(0.1)
  elif row[-1] < th8:
    xxw.append(row)
    yyw.append(1)
    weights.append(0.4)
  elif row[-1] < th9:
    xxw.append(row)
    yyw.append(1)
    weights.append(0.7)
  else:
    xxw.append(row)
    yyw.append(1)
    weights.append(1.0)

sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
# and bigger weights to some outliers
sample_weight_last_ten[15:] *= 5
sample_weight_last_ten[9] *= 15

# for reference, first fit without class weights

# fit the model
clf_weights = svm.SVC()
clf_weights.fit(X, Y, sample_weight=sample_weight_last_ten)

clf_no_weights = svm.SVC()
clf_no_weights.fit(X, Y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
                       "Constant weights")
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
                       "Modified weights")

plt.show()