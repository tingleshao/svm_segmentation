print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pylab as pl
import csv
from sklearn.decomposition import PCA

def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xmin0 = Xo[:,0].min()
    xmax0 = Xo[:,0].max()
    xmin1 = Xo[:,1].min()
    xmax1 = Xo[:,1].max()
    xmin2 = Xo[:,2].min()
    xmax2 = Xo[:,2].max()
    xmin3 = Xo[:,3].min()
    xmax3 = Xo[:,3].max()
    xmin4 = Xo[:,4].min()
    xmax4 = Xo[:,4].max()
    xmin5 = Xo[:,5].min()
    xmax5 = Xo[:,5].max()
    h1, h2, h3, h4, h5, h6 = np.meshgrid(np.linspace(xmin0, xmax0, 50), np.linspace(xmin1, xmax1, 1),np.linspace(xmin2, xmax2, 1),np.linspace(xmin3, xmax3, 1),np.linspace(xmin4, xmax4, 1),np.linspace(xmin5, xmax5, 50))
    
    Z = classifier.decision_function(np.c_[h1.ravel(), h2.ravel(), h3.ravel(), h4.ravel(), h5.ravel(), h6.ravel()])
    Z = Z.reshape(h1.shape)
  
    print Z.shape
    print h1.shape
    print h2.shape
    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(h1[0,:,0,0,0,:], h5[0,:,0,0,0,:], Z[0,:,0,0,0,:], alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(Xo[:, 0], Xo[:, 5], c=Yo, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone)

    axis.axis('off')
    axis.set_title(title)

def evaluate(classifier, ground_truth, data):
  false_pos = 0
  false_neg = 0
  for i in xrange(len(data)):
    p = classifier.predict(data[i])[0]
    if ground_truth[i] > 0 and p == 0:
      false_neg += 1
    elif ground_truth[i] == 0 and p > 0:
      false_pos += 1
  return [false_pos, false_neg]

# replace X by our data
features = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/features.dat','rb') as csvfile:
  feature_reader = csv.reader(csvfile, delimiter=',')
  for row in feature_reader:
    features.append(map(float, row))
features = np.asarray(features)
Xo = np.reshape(features,(40000,6))
   
# normalize the features 
fsum = np.sum(features, axis=0)
xmin = []
xmax = []
for i in xrange(6):
  xmin.append(Xo[:,i].min())
  xmax.append(Xo[:,i].max())
  
for c in xrange(features.shape[1]):
  Xo[:,c] /= (xmax[c]-xmin[c]) 
  
indices = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/sorted_r_index.dat','rb') as csvfile2:
  indices_reader = csv.reader(csvfile2, delimiter=',')
  for row in indices_reader:
    indices.append(map(int, row))
      
indices = indices[0]    
print "1"


# data used to train the svm
xx = []
yy = []
Yo = []
count1 = count2 = count3 = 0
low_threshold = Xo[indices[int(len(indices)*0.1)]-1][-1]
high_threshold = Xo[indices[int(len(indices)*0.9)]-1][-1]
mid_threshold = Xo[indices[int(len(indices)*0.5)]-1][-1]
print low_threshold
print high_threshold
print mid_threshold
for row in Xo:
  if row[-1] < low_threshold: 
    xx.append(row)
    yy.append(0)
    Yo.append('r')
    count1 += 1
  elif row[-1] < high_threshold:
    Yo.append('b')
    count2 += 1
    
  else:
    Yo.append('g')
    xx.append(row)
    yy.append(1)
    count3+= 1
    
print count1, count2, count3 

X = np.asarray(xx)
Y = np.transpose(np.asarray(yy))

print "2"

print "3"

# data used to train the weighted svm
xxw = []
yyw = []
weights = []

th1 = Xo[indices[int(len(indices)*0.1)]-1][-1]
th2 = Xo[indices[int(len(indices)*0.2)]-1][-1]
th3 = Xo[indices[int(len(indices)*0.3)]-1][-1]
th4 = Xo[indices[int(len(indices)*0.4)]-1][-1]
th6 = Xo[indices[int(len(indices)*0.6)]-1][-1]
th7 = Xo[indices[int(len(indices)*0.7)]-1][-1]
th8 = Xo[indices[int(len(indices)*0.8)]-1][-1]
th9 = Xo[indices[int(len(indices)*0.9)]-1][-1]
print th1, th2, th3, th4,  th6, th7, th8, th9
dw = []
for row in Xo:
  if row[-1] < th1:
    xxw.append(row)
    yyw.append(0)
    weights.append(1000*mid_threshold / row[-1])
    dw.append(10*mid_threshold / row[-1])
  elif row[-1] < th2:
    dw.append(1.0)
    
    continue
    
  #  xxw.append(row)
  #  yyw.append(0)
 #   weights.append(14.0*mid_threshold / row[-1])
 
  elif row[-1] < th3:
    dw.append(1.0)
    
    continue
    
  #  xxw.append(row)
  #  yyw.append(0)
#    weights.append(7.0*mid_threshold / row[-1])
    
  elif row[-1] < th4:
    dw.append(1.0)
    
    continue
    
  #  xxw.append(row)
  #  yyw.append(0)
#    weights.append(10*mid_threshold / row[-1])
    
  elif row[-1] < th6:
    dw.append(1.0)
    
    # do nothing
    continue
  elif row[-1] < th7:
    dw.append(1.0)
    
    continue
    
  #  xxw.append(row)
  #  yyw.append(1)
#    weights.append(0.1*row[-1] / mid_threshold)
  elif row[-1] < th8:
    dw.append(1.0)
    
    continue
    
  #  xxw.append(row)
  #  yyw.append(1)
 #   weights.append(row[-1] / mid_threshold)
  elif row[-1] < th9:
    dw.append(1.0)
    
    continue
    
 #   xxw.append(row)
#    yyw.append(1)
  #  weights.append(row[-1] / mid_threshold)
  else:
    xxw.append(row)
    yyw.append(1)
    weights.append(10*row[-1] / mid_threshold)
    dw.append(10*row[-1] / mid_threshold)
    

Xw = np.asarray(xxw)
Yw = np.transpose(np.asarray(yyw))
Ww = np.asarray(weights)

sample_weight_constant = np.ones(len(X))

# fit the sample weighted SVM
pca = PCA(n_components=2)
pca.fit(Xo)
proj_xo = np.transpose(np.dot(pca.components_,np.transpose(Xo)))

x_min, x_max = proj_xo[:,0].min(), proj_xo[:,0].max()
y_min, y_max = proj_xo[:,1].min(), proj_xo[:,1].max()

xo_range = np.linspace(x_min,x_max)

clf_weights = svm.SVC(kernel='linear')
clf_weights.fit(Xw, Yw, sample_weight=Ww)
ww = clf_weights.coef_[0]
proj_ww = np.transpose(np.dot(pca.components_,ww))
aw = -proj_ww[0] / proj_ww[1]
yww_range = aw * xo_range - clf_weights.intercept_[0] / proj_ww[1]

print "4"

# fit the standard svm
clf_no_weights = svm.SVC(kernel='linear')
clf_no_weights.fit(X, Y)
wnw = clf_no_weights.coef_[0]
proj_wnw = np.transpose(np.dot(pca.components_,wnw))
anw = -proj_wnw[0] / proj_wnw[1]
ywnw_range = anw * xo_range - clf_no_weights.intercept_[0] / proj_wnw[1]


h0 = plt.plot(xo_range,yww_range, 'k-')
h1 = plt.plot(xo_range,ywnw_range,'b--')
plt.scatter(proj_xo[:,0], proj_xo[:,1], c=Yo, cmap=pl.cm.Paired, s= dw)
print "5"

# read the ground truth file 
ground_truth = []
for line in open('/Users/chongshao-mikasa/Data/video_data_in_txt/gtx.dat','r'):
  ground_truth.append(line)

ground_truth = np.asarray(map(int,map(float,ground_truth[0].strip().split("   "))))

# call the function to compute the errors 
# TODO: make the test data the middle 20% percent 
[fpww,fnww] = evaluate(clf_weights, ground_truth, Xo)
[fpwnw,fnwnw] = evaluate(clf_no_weights, ground_truth, Xo)
print fpww, fnww 
print fpwnw, fnwnw

plt.show()
