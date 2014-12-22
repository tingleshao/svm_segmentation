print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pylab as pl
import csv
from sklearn.decomposition import PCA


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

def gen_image(classifier,data):
  img = ""
  for i in xrange(len(data)):
    p = classifier.predict(data[i])[0]
    if p == 0:
      img += "0 "
    else:
      img += "255 "
  return img
    

# read the ground truth file 

ground_truth = []
for line in open('/Users/chongshao-mikasa/Data/video_data_in_txt/gtx.dat','r'):
  ground_truth.append(line)

ground_truth = np.asarray(map(int,map(float,ground_truth[0].strip().split("   "))))

# read data

features = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/features.dat','rb') as csvfile:
  feature_reader = csv.reader(csvfile, delimiter=',')
  for row in feature_reader:
    features.append(map(float, row))
features = np.asarray(features)
'''   
features = []
for line in open('/Users/chongshao-mikasa/Data/video_data_in_txt/video1_features.dat','r'):
  features.append(map(float,line.strip().split("  ")))
features = np.asarray(features)
'''
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
'''
indices = []
for line in open('/Users/chongshao-mikasa/Data/video_data_in_txt/video1_indices.dat','r'):
  indices.append(line)

indices = np.asarray(map(int,map(float,indices[0].strip().split("   "))))
'''
print "1"

# randomly pick the data used to train the svm
pos_c = 0
neg_c = 0
xx = []
yy = []
Yo = []

mid_threshold = Xo[indices[int(len(indices)*0.5)]-1][-1]

for i in xrange(len(Xo)):
  #print Xo[i][-1]
  if ground_truth[i] == 0 and neg_c < 100: 
    neg_c += 1
    xx.append(Xo[i])
    yy.append(0)
  elif ground_truth[i] > 0 and pos_c < 100:
    pos_c += 1
    xx.append(Xo[i])
    yy.append(1)
  if neg_c == pos_c == 100:
    break
    
X = np.asarray(xx)
print X.shape
Y = np.transpose(np.asarray(yy))
print Y.shape
print "2"

print "3"

# data used to train the weighted svm
xxw = []
yyw = []
weights = []

pos_c = neg_c = 0
dw = []
for i in xrange(len(Xo)):
  if ground_truth[i] == 0 and neg_c < 100:
    neg_c += 1
    xxw.append(Xo[i])
    yyw.append(0)
    weights.append(100*mid_threshold / Xo[i][-1])
    dw.append(10*mid_threshold / Xo[i][-1])
  elif ground_truth[i]>0 and pos_c < 100:
    pos_c += 1
    xxw.append(Xo[i])
    yyw.append(1)
    weights.append(100*Xo[i][-1] / mid_threshold)
    dw.append(10*Xo[i][-1] / mid_threshold)
  if neg_c == pos_c == 100:
    break
    

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


#h0 = plt.plot(xo_range,yww_range, 'k-')
#h1 = plt.plot(xo_range,ywnw_range,'b--')
#plt.scatter(proj_xo[:,0], proj_xo[:,1], c=ground_truth, cmap=pl.cm.Paired, s= dw)
plt.scatter(proj_xo[:,0], proj_xo[:,1], c=ground_truth, cmap=pl.cm.Paired, s= dw)

print "5"


# call the function to compute the errors 
[fpww,fnww] = evaluate(clf_weights, ground_truth, Xo)
[fpwnw,fnwnw] = evaluate(clf_no_weights, ground_truth, Xo)
print fpww, fnww 
print fpwnw, fnwnw

img_ww = gen_image(clf_weights,Xo)
img_wnw = gen_image(clf_no_weights,Xo)

img_ww_file = open("img_ww_syn.txt", "w")
img_ww_file.write(img_ww.strip())
img_ww_file.close()

img_wnw_file = open("img_wnw_syn.txt","w")
img_wnw_file.write(img_wnw.strip())
img_wnw_file.close()

plt.show()
