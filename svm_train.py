# fitting:


import numpy as np
import pylab as pl
from sklearn import svm
import csv
from sklearn.decomposition import PCA



clf = svm.SVC(kernel='linear')

features = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/features.dat','rb') as csvfile:
  feature_reader = csv.reader(csvfile, delimiter=',')
  for row in feature_reader:
    features.append(map(float, row))
    

indices = []
with open ('/Users/chongshao-mikasa/Data/video_data_in_txt/sorted_r_index.dat','rb') as csvfile2:
  indices_reader = csv.reader(csvfile2, delimiter=',')
  for row in indices_reader:
    indices.append(map(int, row))
      
indices = indices[0]

print "1"
# preparing the features
# correlation scores, intensity over time quantile fit, and weight
x = features
y = []
low_threshold = features[indices[int(len(indices)*0.1)]][-1]
high_threshold = features[indices[int(len(indices)*0.9)]][-1]
count1 = 0
count2 = 0
count3 = 0

# data used to train the svm
xx = []
yy = []

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

print count1, count2, count3
print "\n"
# need to convert x into an array.
print len(features)
f_mat = np.reshape(features,(40000,6))

x = f_mat
x = np.asarray(x)

xxx = np.asarray(xx)
yyy = np.asarray(yy)

# PCA on x 
pca = PCA(n_components=2)
pca.fit(x)
proj_x = np.transpose(np.dot(pca.components_,np.transpose(x)))
print "2"


x_min, x_max = proj_x[:,0].min(), proj_x[:,0].max()
y_min, y_max = proj_x[:,1].min(), proj_x[:,1].max()

clf.fit(xxx, yyy)
w = clf.coef_[0]
proj_w = np.transpose(np.dot(pca.components_,w))

a = -proj_w[0] / proj_w[1]
x_range = np.linspace(x_min,x_max)
y_range = a * x_range - clf.intercept_[0] / proj_w[1]
print "3"



# = = = = = = = = = 
#x = features
#y = []
#low_threshold = features[indices[int(len(indices)*0.1)]][-1]
#high_threshold = features[indices[int(len(indices)*0.9)]][-1]
#count1 = 0
#count2 = 0
#count3 = 0

wxx1 = []
wyy1 = []
wxx2 = []
wyy2 = []

for row in features:
  if row[-1] < high_threshold: 
#    y.append(1)
#    count1 += 1
    wxx1.append(row)
    wyy1.append(0)
#  elif row[-1] < high_threshold:
#    y.append(20000)
#    count2 += 1 
  else:
#    y.append(40000)
#    count3 += 1
    wxx2.append(row)
    wyy2.append(1)

#print count1, count2, count3
#print "\n"
# need to convert x into an array.
#print len(features)
#f_mat = np.reshape(features,(40000,6))

#x = f_mat
#x = np.asarray(x)
wxx = wxx1 + wxx2
wyy = wyy1 + wyy2
wxxx = np.asarray(wxx)
wyyy = np.asarray(wyy)

# = = = = = = =
# ----------------------------------------
#
# visualize it
#h = 0.1                     
#x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1 
#y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max,h),
#                     np.arange(y_min, y_max,h))
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Z = Z.reshape(xx.shape)
#pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
wclf = svm.SVC(kernel='linear', class_weight={1:10})
wclf.fit(wxxx,wyyy)
ww = wclf.coef_[0]
proj_ww = np.transpose(np.dot(pca.components_,ww))
aw = -proj_ww[0] / proj_ww[1]
yw_range = aw * x_range - wclf.intercept_[0] / proj_ww[1]

print "4"

pl.axis('off')
h0 = pl.plot(x_range,y_range, 'k-')
h1 = pl.plot(x_range,yw_range, 'k--')
pl.scatter(proj_x[:,0], proj_x[:,1],c=y, cmap=pl.cm.Paired)
print x_min, x_max, y_min, y_max
print pl.axis()
#pl.ylim([-300,500])
#pl.xlim([-200,1700])
#pl.axis('tight')
pl.show()                     
 
 