
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sample_weight = sample_weight_constant
classifier = clf_no_weights
axis = axes[0]
title = "constant weights"

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
h1, h2, h3, h4, h5, h6 = np.meshgrid(np.linspace(xmin0, xmax0, 500), np.linspace(xmin1, xmax1, 1),np.linspace(xmin2, xmax2, 1),np.linspace(xmin3, xmax3, 1),np.linspace(xmin4, xmax4, 1),np.linspace(xmin5, xmax5, 500))
    
Z1 = classifier.decision_function(np.c_[h1.ravel(), h2.ravel(), h3.ravel(), h4.ravel(), h5.ravel(), h6.ravel()])
Z1 = Z1.reshape(500,500)
print Z.shape
print h1.shape
print h2.shape
# plot the line, the points, and the nearest vectors to the plane
axis.contourf(h1[0,:,0,0,0,:], h5[0,:,0,0,0,:], Z1, alpha=0.75, cmap=plt.cm.bone)
axis.scatter(Xo[:, 0], Xo[:, 5], c=Yo, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone)
								 
sample_weight2 = Ww
classifier = clf_weights
axis = axes[1]
title = "modified weights"

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
    
Z2 = classifier.decision_function(np.c_[h1.ravel(), h2.ravel(), h3.ravel(), h4.ravel(), h5.ravel(), h6.ravel()])
Z2 = Z.reshape(h1.shape)
print Z.shape
print h1.shape
print h2.shape
# plot the line, the points, and the nearest vectors to the plane
axis.contourf(h1[0,:,0,0,0,:], h5[0,:,0,0,0,:], Z2[0,:,0,0,0,:], alpha=0.75, cmap=plt.cm.bone)
axis.scatter(Xw[:, 0], Xw[:, 5], c=Yw, s=100 * sample_weight2, alpha=0.9,
                 cmap=plt.cm.bone)

plt.show()