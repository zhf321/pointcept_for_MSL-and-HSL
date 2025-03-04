import faiss
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

x = np.random.rand(1000, 400).astype('float32')

ncentroids = 100
niter = 20
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(x)

D1, I1 = kmeans.index.search(x, 1)

faiss_I1 = I1.reshape(-1)
# index = faiss.IndexFlatL2 (d)
# index.add(x)
# D2, I2 = index.search (kmeans.centroids, 1)



primitive_labels = KMeans(n_clusters=ncentroids, 
                              n_init=niter, 
                              random_state=0).fit_predict(x)

histogram = np.bincount(ncentroids* primitive_labels.astype(np.int32) + primitive_labels.astype(np.int32), 
                            minlength=ncentroids ** 2).reshape(ncentroids, ncentroids) 


'''Hungarian Matching'''
row_ind, col_ind = linear_sum_assignment(histogram.max() - histogram)
acc_class = histogram[row_ind, col_ind] / (histogram.sum(1) + 1e-10)* 100.

all_acc = histogram[row_ind, col_ind].sum() / histogram.sum()*100.
m_acc = np.nanmean(histogram[row_ind, col_ind] / histogram.sum(1))*100
hist_new = np.zeros((ncentroids, ncentroids))
for idx in range(ncentroids):
    hist_new[:, idx] = histogram[:, col_ind[idx]]

'''Final Metrics'''
tp = np.diag(hist_new)
fp = np.sum(hist_new, 0) - tp
fn = np.sum(hist_new, 1) - tp
IoUs = tp / (tp + fp + fn + 1e-8)
m_iou = np.nanmean(IoUs)*100
print(m_iou)
print("ff")