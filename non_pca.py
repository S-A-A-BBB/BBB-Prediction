import pandas as pd
from sklearn.decomposition import KernelPCA
import sys

sys.path.append("..")
from preprocessing import samp

dataset, labels = samp.smote()

# All the kernels
# rbf,sigmoid,poly,linear kernel
def pca():
    global dataset
    # this is a non linear
    non_pca = KernelPCA(kernel='linear', n_jobs=10)
    non_pca.fit(dataset)
    train_x = non_pca.transform(dataset)
    col = []
    # read about what features used if found.
    for i in range(train_x.shape[1]):
        col.append('Feature_' + str(i))
    df1 = pd.DataFrame(train_x, columns=col)
    return df1, labels
