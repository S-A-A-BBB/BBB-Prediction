# BBB-Prediction
1- This code can be excused to  predict BBB permeability using DNN model and CNN model. The dataset is obtained from Wang et. al 2018.

2- 1D, 2D, 3D descriptors (and) two types of Fingerprints Hashed and MACCS 166 were claculated using Alvascience and Ochem. 

3- This dataset dimensionality is reduced using Kernel PCA. This is done under the folder: preprocessing -> non_pca.py

4- The dataset is resampled with SMOTE. This is done under the folder: preprocessing -> samp.py


5- After dataset is preprocessed, its read by the DNN_KFOLD.py file and the CNN.py file. (DNN refers to the feed forward deep neural networks FFDNN, and CNN refers to Convultional neural networks CNN)
Both CNN and DNN models are trained with k-fold validation.

6- Accuracy, ROC, and intervals are obtained directly from the code.

7- After the files are run, the summary of results is found under the Report folder.

