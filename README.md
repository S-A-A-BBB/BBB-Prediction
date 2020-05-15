# BBB-Prediction
1- This code can be excused to  predict BBB permeability using DNN model and CNN model. The dataset is found under the folder Dataset: ORGINAL DATASET WITH FINGERPRINTS.csv

2- The original dataset is a high dimensional dataset with 2350 records and around 7000 descriptors and fingerprints.

3- This dataset dimensionality is reduced using Kernel PCA. This is done under the folder: preprocessing -> non_pca.py

4- The dataset is resampled with SMOTE. This is done under the folder: preprocessing -> samp.py


5- After dataset is preprocessed, its read by the DNN_KFOLD.py file and the CNN.py file. (DNN refers to the feed forward deep neural networks FFDNN, and CNN refers to Convultional neural networks CNN)
Both CNN and DNN models are trained with k-fold validation.

6- Accuracy, ROC, and intervals are obtained directly from the code.

7- After the files are run, the summary of results is found under the Report folder.

