import sys
from keras import layers, optimizers, Sequential

sys.path.append("..")
from preprocessing import non_pca
from sklearn.model_selection import KFold
import time

from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report, roc_curve, \
    roc_auc_score

import matplotlib.pyplot as plt
import numpy as np
import keras

# Deep neural network
# for non pca
dataset, labels = non_pca.pca()

# CNN 1d takes 3d Data so we change the shape of data where each row now is a 2d Array itself
# shape of data N,1,features where N represents number of instanes and features represent descriptors and fingerprints

dataset = np.expand_dims(dataset, axis=1)


# Sequential and other is functional API
def model_build():
    model = Sequential()
    model.add(layers.Conv1D(1024, 1, activation='tanh', input_shape=(1, dataset.shape[2])))
    model.add(layers.Conv1D(512, 1, activation='tanh'))
    model.add(layers.Conv1D(512, 1, activation='tanh'))
    model.add(layers.Conv1D(256, 1, activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.01, beta_1=0.34, beta_2=0.76, decay=0.4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


totals_splits = 10
name = 'CNN_KPCA9 '
kf = KFold(n_splits=totals_splits, random_state=None, shuffle=True)
count = 0
errors = []

# DIFFERENT SPLITS FROM KFOLD WHERE EACH TIME 9 splits for training 1 for testing
for train_index, test_index in kf.split(dataset):
    model_start_time = time.time()
    X_train, X_test = dataset[train_index], dataset[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    model = model_build()
    call = keras.callbacks.ModelCheckpoint('models/keras_model' + str(count) + '.h5', save_best_only=True)
    model.fit(X_train, y_train, epochs=100, batch_size=800, callbacks=[call], validation_data=(X_test, y_test))
    pred_lbl = []
    pred = model.predict(X_test)
    for k in pred:
        if k < 0.5:
            pred_lbl.append(0)
        else:
            pred_lbl.append(1)

    print('.............' + str(count) + ' Fold Done..........')
    count = count + 1
    keras.backend.clear_session()
    time.sleep(5)

    interval = [1.64, 1.96, 2.33, 2.58]

    conf = confusion_matrix(y_test, pred_lbl)
    tn, fp, fn, tp = conf.ravel()
    acc = accuracy_score(y_test, pred_lbl)
    error = 1 - acc
    errors.append(error)
    sensitivity = tp / (tp + fn)
    specitivity = tn / (tn + fp)
    clas = classification_report(y_test, pred_lbl)
    mcc = matthews_corrcoef(y_test, pred_lbl)

    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC/' + name + '_Roc_plot.png')
    plt.close()
    inter_values = {}

    for i in interval:
        print(len(y_test))
        interval = i * np.sqrt((error * (1 - error)) / len(y_test))
        print('INTERVAL :', i, 'VALUE : ', interval)
        inter_values[i] = interval

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(pred_lbl), len(pred_lbl))
        if len(np.unique(y_test[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_test)[indices], np.array(pred_lbl)[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))

    with open('Report/' + name + '_evalution.txt', 'a') as f:
        f.write('Accuracy on Test Set :- ' + str(acc) + '\n' + 'Sensitivity :- ' + str(
            sensitivity) + '\n' + 'Specitivity :- ' + str(specitivity) + '\n' + 'ROC SCORE :- ' + str(
            roc_auc) + '\n' + 'Confusion Matrix :- \n' + str(conf) + '\n' + 'Classification Report :- \n' + str(
            clas) + '\n MCC :- ' + str(mcc) + '\n\n\n' + str(
            "Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
                confidence_lower, confidence_upper) + '\n\n'))

        if totals_splits - 1 == count:
            for jj in inter_values.items():
                f.write('INTERVAL IS ' + str(jj[0]) + ' VALUE IS ' + str(jj[1]) + '\n')

with open('Report/' + name + '_evalution.txt', 'a') as f:
    f.write('MEAN ERROR IS ' + str(sum(errors) / len(errors)))
