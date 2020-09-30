import xgboost as xgb
from preprocessing import non_pca
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report, roc_curve, \
    roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

dataset, labels, dataset_external, labels_external = non_pca.pca()
import numpy as np
dataset=dataset.values
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(dataset):
    pred_lbl = []
    errors = []
    pred_lbl_external = []
    X_train, X_test = dataset[train_index, :], dataset[test_index, :]
    Y_train, y_test = labels[train_index], labels[test_index]
    model = xgb.XGBClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    for k in pred:
        if k < 0.4:
            pred_lbl.append(0)
        else:
            pred_lbl.append(1)

    interval = [1.64, 1.96, 2.33, 2.58]
    name = 'XGBOOST NON PCA'
    conf = confusion_matrix(y_test, pred_lbl)
    tn, fp, fn, tp = conf.ravel()
    acc = accuracy_score(y_test, pred_lbl)
    error = 1 - acc
    errors.append(error)
    sensitivity = tp / (tp + fn)
    specitivity = tn / (tn + fp)
    clas = classification_report(y_test, pred_lbl)
    mcc = matthews_corrcoef(y_test, pred_lbl)
    external_lbl = model.predict(dataset_external)
    for kkk in external_lbl:
        if kkk < 0.4:
            pred_lbl_external.append(0)
        else:
            pred_lbl_external.append(1)

    acc_external = accuracy_score(labels_external, pred_lbl_external)
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
        f.write('\n Accuracy on External Set :- ' + str(
            acc_external) + '\n' + 'Accuracy on Test Set :- ' + str(acc) + '\n' + 'Sensitivity :- ' + str(
            sensitivity) + '\n' + 'Specitivity :- ' + str(specitivity) + '\n' + 'ROC SCORE :- ' + str(
            roc_auc) + '\n' + 'Confusion Matrix :- \n' + str(conf) + '\n' + 'Classification Report :- \n' + str(
            clas) + '\n MCC :- ' + str(mcc) + '\n\n\n')

        for jj in inter_values.items():
            f.write('INTERVAL IS ' + str(jj[0]) + ' VALUE IS ' + str(jj[1]) + '\n')

    with open('Report/' + name + '_evalution.txt', 'a') as f:
        f.write('MEAN ERROR IS ' + str(sum(errors) / len(errors)))

    # .0967 - .119
