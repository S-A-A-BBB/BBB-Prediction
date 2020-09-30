from sklearn.ensemble import RandomForestClassifier
from preprocessing import non_pca
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, classification_report, roc_curve, \
    roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
errors=[]
dataset, labels, dataset_external, labels_external = non_pca.pca()
name = 'RANDOM FOREST'

totals_splits = 10
count = 0
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(dataset):
    X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)
    pred_lbl = rfc.predict(X_test)
    pred_lbl_train = rfc.predict(X_train)
    conf = confusion_matrix(y_test, pred_lbl)
    tn, fp, fn, tp = conf.ravel()
    interval = [1.64, 1.96, 2.33, 2.58]
    external_lbl = rfc.predict(dataset_external)
    acc_external = accuracy_score(labels_external, external_lbl)
    conf_train = confusion_matrix(y_test, pred_lbl)
    tn_train, fp_train, fn_train, tp_train = conf_train.ravel()
    sensitivity_train = tp_train / (tp_train + fn_train)
    specitivity_train = tn_train / (tn_train + fp_train)
    acc = accuracy_score(y_test, pred_lbl)
    error = 1 - acc
    errors.append(error)
    count = count + 1
    acc_train = accuracy_score(y_train, pred_lbl_train)
    sensitivity = tp / (tp + fn)
    specitivity = tn / (tn + fp)
    clas = classification_report(y_test, pred_lbl)
    mcc = matthews_corrcoef(y_test, pred_lbl)

    fpr, tpr, thresholds = roc_curve(y_test, pred_lbl)
    roc_auc = roc_auc_score(y_test, pred_lbl)
    plt.plot(fpr, tpr, label='test ROC curve (area = %0.3f)' % roc_auc)

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
        f.write('Accuracy on Train Set :- ' + str(acc_train) + '\n' + 'Accuracy on External Set :- ' + str(
            acc_external) + '\n' + 'Sensitivity on train set :- ' + str(
            sensitivity_train) + '\n' + 'Specitivity on train set :- ' + str(
            specitivity_train) + '\n' + 'Accuracy on Test Set :- ' + str(acc) + '\n' + 'Sensitivity :- ' + str(
            sensitivity) + '\n' + 'Specitivity :- ' + str(specitivity) + '\n' + 'ROC SCORE :- ' + str(
            roc_auc) + '\n' + 'Confusion Matrix :- \n' + str(conf) + '\n' + 'Classification Report :- \n' + str(
            clas) + '\n MCC :- ' + str(mcc) + '\n\n\n')

        if totals_splits - 1 == count:
            for jj in inter_values.items():
                f.write('INTERVAL IS ' + str(jj[0]) + ' VALUE IS ' + str(jj[1]) + '\n')
                f.write('ERROR IS : '+str(sum(errors)/len(errors)))
## - 92.04
## - 100 and 0


# [0.908 - 0.952]