from sklearn import svm, metrics, linear_model, ensemble
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visual(dataframe):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    X = dataframe.loc[:, 'dark_black']
    Y = dataframe.loc[:, 'saturation']
    Z = dataframe.loc[:, 'spacial_std']
    ax.scatter(X, Y,Z, c=dataframe.loc[:, 'label'], cmap=cm_bright)

    ax.set_xlabel('dark_black')
    ax.set_ylabel('saturation')
    ax.set_zlabel('spacial_std')

    plt.show()


def one_iter(dataframe):
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    column_len = len(dataframe.columns)
    data_f = dataframe.iloc[:, 1: column_len-1]
    target_f = dataframe.loc[:, 'label']
    data = data_f.as_matrix()
    target = target_f.as_matrix()
    n_samples = len(target)
    # Create a classifier: a support vector classifier
    # classifier = svm.SVC(gamma=0.001, class_weight={0:1.3})
    # classifier = svm.SVC(kernel='poly', degree=2, gamma=0.001, class_weight={0:1.3})
    # classifier = linear_model.LogisticRegression(class_weight={0:1.3})
    classifier = ensemble.RandomForestClassifier(class_weight={0: 1.3}, max_features=5, n_estimators=100)

    training_percentage = 0.5
    # We learn the digits on the first half
    classifier.fit(data[:n_samples * training_percentage], target[:n_samples * training_percentage])

    # Now predict the value on the second half:
    expected = target[n_samples * training_percentage:]
    predicted = classifier.predict(data[n_samples * training_percentage:])

    expected = target[:n_samples * training_percentage]
    predicted = classifier.predict(data[:n_samples * training_percentage])

    predict_label = np.array([-1] * int(n_samples* training_percentage))
    predict_label = np.concatenate((predict_label, predicted))
    print len(predict_label)
    dataframe['predict'] = predict_label
    # dataframe.to_csv("result.csv")
    #
    # importances = classifier.feature_importances_
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(expected, predicted)))
    # return np.array(importances), metrics.accuracy_score(expected, predicted)
    return metrics.accuracy_score(expected, predicted)


# data_joe = pd.read_csv("../data/joe/features.csv")
# data_joe['label'] = 0
#
# data_pro = pd.read_csv("../data/pro2/features.csv")
# data_pro['label'] = 1
#
# dataframe = data_joe.append(data_pro, ignore_index=True)
dataframe = pd.read_csv("../data/all/features_full.csv")

# visual(dataframe)
# total_imp = np.array([0] * 19)
total_acc = 0
for _ in range(100):
    res = one_iter(dataframe)
    # total_imp = total_imp + res[0]
    # total_acc = total_acc + res[1]
    total_acc = total_acc + res
#
# # indices = np.argsort(total_imp)[::-1]
# #
# # # Print the feature ranking
# # print("Feature ranking:")
# #
# # for f in range(len(total_imp)):
# #     print("%d. feature %d(%s) (%f)" % (f + 1, indices[f], dataframe.columns[indices[f]+1], total_imp[indices[f]]))
#
print total_acc/100



