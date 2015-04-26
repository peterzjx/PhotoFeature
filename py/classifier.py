from sklearn import svm, metrics, linear_model, ensemble
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def one_iter(dataframe):
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    column_len = len(dataframe.columns)
    data_f = dataframe.iloc[:, 1: column_len-1]
    target_f = dataframe.loc[:, 'label']
    data = data_f.as_matrix()
    target = target_f.as_matrix()
    n_samples = len(target)
    # Create a classifier: a support vector classifier
    # classifier = svm.SVC(gamma=0.002, class_weight={0:1.3})
    # classifier = svm.SVC(kernel='poly', degree=2, gamma=0.001, class_weight={0:1.3})
    # classifier = linear_model.LogisticRegression(class_weight={0:1.3})
    classifier = ensemble.RandomForestClassifier(class_weight={0:1.3}, max_features=5, n_estimators=100)

    training_percentage = 0.5
    # We learn the digits on the first half
    classifier.fit(data[:n_samples * training_percentage], target[:n_samples * training_percentage])

    # Now predict the value on the second half:
    expected = target[n_samples * training_percentage:]
    predicted = classifier.predict(data[n_samples * training_percentage:])

    predict_label = np.array([-1] * int(n_samples* training_percentage))
    predict_label = np.concatenate((predict_label, predicted))
    print len(predict_label)
    dataframe['predict'] = predict_label
    dataframe.to_csv("result.csv")

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(importances)):
        print("%d. feature %d (%s) (%f)" % (f + 1, indices[f], dataframe.columns[indices[f]+1], importances[indices[f]]))



data_joe = pd.read_csv("../data/joe/features.csv")
data_joe['label'] = 0

data_pro = pd.read_csv("../data/pro2/features.csv")
data_pro['label'] = 1

# dataframe = pd.read_csv("../data/full_data.csv")
dataframe = data_joe.append(data_pro, ignore_index=True)

one_iter(dataframe)

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
#
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#
# X = dataframe.loc[:, 'saturation']
# Y = dataframe.loc[:, 'saturation_diff']
# ax.scatter(X, Y, c=dataframe.loc[:, 'label'], cmap=cm_bright)
#
# ax.set_xlabel('contrast_diff')
# ax.set_ylabel('brightness_diff')

# plt.show()


# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))

# print metrics.accuracy_score(expected, predicted)

