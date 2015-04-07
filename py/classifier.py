from sklearn import datasets, svm, metrics
import pandas as pd
import numpy as np

dataframe = pd.read_csv("../data/full_data.csv")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
data_f = dataframe.loc[:, 'f1':'f6']
target_f = dataframe.loc[:, 'label']
data = data_f.as_matrix()
target = target_f.as_matrix()
n_samples = len(target)
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half
classifier.fit(data[:n_samples * 2 / 3], target[:n_samples * 2 / 3])

# Now predict the value on the second half:
expected = target[n_samples * 2 / 3:]
predicted = classifier.predict(data[n_samples * 2 / 3:])
#
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))