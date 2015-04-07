import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import pandas as pd
import numpy as np
# The digits dataset
# digits = datasets.load_digits()
#
# # The data that we are interested in is made of 8x8 images of digits, let's
# # have a look at the first 3 images, stored in the `images` attribute of the
# # dataset.  If we were working from image files, we could load them using
# # pylab.imread.  Note that each image must have the same size. For these
# # images, we know which digit they represent: it is given in the 'target' of
# # the dataset.
# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# print data
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