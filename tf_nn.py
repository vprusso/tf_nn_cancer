from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

# Preparing the data:
data_file_name = 'breast-cancer-wisconsin.data.txt'

first_line = "id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses,class"
with open(data_file_name, "r+") as f:
	content = f.read()
	f.seek(0, 0)
	f.write(first_line.rstrip('\r\n') + '\n' + content)

df = pd.read_csv(data_file_name)

df.replace('?', np.nan, inplace = True)
df.dropna(inplace=True)
df.drop(['id'], axis = 1, inplace = True)

df['class'].replace('2',0, inplace = True)
df['class'].replace('4',1, inplace = True)

df.to_csv("combined_data.csv", index = False)

# Data sets
CANCER_TRAINING = "cancer_training.csv"
CANCER_TEST = "cancer_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=CANCER_TRAINING,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=CANCER_TEST,
                                                   target_dtype=np.int)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/cancer_model")

# Fit model.
classifier.fit(x=training_set.data, 
               y=training_set.target, 
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new cancer tumor samples.
new_samples = np.array(
    [[5,10,8,4,7,4,8,11,2], [5,1,1,1,1,1,1,1,2]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))