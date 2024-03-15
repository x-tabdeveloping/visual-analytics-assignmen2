from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import cifar10

from utils import class_names, flatten_images, greyscale_images

print("Loading and preprocessing data.")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = greyscale_images(X_train)
X_test = greyscale_images(X_test)
X_train = flatten_images(X_train)
X_test = flatten_images(X_test)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_train = [class_names[y] for y in y_train]
y_test = [class_names[y] for y in y_test]

print("Training Logistic Regression classifier.")
classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
classifier.fit(X_train, y_train)

print("Evaluating.")
report = classification_report(y_test, classifier.predict(X_test))
print(report)

print("Saving report.")
out_path = Path("out")
out_path.mkdir(exist_ok=True)
with open(out_path.joinpath("logistic_regression_report.txt"), "w") as report_file:
    report_file.write(report)
