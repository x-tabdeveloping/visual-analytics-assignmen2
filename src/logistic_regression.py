from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_data


def main():
    print("Loading and preprocessing data.")
    X_train, y_train, X_test, y_test = load_data()

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


if __name__ == "__main__":
    main()
