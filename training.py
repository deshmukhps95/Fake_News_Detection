import time
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from preprocessing import process
from feaure_extraction import FeatureExtractor


class Trainer:
    train_file_path = r"D:\CodeBase\fake-news-detection\Fake_News_Detection\train.csv"
    test_file_path = r"D:\CodeBase\fake-news-detection\Fake_News_Detection\test.csv"
    val_file_path = r"D:\CodeBase\fake-news-detection\Fake_News_Detection\valid.csv"

    def __init__(self):
        self.model = SVC(kernel="linear")
        self.feature_extractor = FeatureExtractor()

    def train(self):
        training_data = pd.read_csv(self.train_file_path)
        test_data = pd.read_csv(self.test_file_path)

        print(training_data.head(5))
        # pre-process the training data
        training_data["Statement"] = training_data["Statement"].map(process)
        print(training_data.head(5))

        # pre-process the test data
        test_data["Statement"] = test_data["Statement"].map(process)

        x_train = training_data["Statement"].values
        y_train = training_data["Label"].values

        x_test = test_data["Statement"].values
        y_test = test_data["Label"].values

        training_features = self.feature_extractor.get_features(x_train)
        training_labels = y_train
        print(f"Size of test sentences: {training_features.shape}")

        test_features = self.feature_extractor.vectorizer.transform(x_test).toarray()
        test_labels = y_test
        print(f"Size of test sentences: {test_features.shape}")

        print("Training started")
        t = time.time()
        self.model.fit(training_features, training_labels)
        print(f"Training time: {(time.time()-t)/60.0} minutes")

        prediction_labels = self.model.predict(test_features)
        print(
            f"Test Accuracy: {metrics.accuracy_score(test_labels, prediction_labels)}"
        )

        print(
            metrics.classification_report(
                test_labels, prediction_labels, target_names=test_data["Label"].unique()
            )
        )

    def validate(self):
        validation_data = pd.read_csv(self.val_file_path)
        # pre-process the validation data
        validation_data["Statement"] = validation_data["Statement"].map(process)

        x_val = validation_data["Statement"].values
        y_val = validation_data["Label"].values

        validation_features = self.feature_extractor.vectorizer.transform(
            x_val
        ).toarray()
        validation_labels = y_val

        val_prediction = self.model.predict(validation_features)
        print(
            f"Validation Accuracy: {metrics.accuracy_score(validation_labels, val_prediction)}"
        )

    def save_trained_model(self, model_checkpoint_path):
        file = open(model_checkpoint_path, "wb")
        pickle.dump(self.model, file, protocol=3)
        pickle.dump(self.feature_extractor.vectorizer, file, protocol=3)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    # trainer.validate()
    trainer.save_trained_model("models/fake_news_model.pkl")
