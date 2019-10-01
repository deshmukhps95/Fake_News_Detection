import pickle
from preprocessing import process
from feaure_extraction import FeatureExtractor

feature_extractor = FeatureExtractor()


def detecting_fake_news(model_checkpoint, sentence):
    file = open(model_checkpoint, "rb")
    model = pickle.load(file)
    vectorizer = pickle.load(file)
    features = vectorizer.transform([sentence]).toarray()
    prediction = model.predict(features)
    # probability = model.predict_proba([sentence])
    print("The given statement is ", prediction[0])
    # print("The truth probability score is ", probability[0][1])


if __name__ == "__main__":
    sentence = input("Please enter the news text you want to verify: ")
    print("You entered: " + str(sentence))
    pre_processed_sentence = process(sentence)
    detecting_fake_news("models/fake_news_model.pkl", pre_processed_sentence)
