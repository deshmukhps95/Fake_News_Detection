from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=1,
            norm="l2",
            encoding="utf-8",
            ngram_range=(1, 3),
            stop_words=None,
            max_features=None,
            binary=False,
        )

    def get_features(self, sentence):
        return self.vectorizer.fit_transform(sentence).toarray()
