import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenize
    tokenized = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set()
    for words in tokenized:
        vocab.update(words)

    # 3. Priors
    priors = {
        0: np.mean(labels == 0),
        1: np.mean(labels == 1)
    }

    # 4. Word probabilities (MLE)
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        # collect words of class c
        words_c = []
        for i in range(len(labels)):
            if labels[i] == c:
                words_c.extend(tokenized[i])

        total_words = len(words_c)

        for word in vocab:
            count = words_c.count(word)
            word_probs[c][word] = count / total_words if total_words > 0 else 0

    # 5. Prediction
    test_words = test_email.split()

    scores = {}

    for c in [0, 1]:
        score = priors[c]

        for word in test_words:
            if word in word_probs[c]:
                score *= word_probs[c][word]
            else:
                score *= 0  # no smoothing

        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    # 1. Load data
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Distance function
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. Predict function
    def predict(X_train, y_train, X_test):
        predictions = []

        for x_test in X_test:
            distances = []

            for i in range(len(X_train)):
                dist = euclidean_distance(x_test, X_train[i])
                distances.append((dist, y_train[i]))

            # sort by distance
            distances.sort(key=lambda x: x[0])

            # take k nearest
            neighbors = distances[:k]

            # majority vote
            labels = [label for _, label in neighbors]
            values, counts = np.unique(labels, return_counts=True)
            pred = values[np.argmax(counts)]

            predictions.append(pred)

        return np.array(predictions)

    # 5. Predictions
    train_preds = predict(X_train, y_train, X_train)
    test_preds = predict(X_train, y_train, X_test)

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds
