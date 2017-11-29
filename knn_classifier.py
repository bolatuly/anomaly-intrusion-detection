import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

def min_max_scaler(features):
    min_max = MinMaxScaler()
    return min_max.fit_transform(features)

def kdd_cup_classifier():
    data = pd.read_csv("data/csv/kddcup.data_10_percent_corrected.csv")
    col_features = pd.read_csv("data/csv/kdd_feature_selected.csv")

    col_features = col_features.drop(["Unnamed: 0"], axis=1)

    labels = data["label_cat"].values
    features = data[col_features['0'].values].values

    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    scores = []

    for train_index, test_index in cv.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        my_classifier = KNeighborsClassifier(n_jobs=-1)
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)
        scores.append(accuracy_score(y_test, predictions))

    return (scores[0] + scores[1]+scores[2])/3


def kdd_cup_plus_classifier():
    # selected columns
    col_features = pd.read_csv("data/csv/kdd+_feature_selected.csv")
    col_features = col_features.drop(["Unnamed: 0"], axis=1)

    # train
    train = pd.read_csv("data/csv/KDDTrain+.csv")
    train_labels = train["label_cat"].values
    train_features = train[col_features['0'].values].values

    # test
    test = pd.read_csv("data/csv/KDDTest+.csv")
    #test = pd.read_csv("data/csv/KDDTest-21.csv")
    test_labels = test["label_cat"].values
    test_features = test[col_features['0'].values].values

    # scaling
    train_features = min_max_scaler(train_features)
    test_features = min_max_scaler(test_features)

    my_classifier = KNeighborsClassifier(n_jobs=-1)

    # scaling
    my_classifier.fit(train_features, train_labels)

    predictions = my_classifier.predict(test_features)

    return accuracy_score(test_labels, predictions)


if __name__ == '__main__':
    score = kdd_cup_classifier()
    #score = kdd_cup_plus_classifier()
    print(score)
