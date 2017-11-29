import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    data = pd.read_csv("data/csv/kddcup.data_10_percent_corrected.csv", header=0)

    data = data.drop(["Unnamed: 0"], axis=1)
    labels = data["label_cat"].values

    data = data.drop(["label"], axis=1)
    data = data.drop(["label_cat"], axis=1)

    col = list(data.columns)

    features = data.values

    regr = RandomForestRegressor(n_jobs=-1)
    regr.fit(features, labels)

    importances = regr.feature_importances_
    print(importances)
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(features.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, data.columns.values[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(features.shape[1]), data.columns.values[indices], rotation=90)
    plt.xlim([-1, 7])
    plt.show()

    col = list(data.columns.values[indices[:7]])
    out = pd.DataFrame(col)

    out.to_csv("data/csv/kdd_feature_selected.csv", date_format='%Y-%m-%d %H:%M:%S')