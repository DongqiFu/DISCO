import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

X = pkl.load(open("preprocessed-dataset/feature_matrix.pkl", "rb"))
y = pkl.load(open("preprocessed-dataset/label_matrix.pkl", "rb"))

# num_articles = 44898   # 0 - 44897
# num_fake_news = 23481  # 0 - 23480
# num_real_news = 21417  # 23481 - 44897
# num_feature_dim = 300

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(32, 2), activation='relu').fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)

dump(clf, 'trained-classifier/saved_mlp.joblib')
