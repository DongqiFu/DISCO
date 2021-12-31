import pickle as pkl
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mufasa_model import mufasa, mufasa_classifier_20, CNN_1d 
from joblib import dump, load

X = pkl.load(open("feature_matrix.pkl", "rb"))
y = pkl.load(open("label_matrix.pkl", "rb"))


# - 1. PCA - #

# --- standardization --- #
X_standard = scale(X)

# --- variance-covariance matrix --- #
covariance_X = np.cov(X_standard.T)

# --- eigen pairs --- #
eigen_values, eigen_vectors = np.linalg.eig(covariance_X)
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

# --- select 80% eigenvalues --- #
total_values = np.sum(eigen_values)
current_values = 0
num_selected_eigenvectors = 0
while current_values <= total_values * 0.8:
    current_values += eigen_values[num_selected_eigenvectors]
    num_selected_eigenvectors += 1
print("number of selected eigen-vectors / new dimension: " + str(num_selected_eigenvectors))

selected_idx = np.arange(num_selected_eigenvectors)
selected_eigen_vectors = eigen_vectors[:, selected_idx]

# --- dimension reduced X --- #
X_new = np.dot(X_standard, selected_eigen_vectors)

# - 2. accuracy of new models trained on truncated data - #
k = 10
for i in range(k):
    print('---------> ' + str(i))
    # --- Option 1: MuFasa --- #
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42)
    print(X_train.shape)
    mf = mufasa_classifier_20()
    mf.train_mufasa(X_train, y_train)
    #dump(mf, 'mufasa_20.joblib')
    score, pred = mf.test_mufasa(X_test, y_test, thre = 0.5)
    print('------------------> ' + str(score))

    # --- Option 2: MLP --- #
    # X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(32, 2), activation='relu').fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # print('------------------> ' + str(score))