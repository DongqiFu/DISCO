import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import re
import pickle as pkl
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from utils import power_iteration
from mufasa_model import mufasa, mufasa_classifier, CNN_1d 


pretrained_model = KeyedVectors.load_word2vec_format('fake-and-real-news-dataset/GoogleNews-vectors-negative300.bin', binary=True)

# ------ Option 1 MuFasa ------ #
mf = load('./mufasa.joblib')

# ------ Option 2 MLP ------ #
# clf = load('saved_mlp.joblib')

fake_data = pd.read_csv("fake-and-real-news-dataset/Fake.csv")
true_data = pd.read_csv("fake-and-real-news-dataset/True.csv")
data = pd.concat([fake_data, true_data], axis=0, ignore_index = True)
investigating_news_list = [15173]
for idx in investigating_news_list:  # - the idx-th news article - #
    print('======> Preprocessing ' + str(idx) + ' news.')

    X = pkl.load(open("feature_matrix.pkl", "rb"))
    prev_prob = [[1.0 - mf.predict_prob(X[idx]), mf.predict_prob(X[idx])]]
    print(prev_prob)
 

    word_2_gain = dict()

    review = re.sub("[^a-zA-Z]", " ", data["text"][idx])
    review = review.lower()
    review = review.split()
    review = [WordNetLemmatizer().lemmatize(word) for word in review if word not in stopwords.words("english")]

    # - Get valid tokens - #
    num_nodes = 0
    id_2_word = dict()
    word_2_id = dict()
    id_2_vec = dict()
    appeared_nodes = set()
    for word in review:
        if word not in word_2_id.keys() and word not in appeared_nodes:  # - find a new word - #
            try:
                vec = pretrained_model[word]  # - 300 dimension - #
                word_2_id[word] = num_nodes
                id_2_word[num_nodes] = word
                id_2_vec[num_nodes] = vec
                num_nodes += 1
                appeared_nodes.add(word)
            except:
                print('============> ' + word + ' could not be found in the Google pre-trained model.')
                appeared_nodes.add(word)

    print("============> number of words: " + str(num_nodes))

    # - Remove each word - #
    for rm_idx in range(num_nodes):
        print("============> current word idx: " + str(rm_idx))

        rm_word = id_2_word[rm_idx]
        id_2_word.pop(rm_idx)
        word_2_id.pop(rm_word)

        # - Construct graph adjacency matrix - #
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for j in range(len(review) - 2):  # - size window is 3 - #
            word_x = review[j]
            word_y = review[j + 1]
            word_z = review[j + 2]
            if word_x in word_2_id.keys() and word_y in word_2_id.keys():
                adj_matrix[word_2_id[word_x]][word_2_id[word_y]] = 1
                adj_matrix[word_2_id[word_y]][word_2_id[word_x]] = 1
            if word_x in word_2_id.keys() and word_z in word_2_id.keys():
                adj_matrix[word_2_id[word_x]][word_2_id[word_z]] = 1
                adj_matrix[word_2_id[word_z]][word_2_id[word_x]] = 1
            if word_y in word_2_id.keys() and word_z in word_2_id.keys():
                adj_matrix[word_2_id[word_y]][word_2_id[word_z]] = 1
                adj_matrix[word_2_id[word_z]][word_2_id[word_y]] = 1

        # - Get graph embedding - #
        h_matrix = np.zeros((num_nodes, 300))
        p_matrix = np.zeros((num_nodes, num_nodes))
        for j in range(num_nodes):
            if j != rm_idx:
                ppr = np.zeros((num_nodes,))
                ppr[j] = 1
                ppr = power_iteration(ppr, adj_matrix)
                p_matrix[j:] = ppr
                h_matrix[j:] = id_2_vec[j]
        z_matrix = np.dot(p_matrix, h_matrix)
        z_vec = np.sum(z_matrix, axis=0)
        print(z_vec.shape)
        now_prob = [[1.0 - mf.predict_prob(z_vec),  mf.predict_prob(z_vec)]]

        # print("============> word: " + rm_word + ", previous: " + str(prev_prob) + ", now: " + str(now_prob))

        gain = 0  # the change of the "probability of accurate detection"
        if idx >= 23481:  # real news
            gain = now_prob[0][1] - prev_prob[0][1]
        else:
            gain = now_prob[0][0] - prev_prob[0][0]

        word_2_gain[rm_word] = gain

    print('======> Preprocessed ' + str(idx) + ' news.')

    for k, v in sorted(word_2_gain.items(), key=lambda item: item[1]):
        print("word: " + str(k) + "," + "probability gain: " + str(v))