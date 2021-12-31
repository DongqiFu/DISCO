import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import re
from utils import power_iteration
import pickle as pkl

pretrained_model = KeyedVectors.load_word2vec_format('fake-and-real-news-dataset/GoogleNews-vectors-negative300.bin', binary=True)

fake_data = pd.read_csv("fake-and-real-news-dataset/Fake.csv")
true_data = pd.read_csv("fake-and-real-news-dataset/True.csv")
fake_data["Label"] = 0
true_data["Label"] = 1
data = pd.concat([fake_data, true_data], axis=0, ignore_index = True)
# data = data.sample(frac = 1).reset_index(drop = True)
data.drop(["title", "subject", "date"], axis=1, inplace=True)
print(data)
# print(data.text)
# print(data["text"][0])
# print(len(data))


X = data.drop(["Label"], axis=1)
y = data["Label"]

num_docs = len(data)
feature_matrix = np.zeros((num_docs, 300))
label_matrix = np.zeros(num_docs)

for i in range(num_docs):
    print('======> Preprocessing ' + str(i) + ' news.')

    # print("Original text ==> ", X["text"][i])
    review = re.sub("[^a-zA-Z]", " ", X["text"][i])
    review = review.lower()
    # print("Original text (cleaned) ==> ", review)
    review = review.split()
    review = [WordNetLemmatizer().lemmatize(word) for word in review if word not in stopwords.words("english")]
    # print("Original text (tokens) ==> ", review)

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
                print('============> ' + word + ' could not be found in the Google pretrained model.')
                appeared_nodes.add(word)

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
        ppr = np.zeros((num_nodes, ))
        ppr[j] = 1
        ppr = power_iteration(ppr, adj_matrix)
        p_matrix[j:] = ppr

        h_matrix[j:] = id_2_vec[j]
    z_matrix = np.dot(p_matrix, h_matrix)
    z_vec = np.sum(z_matrix, axis=0)
    print('======> Preprocessed ' + str(i) + ' news.')
    # print(z_vec)
    # print(y[i])
    feature_matrix[i:] = z_vec
    label_matrix[i] = y[i]

pkl.dump(feature_matrix, open('feature_matrix.pkl', 'wb'))
pkl.dump(label_matrix, open('label_matrix.pkl', 'wb'))