from tkinter import Tk, Label, Button
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import re
import pickle as pkl
import numpy as np
from joblib import dump, load
from utils import power_iteration, track_trans


def process_text(input_text, pretrained_model):
    review = re.sub("[^a-zA-Z]", " ", input_text)
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
                # print('============> ' + word + ' could not be found in the Google pre-trained model.')
                appeared_nodes.add(word)

    # print("============> number of words: " + str(num_nodes))

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
        ppr = np.zeros((num_nodes,))
        ppr[j] = 1
        ppr = power_iteration(ppr, adj_matrix)
        p_matrix[j:] = ppr
        h_matrix[j:] = id_2_vec[j]
    z_matrix = np.dot(p_matrix, h_matrix)
    z_vec = np.sum(z_matrix, axis=0)  # pooling: sum to row

    return num_nodes, id_2_word, word_2_id, adj_matrix, h_matrix, p_matrix, z_vec


def prediction(z_vec, clf):
    prob = clf.predict_proba([z_vec])
    fake_prob = prob[0][0]
    real_prob = prob[0][1]

    label = clf.predict([z_vec])

    return label, prob


def misleading_top_n_words(label, prob, num_nodes, adj_matrix, p_matrix, h_matrix, id_2_word):
    word_2_misleading_degree = dict()

    for rm_idx in range(num_nodes):
        new_adj_matrix = adj_matrix.copy()
        new_adj_matrix[:, rm_idx] = 0
        new_adj_matrix[rm_idx, :] = 0
        new_p_matrix = track_trans(adj_matrix, new_adj_matrix, p_matrix)
        z_matrix = np.dot(new_p_matrix, h_matrix)
        z_vec = np.sum(z_matrix, axis=0)
        new_prob = clf.predict_proba([z_vec])

        misleading_degree = 0
        if label == 0:
            misleading_degree = new_prob[0][0] - prob[0][0]
        else:
            misleading_degree = new_prob[0][1] - prob[0][1]

        rm_word = id_2_word[rm_idx]
        word_2_misleading_degree[rm_word] = misleading_degree

    ranking = sorted(word_2_misleading_degree.items(), key=lambda item: item[1], reverse=True)

    return ranking

print("Starting DISCO ...")

print("Loading the input article ...")
test_text = "Leave it to our Community Organizer In Chief to bully nuns who’ve committed their lives to helping the poor in our country over a contraception mandate If Obama’s rules apply to these nuns why don’t they apply to the tens of millions who are living in the United States illegally?  Pope Francis paid a short visit to the Little Sisters of the Poor community in Washington, D.C. on Wednesday to support them in their court case over the contraception mandate, the Vatican s spokesman revealed. It was a  short visit that was not in the program,  Father Federico Lombardi, director of the Holy See Press Office, said at an evening press conference during the papal visit to the nation s capital. This is a sign, obviously, of support for them  in their court case, he affirmed. The sisters have filed a lawsuit against the Obama administration for its 2012 mandate that employers provide insurance coverage for birth control, sterilizations, and drugs that can cause abortions in employee health plans. The sisters have maintained that to provide this coverage would violate their religious beliefs. Even after the Obama administration modified the rules as an  accommodation  for objecting organizations, the sisters held that the revised rules would force them to violate their consciences. The majority of a three-judge panel for the Tenth Circuit Court of Appeals ruled in July that the Little Sisters of the Poor did not establish that the mandate was a  substantial burden  on their free exercise of religion, and thus ruled they still had to abide by the mandate. The papal visit was not on the official schedule for Pope Francis  Washington, D.C. visit, which included Wednesday visits to the White House, a midday prayer service with the U.S. bishops at St. Matthew s Cathedral, and the canonization mass for St. Junipero Serra at the Basilica of the National Shrine of the Immaculate Conception. It was a  little addition to the program, but I think it has an important meaning,  Fr. Lombardi said. He added that the visit is connected to the words that the Pope has said in support of the position of the bishops of the United States in the speech to President Obama and also in the speech to the bishops. Pope Francis, with President Obama at the White House, called religious freedom  one of America s most precious possessions  and had hearkened to the U.S. bishops  defense of religious freedom.  All are called to be vigilant, precisely as good citizens, to preserve and defend that freedom from everything that would threaten or compromise it,  he had said. In response to the news of the visit with the sisters, Archbishop Joseph Kurtz of Louisville, president of the U.S. Bishops Conference, said that he was  so pleased  to hear of the visit. As you know the last thing the Little Sisters of the Poor want to do is sue somebody. They don t want to sue in court,  he insisted.  They simply want to serve people who are poor and elderly, and they want to do it in a way that doesn’t conflict with their beliefs. Via: Catholic News Agency"
print("  ---> Finished.")

print("Loading Google Pretrained Word2Vec ...")
pretrained_model = KeyedVectors.load_word2vec_format('pretrained-word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print("  ---> Finished.")

print("Building the word graph for the input article ...")
num_nodes, id_2_word, word_2_id, adj_matrix, h_matrix, p_matrix, z_vec = process_text(test_text, pretrained_model)
print("  ---> Finished.")

print("Making predictions ...")
clf = load('trained-classifier/saved_mlp.joblib')
label, prob = prediction(z_vec, clf)
print(label, prob)
print("  ---> Finished.")

print("Analyzing misleading words...")
ranks = misleading_top_n_words(label, prob, num_nodes, adj_matrix, p_matrix, h_matrix, id_2_word)
print(ranks)
print("  ---> Finished.")
