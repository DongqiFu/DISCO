from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
import re
import numpy as np
from joblib import dump, load
from utils import power_iteration, track_trans
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


def disco():
    # print("Building the word graph for the input article ......")
    process_area.insert(tk.END, " --- Building the Word Graph for the Input Article ......\n")
    process_area.see(tk.END)
    process_area.update_idletasks()

    review = re.sub("[^a-zA-Z]", " ", text_area.get('1.0', 'end-1c'))
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

    # print("Extracting Geometric Features ......")
    process_area.insert(tk.END, " --- Extracting Geometric Features ......\n")
    process_area.see(tk.END)
    process_area.update_idletasks()

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


    # print("Making predictions .....")
    process_area.insert(tk.END, " --- Neural Detection .....\n")
    process_area.see(tk.END)
    process_area.update_idletasks()

    clf = load('trained-classifier/saved_mlp.joblib')
    prob = clf.predict_proba([z_vec])
    fake_prob = f'{prob[0][0]:.14f}'
    real_prob = f'{prob[0][1]:.14f}'

    label = clf.predict([z_vec])
    # print(label, prob)
    decision = ""
    if label == 0:
        decision = "Detection Result: Fake\n" + "Fake Probability: " + str(fake_prob) + "\n" + "Real Probability: " + str(real_prob) + "\n"
    else:
        decision = "Detection Result: Real\n" + "Fake Probability: " + str(fake_prob) + "\n" + "Real Probability: " + str(real_prob) + "\n"
    result_area.insert(tk.END, decision)
    result_area.see(tk.END)
    result_area.update_idletasks()

    # print("Analyzing each word misleading degree .....")
    process_area.insert(tk.END, " --- Analyzing Each Word Misleading Degree .....\n")
    process_area.see(tk.END)
    process_area.update_idletasks()

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

    ranking = sorted(word_2_misleading_degree.items(), key=lambda item: item[1], reverse=True)[:10]
    # print(ranking)
    ranks = "[Word]:\t[Misleading Degree]\n"
    for item in ranking:
        ranks += str(item[0]) + ":\t" + str(f'{item[1]:.14f}') + "\n"
    result_area.insert(tk.END, ranks + "\n")
    # result_area.see(tk.END)
    # result_area.update_idletasks()

    process_area.insert(tk.END, " --- DISCO Finished .....\n\n")
    process_area.see(tk.END)
    process_area.update_idletasks()


if __name__ == '__main__':
    print("\nLoading Google pre-trained Word2Vec ......\n")
    pretrained_model = KeyedVectors.load_word2vec_format('pretrained-word2vec/GoogleNews-vectors-negative300.bin', binary=True)

    print("\nStarting GUI of DISCO ......\n")
    root = tk.Tk()
    root.geometry('1520x680')
    root.title("DISCO")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)

    ttk.Label(root, text="DISCO: Comprehensive and Explainable Disinformation Detection", font=("Times New Roman", 18), justify="center").grid(column=0, row=0, columnspan=2)

    ttk.Label(root, text="Enter your articles :", font=("Bold", 12)).grid(column=0, row=1)
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=25, font=("Times New Roman", 15))
    text_area.grid(column=0, row=2, pady=10, padx=10, rowspan=3)
    # placing cursor in text area
    text_area.focus()

    ttk.Label(root, text="Process :", font=("Bold", 12)).grid(column=1, row=1)
    process_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=8, font=("Times New Roman", 15))
    process_area.grid(column=1, row=2, pady=10, padx=10)
    # placing cursor in text area
    process_area.focus()

    ttk.Label(root, text="Results :", font=("Bold", 12)).grid(column=1, row=3)
    result_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=13, font=("Times New Roman", 15))
    result_area.grid(column=1, row=4, pady=10, padx=10)
    # placing cursor in text area
    result_area.focus()

    ttk.Button(root, text='Send', width=10, command=disco).grid(column=0, row=5, pady=10, padx=10)

    root.mainloop()
