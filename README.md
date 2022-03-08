# DISCO

"DISCO" is a disinformation detection toolkit. An online demo video is available [here](https://drive.google.com/file/d/1Nhw1veqjIN9SBz1RLJPDTRVTHuknfjHl).
<p align="center"> <img align="center" src="/user_interface.jpg" width="1000" height="400"> </p>

#### 1. Function of DISCO
* Input: A batch of susceptive information

* Output:
  * The fake news probability and real news probability for an news article query
  * Misleading degree rankings of each word of in that query article

#### 2. Required Library
* numpy 1.20.1
* scipy 1.6.2
* pandas 1.2.4
* nltk 3.6.2
* gensim 4.0.1
* sklearn 0.24.1

#### 3. Quick Start
* Download the code
* Download pre-trained word2vec model ([here](https://code.google.com/archive/p/word2vec/) or [here](https://drive.google.com/file/d/1W8EfxWRBchX_c6ShC6neZRKlokhPV4tR/view?usp=sharing)) and put it in the "pretrained-word2vec" folder
* Run the "gui_disco.py" to get the software as shown in the demo video
* [Optional]: You can train DISCO from the scratch as below
  * First, you can put [raw fake news data](https://drive.google.com/file/d/1T798b0Qi4AB6GzOTccbsCaPmhSI_0iN9/view?usp=sharing) and [raw real news data](https://drive.google.com/file/d/15mOoPsUaI9OeWiHJ5XP-u_oDlrxzeo8z/view?usp=sharing) in "raw-dataset" folder and run "data_preprocessing.py". Then [feature_matrix.pkl](https://drive.google.com/file/d/1TtAc6rBs5rxCyvqMqjWyCtsjWfpl7Mgn/view?usp=sharing) and [label_matrix.pkl](https://drive.google.com/file/d/1Drdyr0WiCbK6KV2TXYVSdMqPvJcK2Eni/view?usp=sharing) will be automatically saved in the "preprocessed-dataset" folder.
  * Then, you can run "model_training.py" to obtain the inner classifier of DISCO, the inner classifier of DISCO will be automatically saved in the "trained-classifier" folder.
  * Now, you get the complete DISCO and could run "gui_disco.py" to get the software as shown in the demo video.

#### 4. Technical Logic of DISCO

<p align="center"> <img align="center" src="/software_architecture.png" width="525" height="323"> </p>

* **Building Word Graph**. We contrust an undirected word graph for each input news article. Briefly, if two words co-occur in a length-specified sliding window, then there will be an edge connecting these two words. For example, "I eat an apple" and the length of the window is 3, then edges could be {I-eat, I-an, eat-an, eat-apple, an-apple} (with stop words kept). More details of constructing a word graph can be found at [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
* **Geometric Feature Extraction**. We use the idea of the [SDG](https://github.com/DongqiFu/SDG) to obtain node embeddings. Briefy, a node's representation is aggregated based on its personalized PageRank vector weighted neighours' features. Then we call any pooling function (like sum pooling or mean pooling) to aggregate node embeddings into the graph-level representation vector for each constructed word graph.
* **Neural Detection**. We train a model-agnostic classification module as the inner classifier of DISCO.
* **Misleading Degree Analysis**. With the support of [SDG](https://github.com/DongqiFu/SDG), we can mask any word node in the contrusted word graph and fast track the new Personalized PageRank to get the new graph-level embedding vector. Without fine-tuning the inner classifier of DISCO, we can investigate each word's contribution (positive or negative) towards the ground-truth label prediction probability.
* [Optional]: You can access our another [repository](https://github.com/DongqiFu/Disinfomation_Case_Study) for a more thorough disinformation study, such as truncated feature dimensions, label noise injection, etc.
