# Disco

"Disco" is a disinformation detection software demo.

#### 1. Function of Disco
* Input: A batch of susceptive information

* Output:
  * The probabilities that each piece of input information is fake or real
  * Top k misleading words of each input information 
  * Classification effectiveness, such as accuracy, precision, recall, and F1-score

#### 2. How to run Disco
* First, put the data file into the "fake-and-real-news-dataset" folder
* Second, run "SDG_model.py" to extract vector representation for each news article, which will store a feature matrix and a label matrix
* Third, run "classification_acc.py", which is responsible for training an accuracy-acceptable classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) and saving the classification model
* Fourth, run "top_n_words.py" to find top n misleading words in a news article. For example, with word w the fake article news a is detected as fake news with probability p, without word w the fake article news a is detected as fake news with probability q, then the probability gain of w to a is (q-p).
* Fifth, run "dimension_redution.py" to reduce the feature dimension by PCA, and train a new classifier (e.g. MuFasa model by "mufasa.py" or MLP by sklearn) on the truncated feature matrix.
* Sixth, run "other_metrics.py" to test the trained classider in terms of other metrics, like precision, recall, and F1-score.

### Dependency
* numpy 1.19.5
* pandas 1.2.4
* nltk 3.6.2
* gensim 4.0.1
* pytorch 1.6.0
