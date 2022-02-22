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
* Run the "gui_disco.py" to get the software as shown in the demo video.
* [Optional], you can put [raw fake news data](https://drive.google.com/file/d/1T798b0Qi4AB6GzOTccbsCaPmhSI_0iN9/view?usp=sharing) and [raw real news data](https://drive.google.com/file/d/15mOoPsUaI9OeWiHJ5XP-u_oDlrxzeo8z/view?usp=sharing) in "raw-dataset" folder and re-train the DISCO model, the inner classifier of DISCO will be automatically saved in "trained-classifier" folder.

#### 4. Techniqual Logic of DISCO

<p align="center"> <img align="center" src="/software_architecture.png" width="525" height="323"> </p>
