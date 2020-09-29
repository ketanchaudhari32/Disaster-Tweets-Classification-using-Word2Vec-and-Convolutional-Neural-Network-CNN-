# Disaster-Tweets-Classification-using-Word2Vec-and-Convolutional-Neural-Network-CNN-

•	Software required:
	Python3
	Jupyter Notebook (For running the code of model testing and evaluating)
	Flask Python Framework (For GUI interface)

•	Create a python3 tensorflow virtual environment and create kernel with 	following libraries:
1.	Pandas
2.	Numpy
3.	Nltk
4.	Wordcloud
5.	Matplotlib
6.	Tensorflow – version 2
7.	Sklearn
8.	Keras
9.	Genism
10.	Tweepy

•	Data based used:
	Socialmedia-disaster-tweets.csv provided by Kaggle located in datasets folder.

•	Before running the code:
1.	Extract all the zip files of trained model located 	in saved_models and GUI interface folders.
2.	Need to download some files:
	glove_twitter_27B_200d.bin (1.91 GB) - http://nlp.stanford.edu/data/glove.twitter.27B.zip
	GoogleNews-vectors-negative300-003.bin (3.39 GB) - https://www.kaggle.com/leadbest/googlenewsvectorsnegative300?select=GoogleNews-vectors-negative300.bin

	Extract glove_twitter and store both glove_twitter and googleNewsvector in processedData folder.



•	Steps:
	On Jupyter Notebook using python3 tensorflow virtual environment kernel run following 	files in given order.
1.	Preprocess.ipynb -- It imports the dataset and remove noise from data, and after 	preprocessing it stored cleaned dataset csv file in processedData folder.
2.	Visualize.ipynb – It is use to create word cloud visualization of most frequent word in the 	dataset.
3.	Word2vec.ipynb – It helps in creating feature vectors (custom weight vectors) that will be 	used in CNN model.
4.	CNN_CLASS.ipynb – it is use for training and evaluating the model and comparing it with 	models trained using google news vectors and twitter glove vectors.
5.	Tweetclassify.ipynb – Used to test the model on random streamed tweets.
6.	Twitter_stream.ipynb – Uses tweepy API to live tweets from twitter based on tags. 

•	FOR GUI:
	In GUI interface folder using flask framework run – python app.py on command prompt.
	Within GUI you need to enter a tweet and output will be provided as classified tweet.
