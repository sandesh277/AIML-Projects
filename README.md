# Artifical Intelligence and Machine Learning-Projects
Course projects completed during PGP -Artificial Intelligence and Machine Learning at Great Learning


# [Project1: Statistical Learning: Project Overview](https://github.com/sandesh277/AIML-Projects/tree/main/01-AppliedStatistics)

* Covers Descriptive Statistics, Probability & Conditional Probability, Hypothesis Testing, Inferential Statistics, Probability Distributions, Types of distribution and Binomial, Poisson & Normal distribution.
* This project used Hypothesis Testing and Visualization to leverage customer's health information like smoking habits, bmi, age, and gender for checking statistical evidence to make valuable decisions of insurance business like charges for health insurance.

# [Project2: Supervised Machine Learning: Project Overview](https://github.com/sandesh277/AIML-Projects/tree/main/02-Supervised-Learning)

* Objective: The classification goal was to predict the likelihood of a liability customer buying personal loans
* Covers Multiple Variable Linear regression, Logistic regression, Naive Bayes classifiers, Multiple regression, K-NN classification, Support vector machines
* Identified potential loan customers for Thera Bank using classification techniques. Compared models built with Logistic Regression and KNN algorithm in order to select the best performing one
* Predicted the possibility that a customer will buy a personal loan or not 
* Exploratory data analysis using matplotlib and seaborn libraries 
* Descriptive and inferential statistics to gain valuable insights from data
* Data processing and preparation like normalization, splitting of data for model
* Trained and tested different supervised learning algorithms – Logistic regression, KNN and Logistic regression to predict a potential customer buying a personal loan
* Model performance assessment using techniques like confusion matrix, classification reports etc

# [Project 3: Ensemble Techniques: Project Overview](https://github.com/sandesh277/AIML-Projects/tree/main/03-Ensemble-Techniques)
* Covers Decision Trees, Bagging, Random Forests, Boosting
* Objective: The classification goal is to predict if the client will subscribe (yes/no) a term deposit
* Leveraged customer information of bank marketing campaigns to predict whether a customer will subscribe to term deposit or not. Different classification algorithms like Decision tree, Logistic Regression were used. Ensemble techniques like Random forest were used to further improve the classification results.
* Exploratory data analysis to extract insights from data
* Data preprocessing and cleansing – detection of missing values, detection and handling of outliers in data etc
* Plotted relationship between numerical and categorical variables, numerical-numerical attributes
* Encoding of categorical variables
* Built and visualized decision trees, feature importance
* Built Ensemble models- Random forest, bagging classifier, Adaboost and gradient boost etc
* Build hybrid models
* Model evaluation – Confusion matrix, precision recall curves, AUC

# [Project 4: Unsupervised Machine Learning: Project Overview](https://github.com/sandesh277/AIML-Projects/tree/main/04-Unsupervised-Learning)
* Covers K-means clustering, High-dimensional clustering, Hierarchical clustering, Dimension Reduction-PCA4
* Objective: The classification goal is to predict type of vehicle from Corgie model dataset
* Classified vehicles into different types based on silhouettes which may be viewed from many angles. Used PCA in order to reduce dimensionality and SVM for classification
* Trained a Support vector machine using the train set
* Performed K-fold cross validation and evaluated by computing the cross validation score of the model 
* Used PCA from Scikit learn, extract Principal Components that capture about 95% of the variance in the data 
* Plotted relationship between numerical and categorical variables, numerical-numerical attributes

# [Project 5: Feature Engineering and model tuning](https://github.com/sandesh277/AIML-Projects/tree/main/05-Feature%20Engineering%20and%20Model%20Tuning)
* Covers Exploratory Data Analysis, Feature Exploration and Selection Techniques, Hyperparameter Tuning
* Objective: Modeling of strength of high performance concrete using Machine Learning
* Used feature exploration and selection technique to predict the strength of high-performance concrete. 
* Used regression models like decision tree regressors to find out the most important features and predict the strength. 
* Cross-validation techniques and grid search were used to tune the parameters for the best model performance.

# [Project 6: Recommendation Systems](https://github.com/sandesh277/AIML-Projects/tree/main/06-Recommendation-Systems)
* Covers Introduction to Recommendation systems, Popularity based model, Hybrid models, Content based recommendation system, Collaborative filtering (User similarity & Item similarity)
* Objective: To make a recommendation system that recommends at least five new products based on the user's habits.
* Used surprise library to make a popularity based recommender system
* Built collaborative filtering based recommender system using KNNwithmeans and SVD (singular value decomposition) techniques from surprise library
* Hyperparameter tuning using GridsearchCV and randomized searchcv
* Model evaluation using RMSE and MAE

# [Project 7: Neural Networks](https://github.com/sandesh277/AIML-Projects/tree/main/07-neural%20networks)
* Covers Gradient Descent, Batch Normalization, Hyper parameter tuning, Tensor Flow & Keras for Neural Networks & Deep Learning, Introduction to Perceptron & Neural Networks, Activation and Loss functions, Deep Neural Networks
* Objective: implement a simple image classification pipeline based on a deep neural network 
* Used Tensorflow and Keras to build deep neural network based image recognition algorithm utilizing GPU in google colab
* Built separate algorithms using Fully connected layers using different activation functions(Relu and sigmoid), optimisers(Adam, SGD)
* Used different numbers of hidden layers and hidden neurons to fine tune the model
* Built models using different weight initialization techniques and learning rate of the parameters
* Used batch normalization and dropout layers to contain overfitting of the model
* Hyperparameter tuning using Gridsearch CV
* Obtained an overall accuracy of 85% on test data
* Made predictions on test data with final models

# [Project 8: Computer Vision: Face Mask Prediction](https://github.com/sandesh277/AIML-Projects/tree/main/08-computer%20vision)
* Covers Introduction to Convolutional Neural Networks, Convolution, Pooling, Padding & its mechanisms, Transfer Learning, Forward propagation & Backpropagation for CNNs, CNN architectures like AlexNet, VGGNet, InceptionNet & ResNet
* Recognize, identify and classify faces within images using CNN and image recognition algorithms. 
* Objective is to build a face recognition system, which includes building a face detector to locate the position of a face in an image

# [Project 9: Advanced Computer Vision: Face Recognition](https://github.com/sandesh277/AIML-Projects/tree/main/09-Face%20Recognition)
* Covers Semantic segmentation, Siamese Networks, YOLO, Object & face recognition using techniques above
* Objective: build a face recognition system, which includes building a face detector to locate the position of a face in an image and a face identification model to recognize whose face it is by matching it to the existing database of faces
* Face recognition deals with Computer Vision a discipline of Artificial Intelligence and uses techniques of image processing and deep learning 
* Data Description: Aligned Faces Dataset from Pinterest (10k+ images of 100 celebs) 
* Face recognition model recognises similar faces with an accuracy of 97% and F1 score of 96%

# [Project 10: NLP: Sentiment Analysis](https://github.com/sandesh277/AIML-Projects/tree/main/10-NLP-Sentiment-Analysis)
* Covers Bag of Words Model, POS Tagging, Tokenization, Word Vectorizer, TF-IDF, Named Entity Recognition, Stop Words
* The objective of this project is to build a text classification model that analyses the customer's sentiments based on their reviews in the IMDB database
* The model uses a complex deep learning model to build an embedding layer followed by a classification algorithm to analyze the sentiment of the customers

# [Project 11: NLP: Sarcasm Detection](https://github.com/sandesh277/AIML-Projects/tree/main/11-NLP-Sarcasm-Detection)
* Covers Introduction to Sequential data, Vanishing & Exploding gradients in RNNs, LSTMs, GRUs (Gated recurrent unit), Case study: Sentiment analysis, RNNs and its mechanisms, Time series analysis, LSTMs with attention mechanism
* Objective: The goal of this hands-on project is to analyse the headlines of the articles from news sources and detect whether they are sarcastic or not.






