# Sentiment-Analysis-using-tensorflow
Sentimental Analysis was done using NLTK and the model was trained on deep neural network using tensorflow.

The dataset sentiment140 was used which contain unstructured comments in csv files, pickle was created for train & test data which was loaded to create feature vector for the entire train and test set. After which it is feed to a simple nueral_network in batches & was trained on train set (a part of train.csv & test.csv I have uploaded). After training the model it was saved in Train_test_nueral_network.py using tensorflow tf.train.Saver object so that can be reused and can be used to check a single comment that we are feeding.

create_pickle.py:
This code is used to create a reuseable pickle from trainData into lexicon-2500-2638.pickle and created a corpus for the test set into a csv. The pickle from train data and corpus from test is used by train_test_nueral_network.py

train_test_nueral_network.py
This code contains the computation graph and how the model has been build , the model is saved to increase reuseability, also in the end after predicting the values for test data, dummy comments have also been provided to the nueral network in the function use_neural_network(input_data)
