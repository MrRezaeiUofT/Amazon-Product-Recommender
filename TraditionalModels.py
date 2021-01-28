from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Dropout,Flatten,Dense,TimeDistributed,Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from Part1_utils import *
np.random.seed(0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
'''Load the Data'''
MaxWordLength=50
Batch_Size = 20000
Dp_rate=0.5
Epochs=200
vocabulary_size = 2000
Data=pd.read_json('./train.json/train.json',lines=True)
RatePairData=pd.read_json('./test.json/test.json',lines=True)
CompensateNumber=Data.overall.value_counts()[5]


RatePairID=pd.read_csv('rating_pairs.csv')

''' Data Pre-processing'''
'''P1--> text processing'''
Data=Data.reset_index()
Text=[]
for i in range(Data.shape[0]):
    print(i)
    Text.append(clean_text(str(Data['summary'][i]) + str(Data['reviewText'][i]) ))
for i in range(RatePairData.shape[0]):

    Text.append(clean_text(str(RatePairData['summary'][i]) + str(RatePairData['reviewText'][i]) ))


# with open('TokenReview.json') as data_file:
#     tokenizerPath = json.load(data_file)
# tokenizer=tf.keras.preprocessing.text.tokenizer_from_json(tokenizerPath)
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(Text)
F_textD = tokenizer.texts_to_matrix(Text, mode='freq')
# F_textD = pad_sequences(sequences, maxlen=MaxWordLength)
F_text=F_textD[:Data.shape[0],:]
F_text_te=F_textD[Data.shape[0]:,:]
'''P2--> Normalize unixReviewTime'''
F_time=( Data.unixReviewTime.values  - Data.unixReviewTime.min())/(Data.unixReviewTime.max() - Data.unixReviewTime.min())
F_time=F_time.reshape([-1,1])
'''P3--> one-hot encoding of categories'''
F_cat=pd.get_dummies(Data.category).values
'''P4 --> for later'''
''' Part 5 --> add popularity of an product as feature. I used histigram method'''
newD=pd.DataFrame(np.concatenate([Data['itemID'].values,RatePairData['itemID'].values],axis=0),columns=['itemID'])
newD['reviewerID']=np.concatenate([Data['reviewerID'].values,RatePairData['reviewerID'].values],axis=0)
newD['itemID']=newD['itemID'].groupby(newD['itemID']).transform('count')
newD['reviewerID']=newD['reviewerID'].groupby(newD['reviewerID']).transform('count')


RID_temp=pd.get_dummies(newD['reviewerID']).values
IID_temp=pd.get_dummies(newD['itemID']).values

F_newD_reviewerID=RID_temp[:Data.shape[0],:]
F_newD_reviewerID_te=RID_temp[Data.shape[0]:,:]

F_newD_itemID=IID_temp[:Data.shape[0],:]
F_newD_itemID_te=IID_temp[Data.shape[0]:,:]

'''test-train split '''
y = Data['overall']
y=Data.overall.to_numpy()
X=np.concatenate([F_text,F_time,F_cat,F_newD_reviewerID,F_newD_itemID],axis=-1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
########### naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)
print('Naive_bayesian RMS=%f'%(metrics.mean_squared_error(y_test,nb_preds)))
########### SVM
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, y_train)
rl_preds=logreg.predict(X_test)

print('Logistic Regression RMS=%f'%(metrics.mean_squared_error(y_test,rl_preds)))
############# Final Predictions
'''P2--> Normalize unixReviewTime'''
F_time_te=( RatePairData.unixReviewTime.values  - Data.unixReviewTime.min())/(Data.unixReviewTime.max() - Data.unixReviewTime.min())
F_time_te=F_time_te.reshape([-1,1])
'''P3--> one-hot encoding of categories'''
F_cat_te=pd.get_dummies(RatePairData.category).values
X_te=np.concatenate([F_text_te,F_time_te,F_cat_te,F_newD_reviewerID_te,F_newD_itemID_te],axis=-1)

y_pred = nb.predict(X_te)
RatePairID.prediction=y_pred


RatePairID.to_csv('rating_predictions.csv',index=False)
