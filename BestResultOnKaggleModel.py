
from collections import defaultdict
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
Dp_rate=0.2
Epochs=200
vocabulary_size = 2000
Data=pd.read_json('./train.json/train.json',lines=True)
RatePairData=pd.read_json('./test.json/test.json',lines=True)

RatePairID=pd.read_csv('rating_pairs.csv')

''' Data Pre-processing'''
'''P1--> text processing'''

Text=[]
for i in range(Data.shape[0]):
    # print(i)
    Text.append(clean_text(str(Data['summary'][i]) + str(Data['reviewText'][i]) ))
for i in range(RatePairData.shape[0]):

    Text.append(clean_text(str(RatePairData['summary'][i]) + str(RatePairData['reviewText'][i]) ))



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
newD['itemID']=(newD['itemID'].values-newD['itemID'].min())/(newD['itemID'].max()-newD['itemID'].min())

newD['reviewerID']=newD['reviewerID'].groupby(newD['reviewerID']).transform('count')
newD['reviewerID']=(newD['reviewerID'].values-newD['reviewerID'].min())/(newD['reviewerID'].max()-newD['itemID'].min())
newD=newD.to_numpy()
F_newD=newD[:Data.shape[0]]
F_newD_te=newD[Data.shape[0]:]

'''test-train split '''
y = Data['overall']
y=Data.overall.to_numpy()

''' Model Selection'''
x_text = Input(shape=(F_text.shape[1]),name='x_text')
x_time = Input(shape=(F_time.shape[1]),name='x_time')
x_cat = Input(shape=(F_cat.shape[1]),name='x_cat')
x_newD = Input(shape=(F_newD.shape[1]),name='x_newD')


H=Dense(MaxWordLength,activation='relu')(x_text)
H=Dropout(Dp_rate)(H)
H=Dense(MaxWordLength,activation='relu')(H)
H=Dropout(Dp_rate)(H)
H=tf.concat([H,x_time,x_cat,x_newD],axis=-1)
H=Dense(MaxWordLength, activation='relu')(H)
H=Dense(MaxWordLength, activation='relu')(H)
# Final predictions and model.
prediction = Dense(1, activation='sigmoid')(H)
# kernel_initializer='ones',
#     kernel_regularizer=tf.keras.regularizers.L1(0.01),
#     activity_regularizer=tf.keras.regularizers.L2(0.01)
model = Model(inputs=[x_text,x_time,x_cat,x_newD], outputs= prediction)
model.compile(loss=tf.keras.losses.MSE,
              optimizer='adam')
model.summary()

''' Train modek'''
history=model.fit([F_text,F_time,F_cat,F_newD], y/5,
          batch_size=Batch_Size,
          epochs=Epochs,
          verbose=1,
          validation_split=0.0,shuffle=True)
plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
##################### prediction
'''P2--> Normalize unixReviewTime'''
F_time_te=( RatePairData.unixReviewTime.values  - Data.unixReviewTime.min())/(Data.unixReviewTime.max() - Data.unixReviewTime.min())
F_time_te=F_time_te.reshape([-1,1])
'''P3--> one-hot encoding of categories'''
F_cat_te=pd.get_dummies(RatePairData.category).values

y_pred = model([F_text_te,F_time_te,F_cat_te,F_newD_te])
RatePairID.prediction=y_pred.numpy()*5


RatePairID.to_csv('rating_predictions.csv',index=False)