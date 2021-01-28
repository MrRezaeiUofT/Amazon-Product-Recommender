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
Batch_Size = 10000
Dp_rate=0.5
Epochs=10
vocabulary_size = 5000
Data=pd.read_json('./train.json/train.json',lines=True)
RatePairData=pd.read_json('./test.json/test.json',lines=True)
CompensateNumber=Data.overall.value_counts()[5]
# for i in range(1,5):
#     D_m=Data.loc[Data.overall == i]
#     df_minority_upsampled = resample(D_m,
#                                  replace=True,     # sample with replacement
#                                  n_samples=CompensateNumber,    # to match majority class
#                                  random_state=12*i) # reproducible results
#
#     Data= pd.concat([Data, df_minority_upsampled])




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
    # tokenizerPath = json.load(data_file)
# tokenizer=tf.keras.preprocessing.text.tokenizer_from_json(tokenizerPath)
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(Text)
sequences = tokenizer.texts_to_sequences(Text)
F_textD = pad_sequences(sequences, maxlen=MaxWordLength)
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


RID_temp=newD['reviewerID'].to_numpy()
RID_temp=RID_temp.reshape([-1,1])
IID_temp=newD['itemID'].to_numpy()
IID_temp=IID_temp.reshape([-1,1])
vocab_size_RID=RID_temp.max()+1
vocab_size_IID=IID_temp.max()+1
F_newD_reviewerID=RID_temp[:Data.shape[0],:]
F_newD_reviewerID_te=RID_temp[Data.shape[0]:,:]

F_newD_itemID=IID_temp[:Data.shape[0],:]
F_newD_itemID_te=IID_temp[Data.shape[0]:,:]

'''test-train split '''

y=Data.overall.to_numpy()

''' Model Selection'''
x_text = Input(shape=(F_text.shape[1]),name='x_text')
x_time = Input(shape=(F_time.shape[1]),name='x_time')
x_cat = Input(shape=(F_cat.shape[1]),name='x_cat')
x_RID = Input(shape=(F_newD_reviewerID.shape[1]),name='x_RID')
x_IID = Input(shape=(F_newD_itemID_te.shape[1]),name='x_IID')

H_t=tf.keras.layers.Embedding(vocabulary_size, 100)(x_text)
H_t=LSTM(100, dropout=Dp_rate, recurrent_dropout=Dp_rate)(H_t)


H_RID=tf.keras.layers.Embedding(vocab_size_RID, 20)(x_RID)
H_RID=tf.reshape(H_RID, (-1,20))


H_IID=tf.keras.layers.Embedding(vocab_size_IID, 20)(x_IID)
H_IID=tf.reshape(H_IID, (-1,20))

H_c=Dense(5,activation='relu')(x_cat)
H_c=Dropout(Dp_rate)(H_c)

H=tf.concat([H_t,H_RID,H_IID,x_cat,x_time],axis=-1)
# Final predictions and model.
prediction = Dense(1, activation='sigmoid')(H)
# kernel_initializer='ones',
#     kernel_regularizer=tf.keras.regularizers.L1(0.01),
#     activity_regularizer=tf.keras.regularizers.L2(0.01)
model = Model(inputs=[x_text,x_time,x_cat,x_RID,x_IID], outputs= prediction)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss=tf.keras.losses.MSE,
              optimizer=optimizer,metrics=['mse'])
model.summary()

''' Train mode'''
history=model.fit([F_text,F_time,F_cat,F_newD_reviewerID,F_newD_itemID], y/5,
          batch_size=Batch_Size,
          epochs=Epochs,
          verbose=1,
          validation_split=.3,shuffle=True)
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

y_pred = model([F_text_te,F_time_te,F_cat_te,F_newD_reviewerID_te,F_newD_itemID_te])
RatePairID.prediction=y_pred.numpy()*5


RatePairID.to_csv('rating_predictions.csv',index=False)