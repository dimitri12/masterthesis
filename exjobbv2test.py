#-*- coding: utf-8 -*-
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
sys.path.append("/Users/dimitrigharam/Desktop/exjobb")
import exjobbv2 as run
import nltk
from nltk.corpus import stopwords 
import numpy as npa
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle
import string
from gensim.models import word2vec
from collections import Counter
import sklearn
print('sklearn: %s' % sklearn.__version__)
'''
By: Dimitri Gharam
This code will perform aside of preprocessing test 21 approaches of embeddings and encodings (15 encodings and 6 embeddings)
Encoding:
1. One-hot Encoding
2. TF-IDF
3. Bag Of Words/CountVectorizer(analyzer=word)

Embeddings:
1. Word2Vec
2. Fasttext
(These word embeddings are the only methods that supports the swedish language.)
(I will upload into my appendix a guide on how to build a corpus outside the python code since there is a benefit for this.)

Classification methods (for the encodings) (according to a source):
1. Logistic Regression
2. Naive Bayes: Gaussian and Multinomial
3. Support Vector Machine
4. Random Forest
5. Ensemble (a combination of the 4 methods above)

Machine learning models (for the embeddings):
1. CNN
2. LSTM
3. GRU

print("##################################")
print("Test Code")
'''
result_frame = pd.DataFrame(columns=["Method","Accuracy"])
df = pd.read_csv("testdata_exjobb.csv", encoding='ISO-8859-1')
with open('df.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
#secret weapon
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode('utf8', 'ignore')
    return txt
text = input_data.apply(clean_text)
'''
#labels = run.create_labels_list(df)

#The labels list contains all the data from columns that has hexadecimal values
#The solution is to perform individual predictions but generate precision as metric

'''
42 indexes
labels by index:
0: caseid
1: FreeText
0 can be excluded
1 is input
2-41 is outputs i.e. 40 output data: 30 with hex names and 10 with names
index 7 (pout), index 8 (prio) and index 6 (operator) is considered multiclass
index 2 (Age) and index 5 (LastContactDays) are considered numerical and needs
to be categorised
After categorisation index 2 and 5 will be assigned to index 42 and 43 
as AgeCat and LcdCat

the rest of the outputs are binary outputs

The Labels who doesn't have a name (Hexvalues) is 99% of the data zeros and there is only a
single 1



'''



'''
Categorise index 2 and 5
'''
df = run.transformAge(df)
df = run.transformLCD(df)

#array = run.doMultiClass(df)
#print(df)
#it works
'''
Perform an Explorative data analysis
'''

input_data = df.FreeText.astype(str)
with open('input.pickle', 'wb') as handle:
        pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#pout,prio,operator,agecat,lcdcat

'''



Part 1: Text Classification with BoW and TF-IDF

#CHALLENGE: solve the multiclass classification problem


'''

print("##################################")
print("Preprocessing:")
print("----------------------------------")



'''
#save
with open('preproc1.pickle', 'wb') as handle:
    pickle.dump(preproc1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('preproc2.pickle', 'wb') as handle:
    pickle.dump(preproc2, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('preproc3.pickle', 'wb') as handle:
    pickle.dump(preproc3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('preproc4.pickle', 'wb') as handle:
    pickle.dump(preproc4, handle, protocol=pickle.HIGHEST_PROTOCOL)

#load
with open('preproc1.pickle', 'rb') as handle:
    input_data = pickle.load(handle)
with open('output_data.pickle', 'wb') as handle:
    pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''


print("Preprocessing finished")
print("##################################")
'''
print("Transforming output data:")
print("----------------------------------")
outputdata = run.transform_output_data(output_data)
outputdata2 = run.transform_output_data(output_data,"int")
#glömde detta:
input_data = preproc2
#print(outputdata2)
print("Inverse Transforming:")
inv_output = run.inverse_transform_output_data(outputdata.NewPout)
'''
print("Embedding/Encoding/Processing starting")
print("----------------------------------")
#the result from these encodings are arrays containing training and test data for both the input and output
#in order: input_train, input_test, output_train, output_test

'''
with open('bow.pickle', 'wb') as handle:
    pickle.dump(bow, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bow.tfidf', 'wb') as handle:
    pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bow.pickle', 'rb') as handle:
    bow = pickle.load(handle)
with open('bow.tfidf', 'rb') as handle:
    tfidf = pickle.load(handle)
'''
def classification():
    with open('input.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    with open('df.pickle', 'rb') as handle:
        df = pickle.load(handle)
    preproc1 = run.pre_processing1(input_data,df)
    output_data = df.hosp_ed
    bow = run.text_processing(preproc1,output_data, None, 1)
    tfidf = run.text_processing(preproc1,output_data, "tfidf", 1)
    method = 'NBG'
    data_array = [bow,tfidf]
    multiclass = 'no'
    result = run.predictor(data_array,method,multiclass)
    run.generate_metrics(result,method)
    #method = 'Logistic_regression'
    #index 0=BOW
    #index1=TF*IDF
    #index[0][0] true data for BoW
    #index[0][1] predicted data for BoW
    #index[0][2] accuracy for BoW
    #if log reg is used: #index[0][3] is loss for BoW
    #else #index[0][3] is pred_proba for BoW
    title=[method+'BoW',method+'TFIDF']
    run.plot_roc(result[0],result[1],title,method)

#run.plot_roc(result[1][0],result[1][3])
#run.plot_metrics(run.create_plot_data(result,'NBG'))

print("Embedding/Encoding/Processing is finished")

print("##################################")
print("Modeling and prediction:")
print("----------------------------------")


print("Modeling and prediction finished")
print("##################################")
'''



Part 2: Word Embeddings using Word2Vec,Fasttext and ANNs with LSTM, GRU and CNN




'''

'''
ANNs:
1.CNN1
2.CNN2
3.LSTM
4.GRU1
5.GRU2
6. Default

embedding layers:
0: Word2Vec
1: fasttext (self made)
2: fasttext downloaded from fasttext web page
'''

#outputdata = run.transform_output_data(output_data,'int')

#print(outputdata)

'''

#run.eda2(df)
'''
def test_we():
    ANN1 = 'lstm'
    ANN2 = 'gru'
    ANN3 = 'bilstm'
    ANN4 = 'bigru'
    ANN5 = 'cnn1'
    ANN6= 'cnn2'
    input_data = df.FreeText.astype(str)
    output_data = df.hosp_ed
    test = run.word_embeddings(input_data, output_data,ANN1,2,1)
    test1 = run.word_embeddings(input_data, output_data,ANN2,2,1)
    run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN1,ANN2,test[2],test1[2])
    test = run.word_embeddings(input_data, output_data,ANN3,2,1)
    test1 = run.word_embeddings(input_data, output_data,ANN4,2,1)
    run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN3,ANN4,test[2],test1[2])
    test = run.word_embeddings(input_data, output_data,ANN5,2,1)
    test1 = run.word_embeddings(input_data, output_data,ANN6,2,1)
    run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN5,ANN6,test[2],test1[2])
def test_imdb():
    import numpy as np
    seed = 7
    np.random.seed(seed)
    from keras.datasets import imdb
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=500)
    max_words = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    EL = Embedding(5000, 100, input_length=max_words)
    ANN1 = 'lstm'
    ANN2 = 'gru'
    dense = 1
    run.predict_model(50,EL,ANN1,X_train,y_train,X_test, y_test,dense)
    run.predict_model(50,EL,ANN2,X_train,y_train,X_test, y_test,dense)
    #we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN1,ANN2,test[2],test1[2])

test_we()

print("##################################")
print("Done")

