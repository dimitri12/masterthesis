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
df = pd.read_csv("190203_data_exjobb.csv", encoding='ISO-8859-1')
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

#run.eda(df)

#load output data and define if it is multiclass or not
input_data = df.freetext.astype(str)
input_data1 = df.iloc[:,1:2].freetext.astype(str)

#pout,prio,operator,agecat,lcdcat

'''
Part 1: Text Classification with BoW and TF-IDF
#CHALLENGE: solve the multiclass classification problem
'''

print("##################################")
print("Preprocessing:")
print("----------------------------------")
'''
preproc1 = run.pre_processing1(input_data,df)
preproc2 = npa.vectorize(run.pre_processing2)
preproc2 = preproc2(df.FreeText)
preproc3 = run.pre_processing1(input_data1,df)
preproc4 = npa.vectorize(run.pre_processing2)
preproc4 = preproc4(input_data1)
'''
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
'''
#input_data = preproc1
#input_data = preproc2
#input_data = preproc3
#input_data = preproc4
output_data = df.hosp_ed
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
bow = run.text_processing(input_data,output_data, None, 1)
tfidf = run.text_processing(input_data,output_data, "tfidf", 1)
with open('bow.pickle', 'wb') as handle:
    pickle.dump(bow, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bow.tfidf', 'wb') as handle:
    pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
'''
with open('bow.pickle', 'rb') as handle:
    bow = pickle.load(handle)
with open('bow.tfidf', 'rb') as handle:
    tfidf = pickle.load(handle)
'''
'''
data_array = [bow,tfidf]
multiclass = 'yes'
result = run.predictor(data_array,'NBM',multiclass)
run.generate_metrics(result,'NBM')
#index 0=BOW
#index1=TF*IDF
#index[0][0] true data for BoW
#index[0][1] predicted data for BoW
#index[0][2] accuracy for BoW
#if log reg is used: #index[0][3] is loss for BoW
#else #index[0][3] is pred_proba for BoW
#run.plot_roc(result[0][3],result[0][0])
#run.plot_roc(result[1][3],result[1][0])
#run.plot_metrics(run.create_plot_data(result1,'NBM'))
'''
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
4.GRU
5.GRU2
6. Default
embedding layers:
0: Word2Vec
1: fasttext (self made)
2: fasttext downloaded from fasttext web page
'''

#outputdata = run.transform_output_data(output_data,'int')
ANN1 = 'lstm'
ANN2 = 'gru'
ANN3 = 'bilstm'
ANN4 = 'bigru'
ANN5 = 'cnn1'
ANN6 = 'cnn2'
#print(outputdata)


df['freetext'] = [run.cleaning(s) for s in df['freetext']]
input_data = df.freetext.astype(str)
input_data1 = df.iloc[:,1:2].freetext.astype(str)
#run.eda2(df)

test = run.word_embeddings(input_data, output_data,ANN1,2,1)
test1 = run.word_embeddings(input_data, output_data,ANN2,2,1)
run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN1,ANN2,test[2],test1[2])

test = run.word_embeddings(input_data, output_data,ANN3,2,1)
test1 = run.word_embeddings(input_data, output_data,ANN4,2,1)
run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN1,ANN2,test[2],test1[2])

test = run.word_embeddings(input_data, output_data,ANN5,2,1)
test1 = run.word_embeddings(input_data, output_data,ANN6,2,1)
run.we_evaluation(test[3],test1[3],test[0],test[1],test1[0],test1[1],ANN1,ANN2,test[2],test1[2])
'''
#run.eda1(df)
#run.eda2(df)
df["FreeText_len"] = df["FreeText"].apply(lambda x: len(x))
print(df['FreeText_len'].sum())
data cleaning
#this works
df['FreeText'] = [run.cleaning(s) for s in df['FreeText']]
input_data = df['FreeText']
run.word_cloud(input_data)
#this also works
#run.word_cloud(input_data.apply(run.clean_text))
'''

print("##################################")
print("Done")