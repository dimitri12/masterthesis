import sys
import pandas as pd
sys.path.append("/Users/dimitrigharam/Desktop/exjobb")
import exjobbv2 as run
import nltk
from nltk.corpus import stopwords 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
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
'''



Part 1: Text Classification with OHE, BoW and TF-IDF




'''

result_frame = pd.DataFrame(columns=["Method","Accuracy"])
df = pd.read_csv("testdata_exjobb.csv", encoding='ISO-8859-1')
#input_data
#print(df.FreeText.head())
input_data = df.FreeText
#print(df.FreeText.iloc[0])
#output_data
output_data = df.pout

#print(df.pout)
'''



Part 1: Text Classification with OHE, BoW and TF-IDF




'''

print("##################################")
print("Preprocessing:")
print("----------------------------------")
preproc1 = run.pre_processing1(df.FreeText,df)
preproc2 = np.vectorize(run.pre_processing2)
preproc2 = preproc2(df.FreeText)
print("Preprocessing finished")
print("##################################")
print("Transforming output data:")
print("----------------------------------")
outputdata = run.transform_output_data(output_data)
outputdata2 = run.transform_output_data(output_data,"int")
#glömde detta:
input_data = preproc1
#print(outputdata2)
print("Inverse Transforming:")
inv_output = run.inverse_transform_output_data(outputdata.NewPout)
print("Embedding/Encoding/Processing starting")
print("----------------------------------")
#the result from these encodings are arrays containing training and test data for both the input and output
#in order: input_train, input_test, output_train, output_test
#one_hot2 = run.text_processing(input_data,output_data, "onehot",1,'int')
bag_of_words2 = run.text_processing(input_data,outputdata2, None, 1)
tf_idf2 = run.text_processing(input_data,outputdata2, "tfidf", 1)
data_array = [bag_of_words2,tf_idf2]
print(one_hot2)
print("Embedding/Encoding/Processing is finished")
'''
print("##################################")
print("Modeling:")
print("----------------------------------")
result_logregr = run.predictor(data_array,None)
result_NBG = run.predictor(data_array,'NBGauss')
result_NBM = run.predictor(data_array,'NBMulti')
result_SVM = run.predictor(data_array,'SVM')
result_RF = run.predictor(data_array,'RF')
result_ensemble = run.predictor(data_array,'ensemble')
 
#CHALLENGE: COLLECT ALL ACC RESULTS TO GENERATE ONE GRAPH
print("Modelling finished")
print("##################################")
print("Evaluation of predictions:")
print("----------------------------------")
print("Logistic Regression:")
print("----------------------------------")
#prints the metrics
run.generate_metrics(result_logregr)
#plots the graph for accuracies over different encoding metods for one particular classification method
result_logres_acc = [result_logregr[0][2],result_logregr[1][2],result_logregr[2][2]]
plot_input1 = [result_logres_acc[0],result_logres_acc[1],result_logres_acc[2]]
print("----------------------------------")
print("Naive Bayes (Gaussian):")
run.generate_metrics(result_NBG)
result_NBG_acc = [result_NBG[0][2],result_NBG[1][2],result_NBG[2][2]]
plot_input2 = [result_NBG_acc[0],result_NBG_acc[1],result_NBG_acc[2]]
print("----------------------------------")
print("Naive Bayes (Multinomial):")
print("----------------------------------")
run.generate_metrics(result_NBM)
result_NBM_acc = [result_NBM[0][2],result_NBM[1][2],result_NBM[2][2]]
plot_input3 = [result_NBM_acc[0],result_NBM_acc[1],result_NBM_acc[2]]
print("----------------------------------")
print("Support Vector Machine:")
print("----------------------------------")
run.generate_metrics(result_SVM)
result_SVM_acc = [result_SVM[0][2],result_SVM[1][2],result_SVM[2][2]]
plot_input4 = [result_SVM_acc[0],result_SVM_acc[1],result_SVM_acc[2]]
print("----------------------------------")
print("Random Forest:")
print("----------------------------------")
run.generate_metrics(result_RF)
result_RF_acc = [result_RF[0][2],result_RF[1][2],result_RF[2][2]]
plot_input5 = [result_RF_acc[0],result_RF_acc[1],result_RF_acc[2]]
print("----------------------------------")
print("Ensemble of Logistic Regression, Naive Bayes (Multinomial) and Random Forest:")
print("----------------------------------")
run.generate_metrics(result_ensemble)
result_ensemble_acc = [result_ensemble[0][2],result_ensemble[1][2],result_ensemble[2][2]]
plot_input6 = [result_ensemble_acc[0],result_ensemble_acc[1],result_ensemble_acc[2]]
print("----------------------------------")
result_acc = [result_logres_acc[0],result_logres_acc[1],result_logres_acc[2], \
result_NBG_acc[0],result_NBG_acc[1],result_NBG_acc[2], \
result_NBM_acc[0],result_NBM_acc[1],result_NBM_acc[2], \
result_SVM_acc[0],result_SVM_acc[1],result_SVM_acc[2], \
result_RF_acc[0],result_RF_acc[1],result_RF_acc[2], \
result_ensemble_acc[0],result_ensemble_acc[1],result_ensemble_acc[2]]
run.plot_metrics(run.create_plot_data(result_acc,None))
#plot auc roc curve
'''
print("##################################")
print("Done")

'''



Part 2: Word Embeddings using Word2Vec,Fasttext and ANNs with LSTM, GRU and CNN




'''
'''
########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
#########################################
nltk.download('stopwords')

#preprocessing using tokenization and lower casing

#dummies is a mashup of multiple df columns
dummies = df.iloc[:,1:2]
#which 1 of these extractions is efficient?
FreeText1 = df['FreeText']
FreeText2 = df.FreeText

Pout1 = df['Pout']
Pout2 = df.Pout

FreeText3 = np.array(df['FreeText'])
Pout3 = np.array(df['pout'])

#from [1]
norm_docs = np.vectorize(pre_processing2)
normalized_documents = norm_docs(dummies)

#fasttext
m = fastText.load_model("cc.sv.300.bin") 
v = m.get_word_vector('hello')
model = FastText.load_fasttext_format('cc.sv.300.bin')


#modify this mothafucka
word_to_vec_map, word_to_index, index_to_words, vocab_size, dim= load_vectors('../input/fasttext-wikinews/wiki-news-300d-1M.vec')


#these has to correspond to the dataframe input and output and X_test for the test data
X = np.array(train.Phrase)
Y = np.array(train.Sentiment)
X_test = np.array(test.Phrase)
print("X.shape", X.shape) 
print("Y.shape", Y.shape)


#to test the CNN from [8]
model = cnn((maxLen,), word_to_vec_map, word_to_index)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
track = model.fit(X_vec, Y, batch_size=128, epochs=9)
#plot acc
plt.plot(track.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
#plot loss
plt.plot(track.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
'''
