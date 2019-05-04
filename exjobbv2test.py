import sys
import pandas as pd
sys.path.append("/Users/dimitrigharam/Desktop/exjobb")
import exjobbv2 as run
import nltk
from nltk.corpus import stopwords 
import numpy as npa
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
result_frame = pd.DataFrame(columns=["Method","Accuracy"])
df = pd.read_csv("testdata_exjobb.csv", encoding='ISO-8859-1')
labels = run.create_labels_list(df)

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
array = run.doMultiClass(df)
print(array[0])
print(array[1])
print(array[2])
print(array[3])
print(array[4])
#print(df)
#it works
'''
Perform an Explorative data analysis
'''

#run.eda(df)

#load output data and define if it is multiclass or not
#input_data = df.FreeText
#pout,prio,operator,agecat,lcdcat

'''



Part 1: Text Classification with BoW and TF-IDF

#CHALLENGE: solve the multiclass classification problem


'''
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
input_data = preproc2
#print(outputdata2)
print("Inverse Transforming:")
inv_output = run.inverse_transform_output_data(outputdata.NewPout)
print("Embedding/Encoding/Processing starting")
print("----------------------------------")
#the result from these encodings are arrays containing training and test data for both the input and output
#in order: input_train, input_test, output_train, output_test
bag_of_words2 = run.text_processing(input_data,outputdata2, None, 1)
tf_idf2 = run.text_processing(input_data,outputdata2, "tfidf", 1)
data_array = [bag_of_words2,tf_idf2]
print(tf_idf2)

print("Embedding/Encoding/Processing is finished")
'''
print("##################################")
print("Modeling and prediction:")
print("----------------------------------")

#result = run.doMultiClasspart2(input_data,df)
#run.plot_metrics(create_plot_data(result,None))
#run.plot_roc(result[1][0],result[1][1])
#result1 = run.doBinary2(input_data,df,labels)
#run.plot_metrics(create_plot_data(result1,None))
#run.plot_roc(result1[1][0],result1[1][1])
#result2 = run.doBinary1part2(input_data,df)
#run.plot_metrics(create_plot_data(result2,None))
#run.plot_roc(result2[1][0],result1[2][1])

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

#print(outputdata)
#test = run.word_embeddings(input_data, output_data)
print("##################################")
print("Done")