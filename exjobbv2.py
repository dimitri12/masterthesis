import matplotlib
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import re
import io
from numpy import array
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import gensim
import nltk
import scikitplot.plotters as skplt
from xgboost import XGBClassifier
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from keras import models
from keras import layers
from keras import regularizers
import fastText
from gensim.models import FastText
from keras.layers import Dense, Input, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, Concatenate, Conv2D, MaxPooling2D, concatenate
from keras.initializers import glorot_uniform
from gensim.models.keyedvectors import KeyedVectors
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
from operator import is_not
from functools import partial
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')

#Global Variables from [7] and others
NB_WORDS = 20  # Parameter indicating the number of words we'll put in the dictionary (to Douglas: i modified this, it was 10000 before)
VAL_SIZE = 10  # Size of the validation set (originally 1000)
NB_START_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 19  # Maximum number of words in a sequence
GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
act = 'relu'
re_weight = True
embed_dim = 128
lstm_out = 196
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5


def test_code(input):
    print("Hello " + input)

#preprocessing function. use the dummies as input
def pre_processing1(input,df):
    sentence = [None] * df.shape[0]
    for i in range(len(sentence)):
        old_sentence = input.iloc[i]
        word = list(old_sentence.split())
        words = [None] * len(word)
        for i in range(len(word)):
            words[i] = re.sub(r'\W+', '', word[i].lower())
        words1 = [x for x in words if x is not None]
        sentence.append(' '.join(words1))
        sentence1 = [x for x in sentence if x is not None]
    values = array(sentence1)
    return values

#this methods is courtesy from [1], it is an alternative preprocessing method that also uses stop words removal and normalization (in the main section)
def pre_processing2(input):
    #np.vectorize(input)
    dummies1=re.sub(r"\w+", " ", input)
    pattern = r"[{}]".format(",.;")
    dummies1=re.sub(pattern, "", input)
    #lower casing
    dummies1= dummies1.lower()
    dummies1 = dummies1.strip()
    WPT = nltk.WordPunctTokenizer()
    #tokenization
    tokens = WPT.tokenize(dummies1)
    #stop words removal
    stop_word_list = nltk.corpus.stopwords.words('swedish')
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    result = ' '.join(filtered_tokens)
    return result

#this function is used to transform string data into float data: e.g. Pout (String) to NewPout (float) using a scoring method where the highest value is the highest priority
def transform_output_data(output_dataframe,datatype=None):
    
    #alternative 1
    data2 = [None] * output_dataframe.shape[0]
    data = output_dataframe
    #float datatype by default
    if datatype is None:
        for i in range(len(data2)):
            if data[i] == '1A':
                data2.append(1.0)
            elif data[i] == '1B':
                data2.append(0.8)
            elif data[i] == '2A':
                data2.append(0.6)
            elif data[i] == '2B':
                data2.append(0.4)
            else:
                data2.append(0.2)
            data1 = [x for x in data2 if x is not None]
        data2 = np.array(data1)
        df_data = pd.DataFrame({'NewPout': data2})
        return df_data
    elif datatype == 'int':
        for i in range(len(data)):
            if data[i] == '1A':
                data2.append(1)
            elif data[i] == '1B':
                data2.append(2)
            elif data[i] == '2A':
                data2.append(3)
            elif data[i] == '2B':
                data2.append(4)
            else:
                data2.append(5)
            data1 = [x for x in data2 if x is not None]
        data2 = np.array(data1)
        df_data = pd.DataFrame({'NewPout': data2})
        return df_data
    elif datatype == 'multi':
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(output_dataframe)
        label_encoded_y = label_encoder.transform(output_dataframe)
        df_data = pd.DataFrame({'NewPout': label_encoded_y})
        return df_data
    #note to self: don't binarize the outputs, if there is binary outputs: use BoW for the binary outputs
def inverse_transform_output_data(input):
    data2 = [None] * input.shape[0]
    data = input
    for i in range(len(data2)):
        if data[i] == 1.0 or data[i] == 1:
            data2.append("1A")
        elif data[i] == 0.8 or data[i] == 2:
            data2.append("1B")
        elif data[i] == 0.6 or data[i] == 3:
            data2.append("2A")
        elif data[i] == 0.4 or data[i] == 4:
            data2.append("2B")
        else:
            data2.append("Referral")
            data1 = [x for x in data2 if x is not None]
    data2 = np.array(data1)
    df_data = pd.DataFrame({'NewPout': data2})
    return df_data
def text_processing(input_data, output_data, processing_method=None, truncated=None,datatype=None):
    #bag-of-words for the none clause
    if processing_method == None:
        #one of these alternatives can be used but it depends on the classification result
        
        #[2]
        #Alternative 1 from [2]
        #try to use aside from word: char or char_wb
        bag_of_words_vector = CountVectorizer(analyzer="word")
        bag_of_words_matrix = bag_of_words_vector.fit_transform(input_data)
        #denna är viktig
        bag_of_words_matrix = bag_of_words_matrix.toarray()
        
        '''
        
        #Alternative 2
        bag_of_words_vector = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(2,2))
        bag_of_words_matrix = bag_of_words_vector.fit_transform(input_data)
        #denna är viktig
        bag_of_words_matrix = bag_of_words_matrix.toarray()
        '''
        #using LSA: Latent Semantic Analysis or LSI
        if truncated == 1:
            svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
            truncated_bag_of_words = svd.fit_transform(bag_of_words_matrix)
            #you can swap bag_of_words with truncated_bag_of_words
            result = feature_engineering(truncated_bag_of_words,output_data)
        result = feature_engineering(bag_of_words_matrix,output_data)
        return result
    elif processing_method =='tfidf':
        
        #[2]
        Tfidf_Vector = TfidfVectorizer(analyzer="char_wb")    
        Tfidf_Matrix = Tfidf_Vector.fit_transform(input_data)
        Tfidf_Matrix = Tfidf_Matrix.toarray()
        '''
        #Alternative 2 from [1]
        Tfidf_Vector = TfidfVectorizer(min_df = 0., max_df = 1., use_idf = True)
        Tfidf_Matrix = Tfidf_Vector.fit_transform(input_data)
        Tfidf_Matrix = Tfidf_Matrix.toarray()
        
        
        #Alternative 3
        Tfidf_Vector = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,1), sublinear_tf=True)
        Tfidf_Matrix = Tfidf_Vector.fit_transform(input_data)
        Tfidf_Matrix = Tfidf_Matrix.toarray()
        '''
       
        
        if truncated == 1:
            svd2 = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
            #do we need to truncate the matrix?
            #do we need to transform Tfidf_Matrix to an array before truncation?
            truncated_tfidf = svd2.fit_transform(Tfidf_Matrix)
            result = feature_engineering(truncated_tfidf,output_data)
        #try to use truncated_tfidf instead tfidf_Matrix to see what happens
        result = feature_engineering(Tfidf_Matrix,output_data)
        return result
    elif processing_method == 'onehot':
        #be warned: one hot encoding only work well with binary outputs
        #originates from [3]
        label_encoder_input = LabelEncoder()
        #label_encoder_output = LabelEncoder()
        print(output_data.shape)
        output1 = output_data.to_numpy()
        array1 = [None] * input_data.shape[0]
        for i in range(len(array1)):
            input = input_data[i].split()
            
            values = array(input)
            values1 = [x for x in values if x is not None]
            array1.append(values1)
        array2 = [x for x in array1 if x is not None]
        array3 = array(array2)
        array4 = np.hstack(array3)
        array4.reshape(-1,len(output1.shape))
        #output1 = output1.reshape(array4.shape)
        #print(array4)
        
        integer_encoded_input = label_encoder_input.fit_transform(array4)
        
        #integer_encoded_output = label_encoder_output.fit_transform(output_data)
        #float by default
        if datatype is None:
            #this method performs one hot encoding to return data of type float
            onehot_encoder_input = OneHotEncoder(sparse=False)

            #using reshaping before encoding
            integer_encoded_input = integer_encoded_input.reshape(-1, 1)
            encoded_input = onehot_encoder_input.fit_transform(integer_encoded_input)
            
            output= transform_output_data(output_data,'multi')
            output1 = output.to_numpy()
            
           
            #encoded_output = onehot_encoder_output.fit_transform(integer_encoded_output)
        if datatype == 'int':
            input_lb = LabelBinarizer()
            encoded_input = input_lb.fit_transform(integer_encoded_input)
            print(encoded_input)
            print(encoded_input.shape)
        #create training and test data using our encoded data
        #change from integer_encoded_output to output_data
        result = feature_engineering(encoded_input,output1)
        
        return result
#split data into train and test data
def feature_engineering(input_data, output_data):
    #alternative 1
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=37)
    '''
    #alternative 2
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3)
    '''
    '''
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=37)
    '''
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    result = [X_train, X_test, y_train, y_test]
    return result
def predictor(data_array,method):
    if method == 'NBGauss':
        NBGres_onehot = initiate_predictions(data_array[0],method)
        NBGres_BoW = initiate_predictions(data_array[1],method)
        NBGres_tfidf = initiate_predictions(data_array[2],method)

        result = [NBGres_onehot,NBGres_BoW,NBGres_tfidf]
    elif method == 'NBMulti':
        NBMres_onehot = initiate_predictions(data_array[0],method)
        NBMres_BoW = initiate_predictions(data_array[1],method)
        NBMres_tfidf = initiate_predictions(data_array[2],method)
        result = [NBMres_onehot,NBMres_BoW,NBMres_tfidf]
    elif method == 'SVM':
        SVMres_onehot = initiate_predictions(data_array[0],method)
        SVMres_BoW = initiate_predictions(data_array[1],method)
        SVMres_tfidf = initiate_predictions(data_array[2],method)
        result = [SVMres_onehot,SVMres_BoW,SVMres_tfidf]
    elif method == 'RF':
        RFres_onehot = initiate_predictions(data_array[0],method)
        RFres_BoW = initiate_predictions(data_array[1],method)
        RFres_tfidf = initiate_predictions(data_array[2],method)
        result = [RFres_onehot,RFres_BoW,RFres_tfidf]
    elif method == 'ensemble':
        res_onehot = initiate_predictions(data_array[0],method)
        res_BoW = initiate_predictions(data_array[1],method)
        res_tfidf = initiate_predictions(data_array[2],method)
        result = [res_onehot,res_BoW,res_tfidf]
    else:
        logres_onehot = initiate_predictions(data_array[0],method)
        logres_BoW = initiate_predictions(data_array[1],method)
        logres_tfidf = initiate_predictions(data_array[2],method)
        result = [logres_onehot,logres_BoW,logres_tfidf]
    return result
#[6] prediction using the processed data
def generate_metrics(result):
    print('Metrics from One hot Encoding on Logreg:')
    print('-'*30)
    result_from_predicitions(result[0])
    print('-'*30)
    print('Metrics from Bag of Words on Logreg:')
    print('-'*30)
    result_from_predicitions(result[1])
    print('-'*30)
    print("plot graph")
    print('Metrics from TF-IDF on Logreg:')
    print('-'*30)
    result_from_predicitions(result[2])
    print('-'*30)
def train_predict_model(classifier,X_train,X_test, y_train, y_test):
    # build model
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if classifier == 'NBGauss':
        model = GaussianNB()
    elif classifier == 'NBMulti':
        model = MultinomialNB()
    elif classifier == 'SVM':
        model = LinearSVC()
    elif classifier == 'RF':
        model = RandomForestClassifier(n_estimators=50, random_state=1)
    elif classifier == 'ensemble':
        model1 = LogisticRegression()
        model2 = MultinomialNB()
        model3 = RandomForestClassifier(n_estimators=50, random_state=1)
        model = VotingClassifier(estimators=[('lr', model1), ('nb', model2), ('rf', model3)], voting='hard')
    else:
        model = LogisticRegression()
    model.fit(X_train, y_train)
    # predict using model
    predicted = model.predict(X_test)
    
    acc = metrics.accuracy_score(y_test,predicted)
    acc = acc*100
    result = [predicted,acc,]
    return result    
def initiate_predictions(train_test_data,method):
    X_train = train_test_data[0]
    X_test = train_test_data[1]
    y_train = train_test_data[2]
    y_test = train_test_data[3]
    prediction = train_predict_model(method,X_train,X_test,y_train,y_test)
    predicted = prediction[0]
    acc = prediction[1]
    true = y_test
    result = [true,predicted,acc]
    return result
def result_from_predicitions(prediction_array):
    '''
    print("Results from prediction:")
    print('-'*30)
    df1=pd.DataFrame({'Actual':true, 'Predicted':predicted})
    print(df1)
    '''
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', np.round(metrics.accuracy_score(prediction_array[0],prediction_array[1]),4))
    print('Precision:', np.round(metrics.precision_score(prediction_array[0],prediction_array[1],average='weighted'),4))
    print('Recall:', np.round(metrics.recall_score(prediction_array[0],prediction_array[1],average='weighted'),4))
    print('F1 Score:', np.round(metrics.f1_score(prediction_array[0],prediction_array[1],average='weighted'),4))
    print('\nModel Classification report:')
    print('-'*30)
    print(metrics.classification_report(prediction_array[0],prediction_array[1]))
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    cm = metrics.confusion_matrix(y_true=prediction_array[0], y_pred=prediction_array[1])
    print(cm)
def create_plot_data(prediction_array,method):
    result_frame = pd.DataFrame(columns=["Method","Accuracy"])
    if method == 'NBGauss':
        NBGresult_name = ["NBGres_onehot","NBGres_BoW","NBGres_tfidf"]
        NBGresult_acc = prediction_array
        for i in range(len(NBGresult_name)):
            NBGresult1= pd.DataFrame([[NBGresult_name[i],NBGresult_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(NBGresult1)
        return result_frame
    elif method == 'NBMulti':
        NBMresult_name = ["NBMres_onehot","NBMres_BoW","NBMres_tfidf"]
        NBMresult_acc = prediction_array
        for i in range(len(NBMresult_name)):
            NBMresult1= pd.DataFrame([[NBMresult_name[i],NBMresult_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(NBMresult1)
        return result_frame
    elif method == 'SVM':
        SVMresult_name = ["SVMres_onehot","SVMres_BoW","SVMres_tfidf"]
        SVMresult_acc = [prediction_array[0],prediction_array[1],prediction_array[2]]
        for i in range(len(SVMresult_name)):
            SVMresult1= pd.DataFrame([[SVMresult_name[i],SVMresult_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(SVMresult1)
        return result_frame
    elif method == 'RF':
        RFresult_name = ["RFres_onehot","RFres_BoW","RFres_tfidf"]
        RFresult_acc = prediction_array
        for i in range(len(RFresult_name)):
            RFresult1= pd.DataFrame([[RFresult_name[i],RFresult_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(RFresult1)
        return result_frame
    elif method == 'ensemble':
        ensemble_result_name = ["ensemble_res_onehot","ensemble_res_BoW","ensemble_tfidf"]
        ensemble_result_acc = prediction_array
        for i in range(len(ensemble_result_name)):
            ensemble_result1= pd.DataFrame([[ensemble_result_name[i],ensemble_result_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(ensemble_result1)
        return result_frame
    elif method == 'log':
        logresult_name = ["logres_onehot","logres_BoW","logres_tfidf"]
        logresult_acc = prediction_array
        for i in range(len(logresult_name)):
            logresult1= pd.DataFrame([[logresult_name[i],logresult_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(logresult1)
        return result_frame
    else:
        result_name = ["lr_OHE","lr_BoW","lr_tfidf","NBG_OHE","NBG_BoW","NBG_tfidf",\
            "NBM_OHE","NBM_BoW","NBM_tfidf", "SVM_OHE","SVM_BoW","SVM_tfidf",\
            "RF_OHE","RF_BoW","RF_tfidf","ensemble_OHE","ensemble_BoW","ensemble_tfidf"]
        result_acc = prediction_array
        for i in range(len(result_name)):
            result1= pd.DataFrame([[result_name[i],result_acc[i]]], columns=["Method","Accuracy"])
            result_frame = result_frame.append(result1)
        return result_frame
#[6] plot data
#plt.cm.RdYlBu
def plot_metrics(result_frame):
    sns.set_color_codes("muted")
    sns.barplot(x='Method', y='Accuracy', data=result_frame, color="r")

    plt.xlabel('Accuracy Method')
    plt.title('Classifier Accuracy Percent')
    plt.show()
#AUCROC index 2
def ROCCurves(true,pred,encoding, method):
    return None
#perform word embeddings inputs: dataframe input data and output data; output: embedded data such as X train and test and y train and test
def word_embeddings(input_data, output_data):
    '''
    #create validation data
    val_data = input_data.sample(frac=0.2,random_state=1)
    train_data = input_data.drop(val_data.index)
    '''
    data_out = we_output_data_transform(output_data,'int')
    data = feature_engineering(input_data, data_out)
    #index 0 = X_train
    #index 1 = X_test
    #index 2 = y_train
    #index 3 = y_test
    assert data[0].shape[0] == data[2].shape[0]
    assert data[1].shape[0] == data[3].shape[0]
    data_in1 = tokenizer(data[0], data[1])
    data_in2 = padding(data_in1[0], data_in1[1])
    
    
    
    #create validation data
    global MAX_LEN
    MAX_LEN = input_data.shape[0]
    data2 = feature_engineering(data_in2[0], data[2])
    assert data2[1].shape[0] == data2[3].shape[0]
    assert data2[0].shape[0] == data2[2].shape[0]
    
    '''
    #fasttext (word_to_vec_map, word_to_index, index_to_words, vocab_size, dim)
    #tip: try to swap the sv.vec file with cc.sv.300.vec
    
    
    callbacks = [EarlyStopping(monitor='val_loss')]

    #load fasttext data into cnn1
    array = load_vectors2('./data/fasttext/sv.vec')
    embedding_layer1 = pretrained_embedding_layer(array[0], array[1])
    embedding_layer=load_vectors_word2vec('./data/word2vec/sv.bin',data_in1[2])
    #change data_in2[0].shape[1] with (MAX_LEN,)
    model = cnn1((MAX_LEN,),embedding_layer1)
    adam = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])
    track = model.fit(data_in2[0], data[2], batch_size=128, epochs=10, verbose=1, validation_data=(data2[0], data2[2]),callbacks=callbacks)
    plot_function(track)
    preds = model.predict(data_in2[1])
    scores = model.evaluate(data_in2[1], data[3], verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #word2vec data on cnn2
    #you can swap data_in2[0].shape[1] for MAX_LEN
    model1 = cnn2(data_in2[0].shape[1],embedding_layer)
    model1.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])
    #we can use validation data for our model
    track2 = model1.fit(data_in2[0], data[2], batch_size=1000, epochs=10, verbose=1, validation_data=(data2[0], data2[2]),callbacks=callbacks)
    plot_function(track2)
    y_pred=model1.predict(data_in2[1])
    scores = model1.evaluate(data_in2[1], data[3], verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #word2vec on cnn1
    model2 = cnn1((MAX_LEN,),embedding_layer)
    adam = Adam(lr=1e-3)
    model2.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])
    track3 = model2.fit(data_in2[0], data[2], batch_size=128, epochs=10, verbose=1, validation_data=(data2[0], data2[2]),callbacks=callbacks)
    plot_function(track3)
    preds1 = model2.predict(data_in2[1])
    scores = model2.evaluate(data_in2[1], data[3], verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    
    #GRU
    GRU_model = gru_model()
    # Train
    track2 = GRU_model.fit(data_in2[0], data[2], epochs=3, batch_size=64)
    plot_function(track2)
    
    # Predict the label for test data
    y_predict = GRU_model.predict(data_in2[1])
    # Final evaluation of the model
    scores = GRU_model.evaluate(data_in2[1], data[3], verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    #LSTM
    LSTM_model = lstm_model()
    # Train
    track3 = LSTM_model.fit(data_in2[0], data[2], epochs=3, batch_size=64)
    plot_function(track3)
    # Predict the label for test data
    y_predict1 = LSTM_model.predict(data_in2[1])
    # Final evaluation of the model
    scores1 = LSTM_model.evaluate(data_in2[1], data[3], verbose=1)
    print("Accuracy: %.2f%%" % (scores1[1]*100))

    #CNN1,CNN2,LSTM,GRU
    preds = preds.argmax(axis=-1)
    preds1 = preds1.argmax(axis=-1)
    y_pred = y_pred.argmax(axis=-1)
    y_predict = y_predict.argmax(axis=-1)
    y_predict1 = y_predict1.argmax(axis=-1)
    result = [preds,y_pred,preds1,y_predict,y_predict1]
    
    #result = [data[0],data[1],data[2],data[3]]
    '''
    result=None
    return result

def plot_function(track):
    plt.plot(track.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    plt.plot(track.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
#tokenizes the words
def tokenizer(train_data, test_data):
    #from [7]
    tk = Tokenizer(num_words=NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tk.fit_on_texts(train_data)
    trained_seq = tk.texts_to_sequences(train_data)
    test_seq = tk.texts_to_sequences(test_data)
    word_index = tk.word_index
    result = [trained_seq, test_seq, word_index]
    return result

#test function from [7] to make sure that the sequences generated from the tokenizer function are of equal length
def test_sequence(train_data):
    seq_lengths = train_data.apply(lambda x: len(x.split(' ')))
    print("The sequences generated are:")
    seq_lengths.describe()
    print("----------------")

#from [7]
def deep_model(model, X_train, y_train, X_valid, y_valid):
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
    history = model.fit(X_train
                       , y_train
                       , epochs=NB_START_EPOCHS
                       , batch_size=BATCH_SIZE
                       , validation_data=(X_valid, y_valid)
                       , verbose=1)
    return history
def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()
def test_model(model, X_train, y_train, X_test, y_test, epoch_stop):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test, y_test)
    
    return results
#in [7] padding is used to fill out null values
def padding(trained_seq, test_seq):
    #you can change MAX_LEN to train_data.shape[1]
    trained_seq_trunc = pad_sequences(trained_seq)
    test_seq_trunc = pad_sequences(test_seq, maxlen=MAX_LEN)
    result = [trained_seq_trunc, test_seq_trunc]
    return result

def we_output_data_transform(y_data,encoding=None):
    if encoding == None:
        y_data = to_categorical(np.asarray(y_data))
    else:
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_data)
        y_data = to_categorical(y_train_le)
    result = y_data
    return result

#embeddings layer
def embeddings_layer(X_train_emb, X_valid_emb,y_train_emb,y_valid_emb):
    emb_model = models.Sequential()
    emb_model.add(layers.Embedding(NB_WORDS, 8, input_length=MAX_LEN))
    emb_model.add(layers.Flatten())
    emb_model.add(layers.Dense(3, activation='softmax'))
    emb_model.summary()
    emb_history = deep_model(emb_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)
    result = emb_history
    return result
#[7]
def we_evaluation(emb_model,X_train_seq_trunc,X_test_seq_trunc,y_train_oh,y_test_oh):
    eval_metric(emb_model, 'acc')
    eval_metric(emb_model, 'loss')
    emb_results = test_model(emb_model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 6)
    print('/n')
    print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1]*100))

#load vec file from fasttext, from [8]
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab_size, dim = map(int, fin.readline().split())
    word_to_vec_map = {}
    words = set()
    for line in fin:
        tokens = line.rstrip().split(' ')
        words.add(tokens[0])
        word_to_vec_map[tokens[0]] = np.array(tokens[1:], dtype=np.float64)
    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1
    return word_to_vec_map, words_to_index, index_to_words, vocab_size, dim
#caller function for load_vectors .vec file
def load_vectors2(fname):
    word_to_vec_map, words_to_index, index_to_words, vocab_size, dim = load_vectors(fname)
    result = [word_to_vec_map, words_to_index, index_to_words, vocab_size, dim]
    return result

#from[8]
def sentences_to_indices(X, word_to_index, maxLen):
    m = X.shape[0]                                   # number of training examples
    print(m)
    X_indices = np.zeros((m, maxLen))
    for i in range(m):
        sentence_words = X[i].lower().strip().split()
        j = 0
        for w in sentence_words:
            if w not in word_to_index:
                w = "person"        #mostly names are not present in vocabulary
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices

#for Word2Vec input word2vec .bin file and word_index = tokenizer.word_index from tokenizer
def load_vectors_word2vec(fname,word_index):
    word_vectors = KeyedVectors.load(fname)
    vocabulary_size = min(MAX_NB_WORDS, len(word_index))+1
    print(vocabulary_size)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        if i>=MAX_NB_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
    del(word_vectors)
    embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)
    return embedding_layer
#[8]
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    #emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_dim = 300
    '''
    #or
    emb_dim = 300
    '''
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False) 

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def cnn2(input_shape,embedding_layer1):
    print(input_shape)
    sentence_indices = Input(shape=(input_shape,))
    
    embedding = embedding_layer1(sentence_indices)
    
    reshape = Reshape((input_shape,EMBEDDING_DIM,1))(embedding)
    conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((input_shape - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
    maxpool_1 = MaxPooling2D((input_shape - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
    maxpool_2 = MaxPooling2D((input_shape - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3*num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=5, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

    # this creates a model that includes
    model = Model(sentence_indices, output)
    print(model.summary())
    return model
#from [8] a CNN approach
def cnn1(input_shape,embedding_layer1):
    
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = embedding_layer1(sentence_indices) 

    X1 = Conv1D(128, 3)(embeddings)
    X2 = Conv1D(128, 3)(embeddings)
    X1 = MaxPooling1D(pool_size=4)(X1)
    X2 = MaxPooling1D(pool_size=5)(X2)
    X = Concatenate(axis=1)([X1, X2])
    X = GRU(units=128, dropout=0.4, return_sequences=True)(X)
    X = LSTM(units=128, dropout=0.3)(X)
    X = Dense(units = 32, activation="relu")(X)
    X = Dense(units=5, activation='softmax')(X)
    model = Model(inputs=sentence_indices, outputs=X)
    print(model.summary())
    return model

def gru_model():
    model = Sequential()
    model.add(Embedding(NB_WORDS, 100, input_length=MAX_LEN))
    model.add(GRU(100))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model

def lstm_model():
    model = Sequential()
    model.add(Embedding(NB_WORDS, embed_dim,input_length = MAX_LEN))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    return model

'''
[1] https://medium.com/deep-learning-turkey/text-processing-1-old-fashioned-methods-bag-of-words-and-tfxidf-b2340cc7ad4b, Medium, Deniz Kilinc visited 6th of April 2019
[2] https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm, from ReiiNakano , Kaggle, visited 5th of April 2019
[3] https://github.com/codebasics/py/tree/master/ML, github, Codebasics from dhavalsays, visited 6th of April 2019
[4] from scikit-learn.org (base code), visited 4th of April 2019
[5] Python One Hot Encoding with SciKit Learn, InsightBot, http://www.insightsbot.com/blog/McTKK/python-one-hot-encoding-with-scikit-learn, visited 6th April 2019 
[6] Kaggle, Sentiment Analysis : CountVectorizer & TF-IDF, Divyojyoti Sinha, https://www.kaggle.com/divsinha/sentiment-analysis-countvectorizer-tf-idf
[7] Kaggle, Bert Carremans, Using Word Embeddings for Sentiment Analysis, https://www.kaggle.com/bertcarremans/using-word-embeddings-for-sentiment-analysis, visited april 11th 2019
[8] Sentiment Analysis with pretrained Word2Vec, Varun Sharma, Kaggle, https://www.kaggle.com/varunsharmaml/sentiment-analysis-with-pretrained-word2vec, visited 12th of april 2019
'''