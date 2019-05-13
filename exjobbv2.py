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
from nltk.corpus import stopwords
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
from keras.constraints import unit_norm
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
from gensim.models import FastText
from keras.layers import Dense, Input, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, Concatenate, Conv2D, MaxPooling2D, concatenate
from keras.initializers import glorot_uniform
from gensim.models.keyedvectors import KeyedVectors
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
from operator import is_not
from functools import partial
from gensim.models.wrappers import FastText
import matplotlib.patches as patches
import time
import datetime
import warnings
import pickle
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')

#Global Variables from [7] and others
NB_WORDS = 116  # Parameter indicating the number of words we'll put in the dictionary, 
#there is a variable in the tokenizer function that can help 
VAL_SIZE = 9  # Size of the validation set (originally 1000)
NB_START_EPOCHS = 8  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 72  # Maximum number of words in a sequence
#MAX_LEN = 62
GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
#activation function
act = 'relu'
re_weight = True
#dimension for fasttext
DIM = 300
#dimension for word2vec
embed_dim = 256
lstm_out = 196
dim = 50
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5
maxLen = 60
kernel_size = 5
filters = 64
pool_size = 4
lstm_output_size = 70



def transformAge(df):
    '''
    This function will categorise the age data into 5 different age categories
    '''
    dfage = df.Age

    array = []
    for i in range(len(dfage)):
        if dfage[i] < 3:
            array.append(0)
        elif dfage[i] >= 3 and dfage[i] < 13:
            array.append(1)
        elif dfage[i] >= 13 and dfage[i] < 19:
            array.append(2)
        elif dfage[i] >= 19 and dfage[i] < 65:
            array.append(3)
        else:
            array.append(4)
    df["AgeCat"] = array
    return df

def transformLCD(df):
    '''
    This function will transform the numerical LCD data into 3 categories
    '''
    dflcd = df.LastContactDays
    array = []
    for i in range(len(dflcd)):
        if dflcd[i] < 2:
            array.append(0)
        elif dflcd[i] >= 2 and dflcd[i] < 31:
            array.append(1)
        else:
            array.append(2)
    df["LcdCat"] = array
    return df

def test_code(input):
    print("Hello " + input)

def eda(df):
    figsize=(20, 10)
    
    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    params = {'figure.figsize' : figsize,
            'axes.labelsize' : labelsize,
            'axes.titlesize' : titlesize,
            'xtick.labelsize': ticksize,
            'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    plt.subplot(441)
    x1=df['prio']
    ylabel = "Count"
    sns.countplot(x1)
    plt.xticks(rotation=90)
    plt.title('Priority')
    plt.xlabel('')
    plt.ylabel(ylabel)


    plt.subplot(442)
    x2=df['operator']
    ylabel = "Count"
    sns.countplot(x2)
   # plt.title("Review Sentiment Count")
    plt.title('Operator')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.plot()
    
    #pout
    plt.subplot(443)
    x3=df['pout']
    ylabel = "Count"
    sns.countplot(x3)
    plt.title('Pout')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)

    plt.subplot(444)
    x4=df['hosp_ed']
    ylabel = "Count"
    sns.countplot(x4)
    plt.title('Hospitaled')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)

    plt.subplot(449)
    x4=df['AgeCat']
    ylabel = "Count"
    sns.countplot(x4)
    plt.title('Age')
    plt.xlabel('')
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    plt.subplot(4,4,10)
    x4=df['Gender']
    ylabel = "Count"
    sns.countplot(x4)
    plt.title('Gender')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)

    plt.subplot(4,4,11)
    x4=df['LastContactN']
    ylabel = "Count"
    sns.countplot(x4)
    plt.title('LastContactN')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)

    plt.subplot(4,4,12)
    x4=df['LcdCat']
    ylabel = "Count"
    sns.countplot(x4)
    plt.title('LastContactDays')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)

    plt.plot()
    plt.show()

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
def predictor(data_array,method,multiclass):
    if method == 'NBG':
        NBGres_BoW = initiate_predictions(data_array[0],method,multiclass)
        NBGres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [NBGres_BoW,NBGres_tfidf]
        return result
    elif method == 'NBM':
        NBMres_BoW = initiate_predictions(data_array[0],method,multiclass)
        NBMres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [NBMres_BoW,NBMres_tfidf]
        return result
    elif method == 'SVM':
        SVMres_BoW = initiate_predictions(data_array[0],method,multiclass)
        SVMres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [SVMres_BoW,SVMres_tfidf]
        return result
    elif method == 'RF':
        RFres_BoW = initiate_predictions(data_array[0],method,multiclass)
        RFres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [RFres_BoW,RFres_tfidf]
        return result
    elif method == 'ensemble':
        res_BoW = initiate_predictions(data_array[0],method,multiclass)
        res_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [res_BoW,res_tfidf]
        return result
    elif method == 'GB':
        GBres_BoW = initiate_predictions(data_array[0],method,multiclass)
        GBres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [GBres_BoW,GBres_tfidf]
        return result
    else:
        logres_BoW = initiate_predictions(data_array[0],method,multiclass)
        logres_tfidf = initiate_predictions(data_array[1],method,multiclass)
        result = [logres_BoW,logres_tfidf]
        #pred_prob is at index 4
    return result
#[6] prediction using the processed data
def generate_metrics(result,clf=None):
    if clf == 'NBG':
        val = 'Naive Bayes Gaussian'
    elif clf == 'NBM':
        val = 'Naive Bayes Multinomial'
    elif clf == 'SVM':
        val = 'Support Vector Machine'
    elif clf == 'RF':
        val = 'Random Forest'
    elif clf == 'ensemble':
        val = 'Ensemble'
    elif clf == 'gb':
        val = 'Gradient Boosting'
    else:
        val = 'logistic regression'
    print('Metrics from Bag of Words on '+ val +':')
    print('-'*30)
    result_from_predicitions(result[0])
    print('-'*30)
    print('Metrics from TF-IDF on '+ val +':')
    print('-'*30)
    result_from_predicitions(result[1])
    print('-'*30)
    print("plot graph")
    print('-'*30)
def train_predict_model(classifier,X_train,X_test, y_train, y_test,multiclass):
    # build model
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if classifier == 'NBG':
        model = GaussianNB()
    elif classifier == 'NBM':
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
    elif classifier == 'GB':
        model = XGBClassifier(n_estimators=100)
    else:
        if multiclass == 'yes':
            model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
        else:
            model = LogisticRegression()
    model.fit(X_train, y_train)
    # predict using model
    predicted = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)
    acc = metrics.accuracy_score(y_test,predicted)
    acc = acc*100
    if classifier == 'None':
        loss = log_loss(y_test,predicted)
        result = [predicted,acc,loss,pred_prob[:,1]]
    else:
    
        result = [predicted,acc,pred_prob[:,1]]
    return result    
def initiate_predictions(train_test_data,method,multiclass):
    X_train = train_test_data[0]
    X_test = train_test_data[1]
    y_train = train_test_data[2]
    y_test = train_test_data[3]
    prediction = train_predict_model(method,X_train,X_test,y_train,y_test,multiclass)
    predicted = prediction[0]
    acc = prediction[1]
    true = y_test
    
    pred_prob = prediction[2]
    if method == None:
        loss = prediction[2]
        pred_prob = prediction[3]
        result = [true,predicted,acc,loss,pred_prob]
    else:
        result = [true,predicted,acc,pred_prob]
    return result
def result_from_predicitions(prediction_array):
    
    print("Results from prediction:")
    print('-'*30)
    df1=pd.DataFrame({'Actual':prediction_array[0], 'Predicted':prediction_array[1]})
    print(df1)
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
#AUCROC for binary class only
def plot_roc(pred_data, test_data):
    #courtesy of DATAI https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv
    # plot arrows, why? to present accuracy
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111,aspect = 'equal')
    ax1.add_patch(
        patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
        )
    ax1.add_patch(
        patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
        )
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    for i in range(len(pred_data)):
        fpr, tpr, _ = roc_curve(test_data, pred_data)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
            label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()
#perform predictions on classification methods
def clf_predictor(input_data,multiclass):
    result_logregr = predictor(input_data,None,multiclass)
    result_NBG = predictor(input_data,'NBGauss',multiclass)
    result_NBM = predictor(input_data,'NBMulti',multiclass)
    result_SVM = predictor(input_data,'SVM',multiclass)
    result_RF = predictor(input_data,'RF',multiclass)
    result_ensemble = predictor(input_data,'ensemble',multiclass)
    result_GB = predictor(input_data,'gb',multiclass)
    print("Evaluation of predictions:")
    print("----------------------------------")
    generate_metrics(result_logregr)
    #plots the graph for accuracies over different encoding metods for one particular classification method
    print("Logistic Regression:")
    print("----------------------------------")
    result_logres_acc = [result_logregr[0][2],result_logregr[1][2]]
    result_logres_loss = [result_logregr[0][3],result_logregr[1][3]]
    print("----------------------------------")
    if multiclass == 'no':
        print("Naive Bayes (Gaussian):")
        generate_metrics(result_NBG,'NBG')
        result_NBG_acc = [result_NBG[0][2],result_NBG[1][2]]
        result_NBG_loss = [result_NBG[0][3],result_NBG[1][3]]
        print("----------------------------------")
    elif multiclass == 'yes':
        print("Naive Bayes (Multinomial):")
        print("----------------------------------")
        generate_metrics(result_NBM,'NBM')
        result_NBM_acc = [result_NBM[0][2],result_NBM[1][2]]
        result_NBM_loss = [result_NBM[0][3],result_NBM[1][3]]
        print("----------------------------------")
    print("Support Vector Machine:")
    print("----------------------------------")
    generate_metrics(result_SVM,'SVM')
    result_SVM_acc = [result_SVM[0][2],result_SVM[1][2]]
    result_SVM_loss = [result_SVM[0][3],result_SVM[1][3]]
    print("----------------------------------")
    print("Gradient Boosting")
    print("----------------------------------")
    generate_metrics(result_GB,'gb')
    result_GB_acc = [result_GB[0][2],result_GB[1][2]]
    result_GB_loss = [result_GB[0][3],result_GB[1][3]]
    print("----------------------------------")
    print("Random Forest:")
    print("----------------------------------")
    generate_metrics(result_RF,'RF')
    result_RF_acc = [result_RF[0][2],result_RF[1][2]]
    result_RF_loss = [result_RF[0][3],result_RF[1][3]]
    print("----------------------------------")
    print("Ensemble of Logistic Regression, Naive Bayes (Multinomial) and Random Forest:")
    print("----------------------------------")
    generate_metrics(result_ensemble,'ensemble')
    result_ensemble_acc = [result_ensemble[0][2],result_ensemble[1][2]]
    result_ensemble_loss = [result_ensemble[0][3],result_ensemble[1][3]]
    print("----------------------------------")
    result_acc = [result_logres_acc[0],result_logres_acc[1], \
    result_NBG_acc[0],result_NBG_acc[1], \
    result_NBM_acc[0],result_NBM_acc[1], \
    result_SVM_acc[0],result_SVM_acc[1], \
    result_RF_acc[0],result_RF_acc[1], \
    result_ensemble_acc[0],result_ensemble_acc[1],result_GB_acc[0],result_GB_acc[1]]

    result_loss = [result_logres_loss[0],result_logres_loss[1], \
    result_NBG_loss[0],result_NBG_loss[1], \
    result_NBM_loss[0],result_NBM_loss[1], \
    result_SVM_loss[0],result_SVM_loss[1], \
    result_RF_loss[0],result_RF_loss[1], \
    result_ensemble_loss[0],result_ensemble_loss[1],result_GB_loss[0],result_GB_loss[1]]

    proba_data = [result_logregr[0][4],result_logregr[1][4],result_NBG[0][4],result_NBG[1][4],\
        result_NBM[0][4],result_NBM[1][4],result_SVM[0][4],result_SVM[1][4],\
        result_GB[0][4],result_GB[1][4],result_RF[0][4],result_RF[1][4],\
        result_ensemble[0][4],result_ensemble[1][4]]
    test_data = [result_logregr[0][1],result_logregr[1][1],result_NBG[0][1],result_NBG[1][1],\
        result_NBM[0][1],result_NBM[1][1],result_SVM[0][1],result_SVM[1][1],\
        result_GB[0][1],result_GB[1][1],result_RF[0][1],result_RF[1][1],\
        result_ensemble[0][1],result_ensemble[1][1]]
  
    result1 = [result_acc, result_loss]
    result2 = [proba_data,test_data]
    result=[result1,result2]
    return result
#perform word embeddings inputs: dataframe input data and output data; output: embedded data such as X train and test and y train and test
def remove_stopwords(input_text):
    stopwords_list = stopwords.words('swedish')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1] 
    return " ".join(clean_words) 
#category is binary by default
def word_embeddings(input_data, output_data,ANN,dense,el,category=None):
    
    #Don't use stop words removal for deep learning
    #input_data = input_data.apply(remove_stopwords)
    
    data = feature_engineering(input_data, output_data)
    data_out = we_output_data_transform(data[2],data[3])
    '''
    if category == None:
        data = feature_engineering(input_data, output_data)
        data_out = we_output_data_transform(data[2],data[3])
    else:
        with open('df.pickle', 'rb') as handle:
            df = pickle.load(handle)
        count = len(df.index)
        data = feature_engineering(input_data, output_data)
        dataset,data_out = we_output_data_transform(data[2],data[3],'multi')
        data_in1 = multivectorizer(dataset)
        data_in2 = feature_engineering(data_in1[0], data_out[:count])
        #validation data
        data2 = feature_engineering(data_in2[0], data_out[0])
    '''
    
    #index 0 = X_train
    #index 1 = X_test
    #index 2 = y_train
    #index 3 = y_test
    assert data[0].shape[0] == data[2].shape[0]
    assert data[1].shape[0] == data[3].shape[0]
    data_in1 = tokenizer(input_data,data[0], data[1])
    
    print(data_in1[2])
    data_in2 = padding(data_in1[0], data_in1[1],input_data)
    
    
    #create validation data 
    data2 = feature_engineering(data_in2[0], data_out[0])
    print(data2[0].shape)
    #data2[0] = X_val
    #data2[2] = y_val
    assert data2[1].shape[0] == data2[3].shape[0]
    assert data2[0].shape[0] == data2[2].shape[0]
    #fasttext (word_to_vec_map, word_to_index, index_to_words, vocab_size, dim)
    #tip: try to swap the sv.vec file with cc.sv.300.vec
    #load fasttext data into cnn1
    #save
    with open('data_in2.pickle', 'wb') as handle:
        pickle.dump(data_in2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data_out.pickle', 'wb') as handle:
        pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    with open('data_in.pickle', 'rb') as handle:
        data_in = pickle.load(handle)
    with open('data_out.pickle', 'rb') as handle:
        data_out = pickle.load(handle)
    '''
   
    corpus = load_vectors2('./data/fasttext/sv.vec')
    #data_train = sentences_to_indices(data[0],corpus[1],len(data_in1[2]))
    #data_test = sentences_to_indices(data[1],corpus[1],len(data_in1[2]))

    embedding_layer1 = pretrained_embedding_layer(corpus[0], corpus[1])
    print(corpus[4])
    embedding_layer0=load_vectors_word2vec('./data/word2vec/sv.bin',data_in1[2])
    embedding_layer = [embedding_layer0,embedding_layer1]
    #second preprocessing method
    #change data_in2[0].shape[1] with (MAX_LEN,)
    
    start_time = time.time()
    print(date_time(1))
    model = predict_model(MAX_LEN,embedding_layer[el],ANN,data_in2[0],data_out[0],data2[0],data2[2],dense)
    #or
    #model = predict_model(MAX_LEN,embedding_layer[el],ANN,data_train,data_out[0],None,None,dense)
    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\nElapsed Time: " + elapsed_time)
    print("Completed Model Trainning", date_time(1))
    with open('model'+ANN+'.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    with open('model'+ANN+'.pickle', 'rb') as handle:
        model = pickle.load(handle)
    '''
    #data_in2[0] = X_train
    #data[2] = y_train
    return [data_in2, data2, data]

def multivectorization(concated):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(concated['freetext'].values)
    sequences = tokenizer.texts_to_sequences(concated['freetext'].values)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    result = [X,word_index]
    return result
def we_evaluation(data1,data2,data3,data4,ANN1,ANN2 ,datax,datay):
    
    with open('model'+ANN1+'.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open('model'+ANN2+'.pickle', 'rb') as handle:
        model1 = pickle.load(handle)
    preds1 = model.predict(data1[1])
    preds2 = model1.predict(data3[1])
    preds1 = np.argmax(preds1, axis=-1)
    preds2 = np.argmax(preds2, axis=-1)
    print(datax[3])
    print('#'*200)
    print(datay[3])
    with open('preds1.pickle', 'wb') as handle:
        pickle.dump(preds1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('preds2.pickle', 'wb') as handle:
        pickle.dump(preds2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('test_data1.pickle', 'wb') as handle:
        pickle.dump(datax[3], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('test_data2.pickle', 'wb') as handle:
        pickle.dump(datay[3], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    #or
    #preds2 = model.predict_classes(data_in2[1])
    '''
    Call the metrics function
    '''

  
    test1 = ANN1
    test2 = ANN2
 
    accuracy1 = model.evaluate(data2[1], data2[3], verbose=1)
    print("Model Performance of model 1: "+ test1 +" (Test Accuracy):")
    print('Accuracy: {:0.2f}%\nLoss: {:0.3f}\n'.format(accuracy1[1]*100, accuracy1[1]))
    accuracy2 = model1.evaluate(data4[1], data4[3], verbose=1)
    print("Model Performance of model 2: "+ test2 +" (Test Accuracy):")
    print('Accuracy: {:0.2f}%\nLoss: {:0.3f}\n'.format(accuracy2[1]*100, accuracy2[1]))
    
    
    '''
    result1 = pd.DataFrame({'model': test1, 'score': accuracy1[1]*100}, index=[-1])
    result2 = pd.DataFrame({'model': test2, 'score': accuracy2[1]*100}, index=[-1])
    result = pd.concat([result2, result1.ix[:]]).reset_index(drop=True)
    plot_model_performace(result)
    '''
    #load pickle
    with open('preds1.pickle', 'rb') as handle:
        preds1 = pickle.load(handle)
    
    with open('preds2.pickle', 'rb') as handle:
        preds2 = pickle.load(handle)
    
    with open('test_data1.pickle', 'rb') as handle:
        test_data1 = pickle.load(handle)
    
    with open('test_data2.pickle', 'rb') as handle:
        test_data2 = pickle.load(handle)
   
    '''
    test_data1 = pd.Series(test_data1)
    test_data2 = pd.Series(test_data2)
    '''
    print("Results from prediction:")
    print('-'*200)
    df1=pd.DataFrame({'Actual':test_data1, 'Predicted':preds1})
    print(df1)
    print('-'*200)
    df2=pd.DataFrame({'Actual':test_data2, 'Predicted':preds2})
    print(df2)
    df1.to_csv('prediction1.csv', encoding='utf-8', index=True)
    df2.to_csv('prediction2.csv', encoding='utf-8', index=True)
def sentences_to_indices(X, word_to_index, maxLen):
    m = X.shape[0] 
    X = np.array(X)
    X_indices = np.zeros((m, maxLen))
    
    
    for i in range(m):
        sentence_words = X[i].lower().strip().split()
        j = 0
        for w in sentence_words:
            if w not in word_to_index:
                w = "person"  
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices
def plot_function(track):
    plt.subplot(221)
    plt.plot(track.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    
    plt.subplot(222)
    plt.plot(track.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
#tokenizes the words
def tokenizer(input_data,train_data, test_data):
    #from [7]
    seq_lengths = input_data.apply(lambda x: len(x.split(' ')))
    print(seq_lengths.describe())
    
    
    max_sequence_len = max([len(x) for x in input_data])
    print(max_sequence_len)
    
    tk = Tokenizer(num_words=max_sequence_len,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tk.fit_on_texts(input_data)
    trained_seq = tk.texts_to_sequences(train_data)
    test_seq = tk.texts_to_sequences(test_data)
    word_index = tk.word_index
    result = [trained_seq, test_seq, word_index]
    return result
def one_hot_seq(seqs, nb_features = NB_WORDS):
    ohs = np.zeros((len(seqs), nb_features))
    for i, s in enumerate(seqs):
        ohs[i, s] = 1.
    return ohs
#test function from [7] to make sure that the sequences generated from the tokenizer function are of equal length
def test_sequence(train_data):
    seq_lengths = train_data.apply(lambda x: len(x.split(' ')))
    print("The sequences generated are:")
    seq_lengths.describe()
    print("----------------")
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today() 
def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()
#in [7] padding is used to fill out null values
def padding(trained_seq, test_seq,input):
    
    
    trained_seq_trunc = pad_sequences(trained_seq, maxlen=MAX_LEN)
    
    test_seq_trunc = pad_sequences(test_seq, maxlen=MAX_LEN)
    result = [trained_seq_trunc, test_seq_trunc]
    return result

def we_output_data_transform(y_train,y_test,encoding='binary',category=None):
    if encoding == 'multi':
        with open('df.pickle', 'rb') as handle:
            df = pickle.load(handle)
        df1 = df['freetext']
        df2 = df['pout']
        frames = [df1,df2]
        data = pd.concat(frames, axis=1, sort=False)
        num_of_categories = df.count+1
        shuffled = data.reindex(np.random.permutation(data.index))
        prio1A = shuffled[shuffled['pout'] == '1A'][:num_of_categories]
        prio1B = shuffled[shuffled['pout'] == '1B'][:num_of_categories]
        prio2A = shuffled[shuffled['pout'] == '2A'][:num_of_categories]
        prio2B = shuffled[shuffled['pout'] == '2B'][:num_of_categories]
        Referal = shuffled[shuffled['pout'] == 'Referal'][:num_of_categories]
        concated = pd.concat([prio1A,prio1B,prio2A,prio2B,Referal], ignore_index=True)
        concated = concated.reindex(np.random.permutation(concated.index))
        concated['LABEL'] = 0
        concated.loc[concated['pout'] == '1A', 'LABEL'] = 0
        concated.loc[concated['pout'] == '1B', 'LABEL'] = 1
        concated.loc[concated['pout'] == '2A', 'LABEL'] = 2
        concated.loc[concated['pout'] == '2B', 'LABEL'] = 3
        concated.loc[concated['pout'] == 'Referal', 'LABEL'] = 4
        labels = to_categorical(concated['LABEL'], num_classes=5)
        if 'pout' in concated.keys():
            concated.drop(['pout'], axis=1)
        return concated,labels
    else:
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.fit_transform(y_test)
        y_train = to_categorical(y_train_le)
        y_test = to_categorical(y_test_le)
    result = [y_train,y_test]
    return result
#embeddings layer
def embeddings_layer(X_train_emb, X_valid_emb,y_train_emb,y_valid_emb,dense):
    emb_model = models.Sequential()
    emb_model.add(layers.Embedding(NB_WORDS, DIM, input_length=MAX_LEN))
    emb_model.add(layers.Flatten())
    emb_model.add(layers.Dense(dense, activation='softmax'))
    emb_model.summary()
    emb_history = deep_model(emb_model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)
    result = emb_history
    return result
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

#for Word2Vec input word2vec .bin file and word_index = tokenizer.word_index from tokenizer
def load_vectors_word2vec(fname,word_index):
    word_vectors = KeyedVectors.load(fname)
    
    vocabulary_size = min(MAX_NB_WORDS, len(word_index))+1
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
    embedding_layer = Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False,input_length = MAX_LEN)
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

    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False,input_length = MAX_LEN) 

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
#CNN
def cnn2(input_shape,embedding_layer1,dense):
    model = Sequential()
    model.add(embedding_layer1)
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size, return_sequences = True))
    model.add(GRU(MAX_LEN))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(dense,activation='softmax',kernel_regularizer=regularizers.l2(0.00000000001),activity_regularizer=regularizers.l1(0.00000000001)))
    sgd = SGD(lr=0.2, decay=1e-6, momentum=1.0, nesterov=True)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model
def cnn1(input_shape,embedding_layer1,dense):
    
    model = Sequential()
    model.add(embedding_layer1)
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(dense,activation='softmax',kernel_regularizer=regularizers.l2(0.00000000001),activity_regularizer=regularizers.l1(0.00000000001)))
    sgd = SGD(lr=0.2, decay=1e-6, momentum=1.0, nesterov=True)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model
def gru_model(input_shape,embedding_layer1,dense):
    model = Sequential()
    model.add(embedding_layer1)
    model.add(GRU(300))
    #model.add(Dropout(0.5))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dense(dense, activation='sigmoid',kernel_regularizer=regularizers.l1(0.00000000001),activity_regularizer=regularizers.l2(0.00000000001)))
    sgd = SGD(lr=0.2, decay=1e-6, momentum=1.0, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    
    return model
def lstm_model(input_shape,embedding_layer1,dense):
    model = Sequential()
    model.add(embedding_layer1)
    #model.add(SpatialDropout1D(0.25))
    model.add(LSTM(MAX_LEN))
    model.add(Dropout(0.0001))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dropout(0.0001))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dropout(0.0001))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dropout(0.001))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='relu'))

    #,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.00000000001)
    model.add(Dense(dense,activation='softmax',kernel_regularizer=regularizers.l2(0.00000000001),activity_regularizer=regularizers.l1(0.00000000001)))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=1.0, nesterov=True)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    return model
    #from https://www.kaggle.com/sakarilukkarinen/embedding-lstm-gru-and-conv1d/versions
def gru_model2(input_shape,embedding_layer1,dense):
    model = Sequential()
    model.add(embedding_layer1)
    model.add(GRU(MAX_LEN, return_sequences = True))
    model.add(GRU(32, activation = 'relu'))
    model.add(Dropout(0.0001))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dropout(0.001))
    model.add(Dense(MAX_LEN, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(dense, activation = 'softmax',kernel_regularizer=regularizers.l2(0.00000000001),activity_regularizer=regularizers.l1(0.00000000001)))
    sgd = SGD(lr=0.2, decay=1e-6, momentum=1.0, nesterov=True)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model

def predict_model(input_shape,embedding_layer,model_type,X_train,y_train,X_val,y_val,dense):
    callbacks = [EarlyStopping(monitor='val_loss')]
    #you can also use rmsprop as optimizer
    adam = Adam(lr=1e-3)
    loss = 'categorical_crossentropy'
    if model_type == 'cnn1':
        model = cnn1((input_shape,),embedding_layer,dense)
        #model.compile(loss=loss,optimizer=adam,metrics=['acc'])
        track = model.fit(X_train, y_train, batch_size=32, epochs=40, verbose=1, validation_data=(X_val, y_val),callbacks=callbacks)
        #plot_function(track)
        plot_performance(track)
    elif model_type == 'cnn2':
        model1 = cnn2((input_shape,),embedding_layer,dense)
        model1.compile(loss=loss,optimizer=adam,metrics=['acc'])
        track2 = model1.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_val, y_val),callbacks=callbacks)
        #plot_function(track2)
        plot_performance(track2)
        model = model1
    elif model_type == 'lstm':
        LSTM_model = lstm_model((input_shape,),embedding_layer,dense)
        #The model has already been compiled in the function call
        track3 = LSTM_model.fit(X_train, y_train, epochs=40, batch_size=32,verbose=1,shuffle=True,validation_data=(X_val, y_val))
        #plot_function(track3)
        plot_performance(track3)
        model = LSTM_model
    elif model_type == 'gru':
        GRU_model = gru_model((input_shape,),embedding_layer,dense)
        #The model has already been compiled in the function call
        track2 = GRU_model.fit(X_train, y_train, epochs=25, batch_size=128,verbose=1,shuffle=True, validation_data=(X_val, y_val),callbacks=callbacks)
        #plot_function(track2)
        plot_performance(track2)
        model = GRU_model
    else:
        gru2 = gru_model2((input_shape,),embedding_layer,dense)
        track2 = gru2.fit(X_train, y_train, epochs=40, batch_size=32,verbose=1,shuffle=True, validation_data=(X_val, y_val))
        #plot_function(track2)
        plot_performance(track2)
        model = gru2
    #model = None
    return model
def plot_model_performace(result):
    sns.set_style("ticks")
    figsize=(22, 6)

    ticksize = 12
    titlesize = ticksize + 8
    labelsize = ticksize + 5

    xlabel = "Model"
    ylabel = "Score"

    title = "Model Performance"

    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)

    col1 = "model"
    col2 = "score"
    sns.barplot(x=col1, y=col2, data=result)
    plt.title(title.title())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.grid()
    plt.plot()
    plt.show()
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