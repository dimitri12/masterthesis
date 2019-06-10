# masterthesis

This code is part of a project called EMDAI which centralizes on implementing AI into the medical dispatcher systems.
This code takes inspiration of Sentiment Analysis but explores possible methods from text classification to word embeddings
and analyze how these methods can predict future outcomes based on the given data.


To run predictions based only on classification methods, open in the Jupyter Notebook the "exjobb_final2.ipynb" file and run all cells (except for the "tree" which is decision trees and it is not included in the report, you can remove it)

To run predictions based only on deep learning methods, open in the Jupyter Notebook the "exjobb_final3.ipynb" file and run all cells.

Inside the models function: (CNN1,CNN2,LSTM,GRU,BILSTM and GRU2) try and comment out the embedding_layer1 definition since it overwrites the embedding_layer1 that was created using fasttext
