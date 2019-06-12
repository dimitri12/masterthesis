# masterthesis

This code is part of a project called EMDAI which centralizes on implementing AI into the medical dispatcher systems.
This code takes inspiration of Sentiment Analysis but explores possible methods from text classification to word embeddings
and analyze how these methods can predict future outcomes based on the given data.


To run predictions based only on classification methods, open in the Jupyter Notebook the "exjobb_final2.ipynb" file and run all cells (except for the "tree" which is decision trees and it is not included in the report, you can remove it)

To run predictions based only on deep learning methods, open in the Jupyter Notebook the "exjobb_final3.ipynb" file and run all cells.

Inside the models function: (CNN1,CNN2,LSTM,GRU,BILSTM and GRU2) try and comment out the embedding_layer1 definition since it overwrites the embedding_layer1 that was created using fasttext

I have included a new API called Talos, which is a variant of the gridsearch algorithm that tests a lot of defined parameters inside the function "Word_embedding" and then predicting the best model. The algorithm exists in a new file called "exjobb_final3v2" to keep it different from the original final3 file. 

To run final3v2 make sure that you install Talos on your computer using pip and just run all the cells; the process can take a while and after the run, upload the final3v2 file to github so that i can write the results.
