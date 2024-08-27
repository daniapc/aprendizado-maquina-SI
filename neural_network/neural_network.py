import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import seaborn as sn
import matplotlib.pyplot as plt

import random
import itertools
from datetime import datetime

class NeuralNetwork:

    def __init__(self):
        self.solver='sgd' #{'lbfgs', 'sgd', 'adam'}
        self.activation = 'identity' #{'identity', 'logistic', 'tanh', 'relu'}
        self.batch_size = 'auto' #int
        self.hidden_layer_sizes = (10, 20, 10,) 
        self.max_iter = 200 #num epocas
        self.learning_rate_init = 0.001 

        self.model = MLPClassifier()

    def set_params(self, params):
        self.model.set_params(**params)

    def fit(self, X,y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)

#This function performs an hyperparameter tunning using cross-validation
def crossvalidation(X, y, model, n_folds, params, seed=4, save=True, conf_matrix=False):
    start=datetime.now()
    #Creating combinations of all hyperparameters (like and expand grid)
    keys, values = zip(*params.items())
    params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    #Initialice a list to store the accuracy for each set of parameters
    accuracy = []
    #Iterete each set of parameters
    for j in range(0, len(params)):
        #Creating the folds indexes
        indexes=np.array(X.index)
        #Shuffle randomly the indexes
        random.Random(seed).shuffle(indexes)
        #Save the indexes intro the folds
        kf = KFold(n_splits=n_folds)

        folds = np.array_split(indexes,n_folds)

        # print(train)
        # print(test)

        acc = [] #Initialice a local accuracy for each fold
        total_y = [0]
        total_pred = [0]
        #Compute the accuracy of each fold using the first one as a test set
        for train,test in kf.split(X,y):
            #Change the parameters of the model
            model.set_params(params[j])
            #Train the model with the fold
            model.fit(X.loc[train], y.loc[train])
            total_y += list(y.loc[test])
            #Predict using the first fold as test set
            # prediction = X.loc[folds[1]].apply(lambda row : model.predict(row), axis = 1)
            prediction = model.predict(X.loc[test])
            total_pred += list(prediction)
            #Store the accuracy of the fold
            acc.append(sum((prediction  ==  y.loc[test]))/len(y.loc[test]))
        #Compute the accuracy of all the folds for the set of parameters
        # print(total_pred)
        # print(total_y)

        accuracy.append(sum(acc)/len(acc))
        #Print the results for the set of parameters
        print(f"Parameters: {params[j]}, Accuracy: {accuracy[j]}")

        if save:
            param_dict = params[j].copy()
            param_dict["Dict"] = str(params[j])
            param_dict["Accuracy"] = accuracy[j]
            save_df = pd.DataFrame([param_dict])
            save_df.to_csv("./neural_network/predictions.csv", mode='a', index=False, header=False)
            #Change the seed to do other folds

        if conf_matrix:

            total_y.pop(0)
            total_pred.pop(0)

            array = confusion_matrix(total_y, total_pred)
            df_cm = pd.DataFrame(array)

            ax = plt.subplot()
            sn.set_theme(font_scale=1.0) # Adjust to fit
            sn.heatmap(df_cm, annot=True, ax=ax, cmap="Blues", fmt="g")
            label_font = {'size':'8'}  # Adjust to fit
            ax.set_xlabel('Predicted labels', fontdict=label_font);
            ax.set_ylabel('Observed labels', fontdict=label_font);

            title_font = {'size':'8'}  # Adjust to fit
            ax.set_title('Confusion Matrix', fontdict=title_font);

            ax.tick_params(axis='both', which='major', labelsize=8)

            plt.show()

        seed=seed+1
    #Select and print the set of best hyperparameters
    max_value = max(accuracy)
    max_index = accuracy.index(max_value)
    print('Best hyperparameters:')
    print(str(params[max_index]) + ", Accuracy: " + str(max_value))
    
    end = datetime.now()
    tempo = round((end-start).seconds + (end-start).microseconds/1000000,3)
    print(f"Tempo de execução: {tempo} s")
