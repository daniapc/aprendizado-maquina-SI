# https://rpubs.com/FelipeMonroy/685798

import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
import random
import itertools
from datetime import datetime

from sklearn.metrics import confusion_matrix

import seaborn as sn
import matplotlib.pyplot as plt

Number = (int, float, complex)

class TreeNode:
#Inicialization of the class, with some default parameters
    def __init__(self, min_samples_split=2, max_depth=None, seed=2, 
                 verbose=False):
        # Sub nodes -- recursive, those elements of the same type (TreeNode)
        self.children = {} 
        self.decision = None # Undecided
        self.split_feat_name = None # Splitting feature
        self.threshold = None #Where to split the feature
        #Minimun number of samples to do a split
        self.min_samples_split=min_samples_split
        #Maximun number of nodes/end-nodes: 0 => root_node
        self.max_depth=max_depth
        self.seed=seed #Seed for random numbers
        self.verbose=verbose #True to print the splits
        self.stop=False

    def set_params(self, params):
        # {'min_samples_split': [2,3,4,5], 
        #  'max_depth': [2,3,4,None], 
        #       'seed': [2]}
        self.min_samples_split = params['min_samples_split']
        self.max_depth = params['max_depth']
        self.seed = params['seed']

    def fit(self, X, y):
        if (self.validate_data(X, y)):
            self.recursiveGenerateTree(X, y, 0)

#This function validate the datatype and equal len of data and target
    def validate_data(self, data, target):
        #Validating data type for sample (X)
        if not isinstance(data,(list,pd.core.series.Series,np.ndarray, 
                                pd.DataFrame)):
            return False
        #Validating data type for target (y)
        if not isinstance(target,(list,pd.core.series.Series,np.ndarray)):
            return False
        #Validating same len of X and y
        if len(data) != len(target):
            return False
        return True
    
#This is a recursive function, which selects the decision if possible. 
#If not, create children nodes
    def recursiveGenerateTree(self, sample_data, sample_target, current_depth):
        #If there is only one possible outcome, select that as the decision
        if len(sample_target.unique())==1:
            self.decision = sample_target.unique()[0]
#If the sample has less than min_samples_split select the majority class
        elif len(sample_target)<self.min_samples_split:
            self.decision = self.getMajClass(sample_target)
#If the deep of the current branch is equal to max_depth \
#select the majority class
        elif current_depth == self.max_depth:
            self.decision = self.getMajClass(sample_target)
        else:
#Call the function to select best_attribute to split
            best_attribute,best_threshold,splitter = self.splitAttribute(sample_data,
                                                                         sample_target)
            self.children = {} #Initializing a dictionary with the children nodes
            self.split_feat_name = best_attribute  #Name of the feature
            self.threshold = best_threshold #Threshold for continuous variable
            current_depth += 1 #Increase the deep by 1
            if self.stop:
                self.max_depth = current_depth
#Create a new node for each class of the best feature
            for v in splitter.unique():
                index = splitter == v #Select the indexes of each class
                #If there is data in the node, create a new tree node with that partition
                if len(sample_data[index])>0:
                    self.children[v] = TreeNode(min_samples_split = self.min_samples_split,
                                                max_depth=self.max_depth,
                                                seed=self.seed,
                                                verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(sample_data[index],
                                                           sample_target[index],
                                                           current_depth)
#If there is no data in the node, use the previous node data (this one) \
#and make a decision based on the majority class
                else:
                    #Force to make a decision based on majority class \
                    #simulating that it reached max_depth
                    self.children[v] = TreeNode(min_samples_split = self.min_samples_split,
                                                max_depth=1,
                                                seed=self.seed,
                                                verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(sample_data,                                          
                                                           sample_target,
                                                           current_depth=1)
                    
#This function define which is the best attribute to split (string \
#or continious)
    def splitAttribute(self, sample_data,sample_target):
        info_gain_max = -1*float("inf") #Info gain set to a minimun
        #Creating a blank serie to store variable in which the split is based
        splitter = pd.Series(dtype='str') 
        best_attribute = None #Setting attribute to split to None
        best_threshold = None #Setting the threshold to None
        #Iterate for every attribute in the sample_data
        for attribute in sample_data.keys():
#If the attribute is a string
            if is_string_dtype(sample_data[attribute]):
                #Compute information gain using that attribute to split the target
                aig = self.compute_info_gain(sample_data[attribute], sample_target)
                #If the information gain is more than the previous one, store
                if aig > info_gain_max:
                    splitter = sample_data[attribute] #Store the variable
                    info_gain_max = aig #Store the information gain
                    best_attribute = attribute #Store the name of the attribute
                    #In this case there is no threshold (string)
                    best_threshold = None 
#If the attribute is a continuous
            else:
                #Sort the continuous variable in an asc order. Change the target order \
                # based on that
                sorted_index = sample_data[attribute].sort_values(ascending=True).index
                sorted_sample_data = sample_data[attribute][sorted_index]
                sorted_sample_target = sample_target[sorted_index]
                #Iterate between each sample, except the last one
                for j in range(0, len(sorted_sample_data) - 1):
                    #Create a blank serie to store the classification (less or greater)
                    classification = pd.Series(dtype='str')
                    #If two consecutives samples are not the same, use its mean as \
                    #a threshold
                    if sorted_sample_data.iloc[j] != sorted_sample_data.iloc[j+1]:
                        threshold = (sorted_sample_data.iloc[j] + 
                                     sorted_sample_data.iloc[j+1]) / 2
                        #Assign greater or less acording to the threshold
                        # print(sample_data)

                        classification = sample_data[attribute] > threshold
                        classification.replace({False: "less", True: "greater"}, inplace=True)
                        # classification[classification] = 'greater'
                        # classification[classification == False] = 'less'
                        # print(classification)
                        
                        # #Calculate the information gain using previous variable \
                        # (now categorical)
                        aig = self.compute_info_gain(classification, sample_target)
                        #If the information gain is more than the previous one, store
                        if aig >= info_gain_max:
                            splitter = classification #Store the variable
                            info_gain_max = aig #Store the information gain
                            best_attribute = attribute #Store the name of the attribute
                            best_threshold = threshold #Store the threshold 
#If verbose is true print the result of the split
        if info_gain_max == 0.0:
            self.stop = True
        if self.verbose:
            if is_string_dtype(sample_data[best_attribute]):
                print(f"Split by {best_attribute}, IG: {info_gain_max:.2f}")
            else:
                print(f"Split by {best_attribute}, at {threshold}, IG: {info_gain_max:.2f}")
        return (best_attribute,best_threshold,splitter)

#This function calculates the entropy based on the distribution of \
#the target split
    def compute_entropy(self, sample_target_split):
        #If there is only only one class, the entropy is 0
        if len(sample_target_split.unique()) < 2:
            return 0
        #If not calculate the entropy
        else:
            # print(sample_target_split)
            freq = np.array(sample_target_split.value_counts(normalize=True))
            return -(freq * np.log2(freq + 1e-6)).sum()

#This function computes the information gain using a specific \
#attribute to split the target
    def compute_info_gain(self, sample_attribute, sample_target):
        #Compute the proportion of each class in the attribute
        values = sample_attribute.value_counts(normalize=True)
        split_ent = 0 #Set the entropy to 0
        #Iterate for each class of the sample attribute
        for v, fr in values.items():
            #Calculate the entropy for sample target corresponding to the class
            index = sample_attribute==v
            sub_ent = self.compute_entropy(sample_target[index])
            #Weighted sum of the entropies
            split_ent += fr * sub_ent
        #Compute the entropy without any split
        ent = self.compute_entropy(sample_target)
        #Return the information gain of the split
        return ent - split_ent
    
#This function selects the majority class of the target to make a decision
    def getMajClass(self, sample_target):
        #Compute the number of records per class and order it (desc)
        freq = sample_target.value_counts().sort_values(ascending=False)
        #Select the name of the class (classes) that has the max number of records
        MajClass = freq.keys()[freq==freq.max()]
        #If there are two classes with equal number of records, select one randomly
        if len(MajClass) > 1:
            decision = MajClass[random.Random(self.seed).randint(0,len(MajClass)-1)]
        #If there is only onle select that
        else:
            decision = MajClass[0]
        return decision
    
#This function returns the class or prediction given an X
    def predict(self, sample):
        #If there is a decision in the node, return it
        if self.decision is not None:
            #Print when verbose is true
            if self.verbose:
                print("Decision:", self.decision)
            return self.decision #Return decision
        #If not, it means that it is an internal node
        else:
            #Select the value of the split attribute in the given data
            attr_val = sample[self.split_feat_name]
            #Print if verbose is true
            if self.verbose:
                print('attr_val')
            #If the value for the feature is not numeric just go to the\
            # corresponding child node and print
            if not isinstance(attr_val, Number):
                if attr_val in self.children:
                    child = self.children[attr_val]
                else:
                    return self.decision

                if self.verbose:
                    print("Testing ", self.split_feat_name, "->", attr_val)
            #If the value is numeric see if it is greater or less than the \
            #threshold
            else:
                if attr_val > self.threshold:
                    child = self.children['greater']
                    if self.verbose:
                        print("Testing ", self.split_feat_name, "->",
                              'greater than ', self.threshold)
                else:
                    child = self.children['less']
                    if self.verbose:
                        print("Testing ", self.split_feat_name, "->",
                              'less than or equal', self.threshold)
            #Do it again with the child until finding the terminal node
            return child.predict(sample)
        
#This function performs an hyperparameter tunning using cross-validation
def crossvalidation(X, y, model, n_folds, params, seed=4, q = None, save=True, conf_matrix=False):
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
        folds = np.array_split(indexes,n_folds)
        acc = [] #Initialice a local accuracy for each fold
        total_y = [0]
        total_pred = [0]
        #Compute the accuracy of each fold using the first one as a test set
        for k in range(1,n_folds):
            #Change the parameters of the model
            model.set_params(params[j])
            #Train the model with the fold
            model.fit(X.loc[folds[k]], y.loc[folds[k]])
            total_y += list(y.loc[folds[k]])
            #Predict using the first fold as test set
            prediction = X.loc[folds[0]].apply(lambda row : model.predict(row), axis = 1)
            total_pred += list(prediction)
            #Store the accuracy of the fold
            acc.append(sum((prediction  ==  y.loc[folds[0]]))/len(y.loc[folds[0]]))
        #Compute the accuracy of all the folds for the set of parameters
        accuracy.append(sum(acc)/len(acc))
        #Print the results for the set of parameters
        print(f"Parameters: {params[j]}, Accuracy: {accuracy[j]}")

        if save:
            param_dict = params[j].copy()
            param_dict["q"] = str(q)
            param_dict["Dict"] = str(param_dict)
            param_dict["Accuracy"] = accuracy[j]
            save_df = pd.DataFrame([param_dict])
            save_df.to_csv("./decision_tree/predictions.csv", mode='a', index=False, header=False)

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

        #Change the seed to do other folds
        seed=seed+1
    #Select and print the set of best hyperparameters
    max_value = max(accuracy)
    max_index = accuracy.index(max_value)
    print('Best hyperparameters:')
    print(str(params[max_index]) + ", Accuracy: " + str(max_value))
    
    end = datetime.now()
    tempo = round((end-start).seconds + (end-start).microseconds/1000000,3)
    # print(f"Tempo de execução: {tempo} s")