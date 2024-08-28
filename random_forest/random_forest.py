import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display
import graphviz

class RandomForest:

    def __init__(self,
            max_features = 3, # how many features are available at each split
            cost_function = "entropy", # method used to identify the best split in each tree — gini impurity, feature importance, etc.
            n_estimators = 50, # how many trees are in the forest
            max_depth = None, # how far from the root node can each tree grow
            min_samples = 2, # number of records that must be in each node to proceed with another split
            bootstrapped = False,
            max_samples_boot = None,
            verbose = 0,
            split_size = 5
        ):
        self.split_size = split_size

        ff1 = []
        facc = []
        # for i in range(5):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=cost_function,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples,
            bootstrap=bootstrapped,
            max_samples=max_samples_boot,
            verbose=verbose
        )

        # self.search_cross_validation()
        acc, f1 = self.simple_cross_validation()
        ff1.append(f1)
        facc.append(acc)
        #self.show()
        print(f"f1: {np.mean(ff1)}")
        print(f"facc: {np.mean(facc)}")

    def simple_cross_validation(self):
        k_fold_split, X, y = self.import_data()
        # self.show_dataset_balance(y)

        k_results = []
        tot_y_test = []
        tot_y_pred = []
        for i, indexes in enumerate(k_fold_split):
            print(f"Fold {i}")
            train_index, test_index = indexes
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.train(X_train, y_train)
            acc, f1, y_pred = self.results(X_test, y_test, X_train)
            print(tot_y_test)
            tot_y_test.extend(y_test)
            tot_y_pred.extend(y_pred)
            k_results.append([acc, f1])

        facc = np.mean(k_results[1:])
        print(f"Final accuracy: {facc}")
        ff1 = np.mean(k_results[:-1])
        print(f"Final F1: {ff1}")

        cm = confusion_matrix(tot_y_test, tot_y_pred)
        # Create a confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        return facc, ff1

    def import_data(self):
        data = np.loadtxt('/home/gustavo/aprendizado-maquina-SI/treino_sinais_vitais_com_label.txt', delimiter=',')          
        # Split data into features and target
        X = data[:, 2:-2]  # All columns except the last
        y = data[:, -1]   # The last column

        # Assuming X is your features matrix
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        kf = KFold(n_splits=self.split_size)
        return kf.split(X), X, y

    def show_dataset_balance(self, y):
        # Count the frequency of each class
        unique, counts = np.unique(y, return_counts=True)

        # Plot a bar chart
        plt.bar(unique, counts, color='skyblue')
        plt.xlabel('Classe')
        plt.ylabel('Frequência')
        plt.title('Distribuição das Classes')
        plt.xticks(unique)  # Ensures class labels are correctly aligned with bars

        # Annotate each bar with the exact count
        for i, count in enumerate(counts):
            plt.text(unique[i], count + 0.1, str(count), ha='center', va='bottom')

        plt.show()
    
    def train(self, X, y):
        self.rf.fit(X, y)

    def results(self, X_test, y_test, X_train):
        y_pred = self.rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print("F1:", f1)

        return accuracy, f1, y_pred

    def show(self):
        # Export the first three decision trees from the forest
        for i in range(3):
            tree = self.rf.estimators_[i]
            dot_data = export_graphviz(tree,
                                    feature_names=["qPA", "pulso", "resp", "grav"],  
                                    filled=True,  
                                    max_depth=None, 
                                    impurity=False, 
                                    proportion=True)
            graph = graphviz.Source(dot_data)
            display(graph)

            # Render the graph and save it as a .png image (you can also use other formats like pdf, svg, etc.)
            graph.render(filename=f'tree_{i}', format='png')

    def search_cross_validation(self):
        param_dist = {'n_estimators': [1, 10, 100, 500, 1000],
              'max_depth': [3],
              'max_features': [1, 2, 3, "sqrt"],
              'min_samples_split': [2, 25, 50, 100],
              'criterion': ['entropy', 'gini'],
              'bootstrap': [True, False]}

        # Create a random forest classifier
        rf = RandomForestClassifier()

        # Use random search to find the best hyperparameters
        rand_search = GridSearchCV(rf, 
                                        param_grid = param_dist, 
                                        n_jobs=-1, 
                                        cv=self.split_size,
                                        verbose=2)
        
        _, X, y = self.import_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/self.split_size))

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)

        y_pred = rand_search.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Create a variable for the best model
        best_rf = rand_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  rand_search.best_params_)

        # Save results to a DataFrame and CSV
        results_df = pd.DataFrame(rand_search.cv_results_)
        results_df.to_csv('random_search_results.csv', index=False)
RandomForest()