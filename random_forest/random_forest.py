import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import randint
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display
import graphviz

class RandomForest:

    def __init__(self,
            max_features = None, # how many features are available at each split
            cost_function = "entropy", # method used to identify the best split in each tree — gini impurity, feature importance, etc.
            n_estimators = 100, # how many trees are in the forest
            max_depth = None, # how far from the root node can each tree grow
            min_samples = 10, # number of records that must be in each node to proceed with another split
            bootstrapped = True,
            max_samples_boot = None,
            verbose = 0,
            split_size = 5
        ):
        self.split_size = split_size

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
        self.simple_cross_validation()

    def search_cross_validation(self):
        param_dist = {'n_estimators': randint(50,500),
              #'max_depth': randint(1,20),
              'max_features': randint(1,5),
              'min_samples_split': randint(2, 100)}

        # Create a random forest classifier
        rf = RandomForestClassifier()

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(rf, 
                                        param_distributions = param_dist, 
                                        n_iter=5, 
                                        cv=self.split_size)
        
        _, X, y = self.import_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)

        y_pred = rand_search.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Create a variable for the best model
        best_rf = rand_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  rand_search.best_params_)

    def simple_cross_validation(self):
        k_fold_split, X, y = self.import_data()

        k_results = []
        for i, indexes in enumerate(k_fold_split):
            print(f"Fold {i}")
            train_index, test_index = indexes
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.train(X_train, y_train)
            k_results.append(self.results(X_test, y_test, X_train))
        print(f"Final accuracy: {np.mean(k_results)}")

    def import_data(self):
        data = np.loadtxt('/home/gustavo/aprendizado-maquina-SI/treino_sinais_vitais_com_label.txt', delimiter=',')          
        # Split data into features and target
        X = data[:, :-2]  # All columns except the last
        y = data[:, -1]   # The last column
        kf = KFold(n_splits=self.split_size)
        return kf.split(X), X, y
    
    def train(self, X, y):
        self.rf.fit(X, y)

    def results(self, X_test, y_test, X_train):
        y_pred = self.rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        return accuracy

    def show(self):
        # Export the first three decision trees from the forest
        for i in range(3):
            tree = self.rf.estimators_[i]
            dot_data = export_graphviz(tree,
                                    feature_names=["pSist", "pDiast", "qPA", "pulso", "resp", "grav"],  
                                    filled=True,  
                                    max_depth=2, 
                                    impurity=False, 
                                    proportion=True)
            graph = graphviz.Source(dot_data)
            display(graph)

            # Render the graph and save it as a .png image (you can also use other formats like pdf, svg, etc.)
            graph.render(filename=f'tree_{i}', format='png')

RandomForest()