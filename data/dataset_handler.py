import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DatasetHandler:
    def __init__(self):
        self.scaler = None

    def ler_dataset(self, path):
        return pd.read_csv(path)

    def remover_colunas(self,df, lista):
        return df.drop(lista, axis=1)
    
    def extrair_features_target(self,df, target):
        y = df[target]
        X = df.drop(target, axis=1)

        return X,y
    
    def discretizar_coluna(self,X, coluna, q=3, labels=[]):
        if len(labels) != q:
            labels = []
            for i in range(q):
                labels.append("l"+str(i))
        else:
            q = len(labels)

        X[coluna]=pd.qcut(X[coluna], 
                q=q, 
                labels=labels).astype(str)
        
        return X
    
    # https://www.digitalocean.com/community/tutorials/normalize-data-in-python
    def normalizar_dados(self, range, df):
        self.scaler = MinMaxScaler(feature_range=range)
        normalized = self.scaler.fit_transform(df)

        return pd.DataFrame(normalized)
    
    # https://stackoverflow.com/questions/43382716/how-can-i-cleanly-normalize-data-and-then-unnormalize-it-later
    def desnormalizar_dados(self, df):
        unnormalized = self.scaler.inverse_transform(df)
        
        return pd.DataFrame(unnormalized)

# df = pd.read_csv("data/treino_sinais_vitais_com_label.csv")

# df = df.drop(["s1","s2", "gravidade", "vitima"], axis=1)

# y = df["classe"]
# X = df.drop("classe", axis=1)