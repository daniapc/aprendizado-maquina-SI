import pandas as pd

class DatasetHandler:
    def __init__(self):
        pass

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

# df = pd.read_csv("data/treino_sinais_vitais_com_label.csv")

# df = df.drop(["s1","s2", "gravidade", "vitima"], axis=1)

# y = df["classe"]
# X = df.drop("classe", axis=1)