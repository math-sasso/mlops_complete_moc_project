import os
import pandas as pd

df = pd.read_csv(os.path.join("data","raw","casas.csv"))
X = df.drop("preco",axis=1)
X.to_csv(os.path.join("data","processed","casas_X.csv"),index=False)
#y = df["preco"].copy()