import os
import xgboost
import mlflow
logged_model = 'runs:/69d31055bbd045e88c20fa36fde52253/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv(os.path.join("data","processed","casas_X.csv"))
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv(os.path.join("data","processed","precos.csv"))



