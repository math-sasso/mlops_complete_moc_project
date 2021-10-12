import os
import mlflow
import pandas as pd


#dirpaths
ROOT_DIR  = os.path.join(os.path.abspath(os.getcwd()))
data_dir = os.path.join(ROOT_DIR,"data")
data_raw_dir = os.path.join(data_dir,"raw")
data_prepared_dir = os.path.join(data_dir,"prepared")

# Load model as a PyFuncModel.
logged_model = 'runs:/69d31055bbd045e88c20fa36fde52253/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
data = pd.read_csv(os.path.join(data_prepared_dir,"X_valid.csv"))
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv(os.path.join(data_prepared_dir,"mlflow_predicted.csv"))