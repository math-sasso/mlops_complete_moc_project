import os
import json
import yaml
import pickle
import mlflow
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from src.utils import get_regression_metrics

#TODO: Aplicar a DVC API para utilizar a versão dos dados que nós queremos na entrada
#https://www.youtube.com/watch?v=EE7Gk84OZY8

#dirpaths
ROOT_DIR  = os.path.join(os.path.abspath(os.getcwd()))
data_dir = os.path.join(ROOT_DIR,"data")
data_raw_dir = os.path.join(data_dir,"raw")
data_prepared_dir = os.path.join(data_dir,"prepared")
data_train_dir = os.path.join(data_dir,"train")

#filepaths
params_file = os.path.join(ROOT_DIR,"params.yaml")

#getting params
params = yaml.safe_load(open(params_file))["train"]
max_depth = params["max_depth"]
seed = params["seed"]

# Getting train data
X_train = pd.read_csv(os.path.join(data_prepared_dir,"X_train.csv"))
y_train = pd.read_csv(os.path.join(data_prepared_dir,"y_train.csv"))
X_valid = pd.read_csv(os.path.join(data_prepared_dir,"X_valid.csv"))
y_valid = pd.read_csv(os.path.join(data_prepared_dir,"y_valid.csv"))


# MLFLOW monitoring
mlflow.set_tracking_uri('http://127.0.0.1:5000')# Setando o caminho do MLFLOW
mlflow.set_experiment('house-prices-script') # Nome dop experimento
with mlflow.start_run(): # Start do log
    
    regr = RandomForestRegressor(max_depth=max_depth, random_state=seed)
    regr.fit(X_train, y_train)
    mlflow.log_params(params)
    mlflow.sklearn.log_model(regr,'RandomForestRegressor')
    y_valid_predicted = regr.predict(X_valid)
    metrics = get_regression_metrics(y_valid, y_valid_predicted)
    mlflow.log_metric('mse', metrics["mse"])
    mlflow.log_metric('rmse', metrics["rmse"])
    mlflow.log_metric('r2', metrics["r2"])
mlflow.end_run()

# Saving Results
with open(os.path.join(data_train_dir,"model.pkl"), "wb") as fd:
    pickle.dump(regr, fd)

# Now print to file
with open(os.path.join(data_train_dir,"metrics.json"), 'w') as fp:
    json.dump(metrics,fp)