import os
import json
import pandas as pd
import pickle
from src.utils import get_regression_metrics,plot_comparing

def read_pickle(filepath:str):
    with open(filepath, 'rb') as f:
        content = pickle.load(f)
    return content

def dump_json(filepath:str,d,ensure_ascii=False,command='a'):
        with open(filepath, command) as fp:
            json.dump(d, fp, ensure_ascii=ensure_ascii)

#dirpaths
ROOT_DIR  = os.path.join(os.path.abspath(os.getcwd()))
data_dir = os.path.join(ROOT_DIR,"data")
data_prepared_dir = os.path.join(data_dir,"prepared")
data_train_dir = os.path.join(data_dir,"train")
data_evaluation_dir = os.path.join(data_dir,"evaluation")

#Getting data
X_test = pd.read_csv(os.path.join(data_prepared_dir,"X_test.csv"))
y_test = pd.read_csv(os.path.join(data_prepared_dir,"y_test.csv"))

#Getting 
regr = read_pickle(os.path.join(data_train_dir,"model.pkl"))
y_test_predicted = regr.predict(X_test)
metrics = get_regression_metrics(y_test, y_test_predicted)
dump_json(filepath=os.path.join(data_evaluation_dir,"test_metrics.json"),d=metrics)
plot_comparing(y_ref=y_test,y_pred=y_test_predicted,outpath=os.path.join(data_evaluation_dir,"histograms.png"))

