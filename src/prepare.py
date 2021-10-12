import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

#dirpaths
ROOT_DIR  = os.path.join(os.path.abspath(os.getcwd()))
data_dir = os.path.join(ROOT_DIR,"data")
data_raw_dir = os.path.join(ROOT_DIR,"raw_data")
#data_raw_dir = os.path.join(data_dir,"raw")
data_prepared_dir = os.path.join(data_dir,"prepared")

#filepaths
params_file = os.path.join(ROOT_DIR,"params.yaml")

#getting params
params = yaml.safe_load(open(params_file))["prepare"]
eval_size_from_data = params["eval_size_from_data"]
test_size_from_eval = params["test_size_from_eval"]
random_state = params["seed"]

# preparing dataset
df = pd.read_csv(os.path.join(data_dir,"raw","casas.csv"))
X = df.drop('preco', axis=1)
y = df['preco'].copy()
X_train, X_, y_train, y_= train_test_split(X,
                                            y,
                                            test_size=eval_size_from_data,
                                            random_state=random_state)

X_valid, X_test, y_valid, y_test = train_test_split(X_,
                                            y_,
                                            test_size=test_size_from_eval,
                                            random_state=random_state)


# Saving results to repp
X_train.to_csv(os.path.join(data_prepared_dir,"X_train.csv"))
y_train.to_csv(os.path.join(data_prepared_dir,"y_train.csv"))
X_valid.to_csv(os.path.join(data_prepared_dir,"X_valid.csv"))
y_valid.to_csv(os.path.join(data_prepared_dir,"y_valid.csv"))
X_test.to_csv(os.path.join(data_prepared_dir,"X_test.csv"))
y_test.to_csv(os.path.join(data_prepared_dir,"y_test.csv"))
