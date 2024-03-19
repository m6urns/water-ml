import os
import logging
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(filename='grid_search.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Load Datasets
data_train_raw = pd.read_csv('~/water-ml/datasets/sheet_1.csv')
data_test_raw = pd.read_csv('~/water-ml/datasets/sheet_2_3.csv')
data_train_unlabeled = pd.read_csv('~/water-ml/datasets/sheet_3.csv')

# Remove DWDS_sim_rows from data
dwds = data_test_raw[data_test_raw['Location'] == 'DWDS Simulator (EPA, 2016)']

# Drop DWDS sim data from sheets 2&3 (test_data)
data_test = data_test_raw[data_test_raw['Location'] != 'DWDS Simulator (EPA, 2016)']

# Concatenate train data and dwds data
data_train = pd.concat([data_train_raw, dwds])

# Prepare train data
target_columns = ['Scheme', 'Sample (reference)']
X_train = data_train_raw.drop(target_columns, axis=1)
y_train = data_train_raw['Scheme'].map({'Stable': 1, 'Failure': 0})
X_train.replace('ND', 0, inplace=True)

# Prepare test data
target_columns = ['Scheme', 'Sample', 'Location']
X_test = data_test.drop(target_columns, axis=1)
y_test = data_test['Scheme'].map({'Stable': 1, 'Failure': 0})
X_test.replace('ND', 0, inplace=True)
X_test.fillna(0, inplace=True)

# Define the parameter grid for ExtraTreesClassifier
param_grid_etc = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': ['sqrt', 'log2', None],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [-1],
    'random_state': [42],
    'verbose': [0],
    'warm_start': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'ccp_alpha': [0.0, 0.1, 0.2],
    'max_samples': [None, 0.5, 0.7, 0.9]
}

# Create an instance of ExtraTreesClassifier
etc = ExtraTreesClassifier()

# Perform grid search
grid_search_etc = GridSearchCV(
    estimator=etc,
    param_grid=param_grid_etc,
    scoring='accuracy',
    cv=10,
    n_jobs=-1,
    verbose=1
)

# Fit the grid search object
logging.info('Starting grid search')
grid_search_etc.fit(X_train, y_train)

# Get the best model and its parameters
best_etc = grid_search_etc.best_estimator_
best_params_etc = grid_search_etc.best_params_
logging.info(f"Best parameters for ExtraTreesClassifier: {best_params_etc}")

# Evaluate the best model on the test set
y_pred_etc = best_etc.predict(X_test)
accuracy_etc = accuracy_score(y_test, y_pred_etc)
logging.info(f"Test set accuracy for ExtraTreesClassifier: {accuracy_etc}")

print(f"Best parameters for ExtraTreesClassifier: {best_params_etc}")
print(f"Test set accuracy for ExtraTreesClassifier: {accuracy_etc}")
