# # Task --21 -  evaluate and compare with models Accuracies

# import pandas as pd 
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# df = pd.read_csv('diabetes dataset.csv')
# '''
# df.info()
# missing = df.isnull().sum()
# print("missing values:", missing)
# unique = df.nunique().sum()
# print("unique values:", unique) ''' 

# cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)

# # 0s with mean vbalues 
# imputer = SimpleImputer(strategy="mean")
# df[cols_with_zero_invalid] = imputer.fit_transform(df[cols_with_zero_invalid])


# # feature and target  
# X = df.drop('Outcome', axis=1)
# y = df['Outcome']

# # split of train test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# # standardization
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

# models = {
#     "Logistic Regression": LogisticRegression(max_iter = 1000, random_state = 42),
#     "SVM" : SVC(kernel= 'rbf', probability = True, C=1, gamma= 1, random_state = 42),
#     "Neural Network" : MLPClassifier(hidden_layer_sizes = (50,), max_iter = 1000, random_state = 42)
# }

# results = {}

# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
    
#     # Classification report
#     report = classification_report(y_test, y_pred, output_dict=True)
    
#     results[name] = {
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": report["1"]["precision"],
#         "Recall": report["1"]["recall"],
#         "F1-Score": report["1"]["f1-score"],
#         "Confusion Matrix": confusion_matrix(y_test, y_pred)
#     }

# # Print results
# for model_name, metrics in results.items():
#     print(f"\nModel: {model_name}")
#     print(f"Accuracy: {metrics['Accuracy']:.4f}")
#     print(f"Precision: {metrics['Precision']:.4f}")
#     print(f"Recall: {metrics['Recall']:.4f}")
#     print(f"F1-Score: {metrics['F1-Score']:.4f}")
#     print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")


# df.to_csv("new_diabetes_dataset.csv", index=False)


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('diabetes dataset.csv')

# Replace 0s with NaN in columns with invalid 0s
cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)

# Impute missing values with the mean
imputer = SimpleImputer(strategy="mean")
df[cols_with_zero_invalid] = imputer.fit_transform(df[cols_with_zero_invalid])

# Feature and target variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and hyperparameters for GridSearchCV
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
}

# Hyperparameters for optimization
param_grid = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ['liblinear', 'lbfgs']
    },
    "SVM": {
        "C": [0.01, 0.1, 1, 10],
        "gamma": ['scale', 'auto'],
        "kernel": ['linear', 'rbf', 'poly']
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50,), (100,), (150,)],
        "activation": ['relu', 'tanh'],
        "learning_rate_init": [0.001, 0.01, 0.1]
    }
}

# Store the results
results = {}

# Loop through each model and perform GridSearchCV
for name, model in models.items():
    print(f"Optimizing {name}...")
    
    grid_search = GridSearchCV(model, param_grid[name], cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Best Hyperparameters": grid_search.best_params_
    }

# Print results
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
    print(f"Best Hyperparameters: {metrics['Best Hyperparameters']}")
