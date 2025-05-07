# # Task 19 -- Feature Importance Analysis Using SHAP

# import shap
# import pandas as pd 
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt 
# import seaborn as sns

# # Load dataset
# df = pd.read_csv('diabetes dataset.csv')

# # Separate Features and Target Variables
# X = df.drop(columns='Outcome')
# y = df['Outcome']

# # Create Train & Test Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)

# # Ensure X_test is a DataFrame (required for SHAP to work with correct columns)
# X_test = pd.DataFrame(X_test, columns=X.columns)

# # Build and train the model
# rf_clf = RandomForestClassifier(max_features=2, n_estimators=100, bootstrap=True)
# rf_clf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = rf_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # shap 
# print("\nCalculating SHAP values...")
# explainer = shap.TreeExplainer(rf_clf)
# shap_values = explainer(X)

# np.shape(shap_values.values)
# print(shap_values.shape)


# # waterfall plot
# shap.plots.waterfall(shap_values[24, :, 0])
# shap.plots.waterfall(shap_values[24, :, 1])

# # Bar plot
# shap.plots.bar(shap_values[:, :, 0])
# shap.plots.bar(shap_values[:, :, 1])

# # Beeswarm plot
# shap.plots.beeswarm(shap_values[:, :, 1])
# shap. plots.beeswarm(shap_values[:, :, 0])

# # scatter plot
# shap.plots.scatter(shap_values[:, "Glucose", 1])
# shap.plots.scatter(shap_values[:, "Glucose", 0])

# # Get mean absolute SHAP values
# shap_df = pd.DataFrame(np.abs(shap_values.values[:, :, 1]).mean(axis=0), index=X.columns, columns=["mean_abs_shap"])
# shap_df = shap_df.sort_values("mean_abs_shap", ascending=False) # sorting the mean abs of SHAP
# print(shap_df) # printing the SHAP values of features


import shap
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes dataset.csv')

# Separate features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Feature scaling for model robustness
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=13
)

# Train a more robust Random Forest model
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=5, max_features='sqrt', random_state=42
)
rf_clf.fit(X_train, y_train)

# Evaluate model
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# SHAP Analysis
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer(X_scaled)

# Waterfall plot for one individual
print("\nWaterfall plot for a sample:")
shap.plots.waterfall(shap_values[24])

# Bar plot for average feature importance
print("\nBar plot of mean absolute SHAP values:")
shap.plots.bar(shap_values)

# Beeswarm plot (global interpretation)
print("\nBeeswarm plot (global feature impact):")
shap.plots.beeswarm(shap_values)

# Scatter plot for a feature
print("\nSHAP scatter plot for Glucose:")
shap.plots.scatter(shap_values[:, "Glucose"])

# Create and display SHAP summary DataFrame
shap_df = pd.DataFrame(
    np.abs(shap_values.values).mean(axis=0),
    index=X.columns,
    columns=["mean_abs_shap"]
).sort_values("mean_abs_shap", ascending=False)

print("\nMean Absolute SHAP Values by Feature:")
print(shap_df)
