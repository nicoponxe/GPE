

# This code is used for testing a previously saved model with new data that the model hasn't encountered before.

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as MSE, r2_score
from GPE import preprocess_dataset


file_names = [ 'ShkAngW_15ms.mat', 'ThiAngW_15ms.mat']
datasets = [preprocess_dataset(file_name) for file_name in file_names]

shankDF = datasets[0]
thighDF = datasets[1].drop('gait_percentage', axis=1)
bothDF = pd.concat([shankDF, thighDF], axis=1)

# Load the saved model from a file
with open('xgb_model.pkl40', 'rb') as f:
    xgb_r_loaded = pickle.load(f)

# Predict using the trained model
X_new = bothDF[['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']]
y_pred = xgb_r_loaded.predict(X_new)
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Gait Percentage'])

# Calculate the root mean squared error (RMSE)
actual = bothDF['gait_percentage']
rmse = np.sqrt(MSE(actual, y_pred))
print("RMSE", rmse)
# Calculate R2 score

# Plot actual vs predicted values
plt.scatter(actual, y_pred, s=10)
plt.plot(actual, actual, color='red', linewidth=2)
plt.title('Actual vs Predicted Gait Percentage (RMSE: {:.2f})'.format(rmse))
plt.xlabel('Actual Gait Percentage')
plt.ylabel('Predicted Gait Percentage')
plt.show()

r2_xgb_test = r2_score(actual, y_pred)
print("XGB Test R^2:", r2_xgb_test)