import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import nbformat as nbf

print("Starting SVM model training process...")
X_train = pd.read_csv('./data/X_train_balanced.csv')
y_train = pd.read_csv('./data/y_train_balanced.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv')
print("Data loaded successfully.")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train.values.ravel())
print("SVM model training completed.")
joblib.dump(svm_model, './models/svm_model.pkl')
print("SVM model saved as 'svm_model.pkl'.")
y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)[:, 1]
print("Predictions completed.")
results = pd.DataFrame({
    'TransactionID': X_test.index,
    'Fraud_Probability': y_proba.flatten(),
    'Predicted_Class': y_pred.flatten(),
    'Actual_Class': y_test.values.flatten()
})
fraudulent_transactions = results[results['Predicted_Class'] == 1]
fraudulent_transactions.to_csv('./outputs/svm_detected_frauds.csv', index=False)
print("Output saved to 'svm_detected_frauds.csv'.")
print("Generating classification report...")
classification_rep = classification_report(y_test, y_pred)
print("SVM Classification Report:")
print(classification_rep)

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_code_cell(f"""
print("SVM Classification Report:")
print('''{classification_rep}''')
"""))
nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nfrauds = pd.read_csv('./outputs/svm_detected_frauds.csv')\nfrauds.head()"))
with open('./outputs/svm_model_results.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Output notebook saved to 'svm_model_results.ipynb'.")
