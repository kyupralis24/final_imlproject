import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, auc
import joblib
import nbformat as nbf

X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv')

print("Loading the saved SVM model...")
svm_model = joblib.load('./models/svm_model.pkl')
print("SVM model loaded successfully.")

print("Predicting probabilities...")
y_proba = svm_model.predict_proba(X_test)[:, 1]
print("Probabilities predicted.")

custom_threshold = 0.7
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print("Generating classification report with updated threshold...")
classification_rep = classification_report(y_test, y_pred_custom)
print("Classification Report with Custom Threshold:")
print(classification_rep)

precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc_value = auc(recall, precision)
print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc_value:.4f}")

print("Generating output CSV with fraudulent transactions...")
results = pd.DataFrame({
    'TransactionID': X_test.index,
    'Fraud_Probability': y_proba.flatten(),
    'Predicted_Class': y_pred_custom.flatten(),
    'Actual_Class': y_test.values.flatten()
})

fraudulent_transactions = results[results['Predicted_Class'] == 1]
fraudulent_transactions.to_csv('./outputs/svm_detected_frauds_updated_threshold.csv', index=False)
print("Output saved to 'svm_detected_frauds_updated_threshold.csv'.")

evaluation_results = {
    "Classification Report": classification_rep,
    "AUPRC": [auprc_value]
}
eval_df = pd.DataFrame.from_dict(evaluation_results, orient="index", columns=["Metric"])
eval_df.to_csv('./outputs/svm_model_evaluation_metrics.csv', index=True)
print("Evaluation metrics saved to 'svm_model_evaluation_metrics.csv'.")

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# SVM Model Evaluation with Updated Threshold"))
nb.cells.append(nbf.v4.new_code_cell(f"""
print("Classification Report with Custom Threshold:")
print('''{classification_rep}''')
print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc_value:.4f}")
"""))
nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nfrauds = pd.read_csv('./outputs/svm_detected_frauds_updated_threshold.csv')\nfrauds.head()"))
with open('./outputs/svm_update_threshold_results.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Output notebook saved to 'svm_update_threshold_results.ipynb'.")
