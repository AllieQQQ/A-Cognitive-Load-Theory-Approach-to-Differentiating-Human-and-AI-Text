# 3.2 Identification Test: "The initial classifier trained on all 48 metrics achieved an accuracy of 69.2% when identifying the real author of a given text. The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) is 0.797, suggesting a moderate model performance."
# output after running the initial model
Classification Report:
              precision    recall  f1-score   support

       Human       0.77      0.67      0.71        15
          AI       0.62      0.73      0.67        11

    accuracy                           0.69        26
   macro avg       0.69      0.70      0.69        26
weighted avg       0.70      0.69      0.69        26

AUC-ROC Score: 0.797

# output with feature selection improvement
Features to drop due to high correlation (>0.8): ['VS2', 'T50W', 'TTR', 'CTTR', 'RTTR', 'LogTTR', 'Uber', 'HDD', 'MTLD-MA', 'MTLD-bii', 'LV', 'CVV1', 'CVV2', 'NV', 'MODV', 'MLS', 'MLT', 'C/S', 'VP/T', 'C/T', 'DC/C', 'DC/T', 'CN/T', 'CN/C']

Selected Features (Top 10): ['letters_per_word', 'words_per_sentence', 'words_per_text', 'LS2', 'T', 'T50S', 'MATTR', 'VV1', 'CP/T', 'CP/C']

New Model (with Feature Selection) Classification Report:
              precision    recall  f1-score   support

       Human       0.80      0.80      0.80        15
          AI       0.73      0.73      0.73        11

    accuracy                           0.77        26
   macro avg       0.76      0.76      0.76        26
weighted avg       0.77      0.77      0.77        26

New Model AUC-ROC Score: 0.861

Performance Comparison:
Original Model Accuracy: 0.692
New Model Accuracy: 0.769
Original Model AUC-ROC: 0.797
New Model AUC-ROC: 0.861

# output for hyperameters tuning
Best Parameters from GridSearchCV: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}

Tuned Model (with Feature Selection) Classification Report:
              precision    recall  f1-score   support

       Human       0.87      0.87      0.87        15
          AI       0.82      0.82      0.82        11

    accuracy                           0.85        26
   macro avg       0.84      0.84      0.84        26
weighted avg       0.85      0.85      0.85        26

Tuned Model AUC-ROC Score: 0.885