import json
import pandas as pd
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
import seaborn as sns
from scipy.stats import ttest_ind

# merge files with results to the metrics(this part of the code was run in colab so blocks concerning saving and downloading files probably needs revision)
# Basic length measures JSON
basic_data = json.load(io.StringIO(uploaded['add_the basic length results file.json'].decode('utf-8')))

# Lexical complexity CSV
lexical_df = pd.read_csv(io.StringIO(uploaded['add lexical_metrics_results.csv'].decode('utf-8')))

# Syntactic complexity JSON
syntactic_data = json.load(io.StringIO(uploaded['add the actual syntactic_metrics_comparison.json'].decode('utf-8')))

basic_rows = []
for item in basic_data:
    article_id = item['article_id']
    for kind, metrics in item.items():
        if kind in ['ground_truth_metrics', 'generated_continuation_metrics']:
            prefix = 'ground_truth_' if 'ground' in kind else 'generated_'
            flat = {f'{prefix}{k}': v for k, v in metrics.items()}
            flat['article_id'] = article_id
            basic_rows.append(flat)

basic_df = pd.DataFrame(basic_rows)
basic_df = basic_df.groupby('article_id').first().reset_index()

# Filter only ground_truth and generated rows
lexical_filtered = lexical_df[lexical_df['text_type'].isin(['ground_truth', 'generated'])]
lexical_pivot = lexical_filtered.pivot(index='article_id', columns='text_type').reset_index()
# Flatten MultiIndex columns
lexical_pivot.columns = ['article_id'] + [f'{col[1]}_{col[0]}' for col in lexical_pivot.columns[1:]]

syntactic_rows = []
for article_id, item in syntactic_data.items():
    row = {'article_id': article_id}
    for key, metrics in item.items():
        prefix = 'ground_truth_' if 'ground' in key else 'generated_'
        row.update({f'{prefix}{k}': v for k, v in metrics.items()})
    syntactic_rows.append(row)

syntactic_df = pd.DataFrame(syntactic_rows)

# Convert article_id in all DataFrames to string before merging
basic_df['article_id'] = basic_df['article_id'].astype(str)
lexical_pivot['article_id'] = lexical_pivot['article_id'].astype(str)
syntactic_df['article_id'] = syntactic_df['article_id'].astype(str)

# Now safe to merge
merged_df = basic_df.merge(lexical_pivot, on='article_id')
merged_df = merged_df.merge(syntactic_df, on='article_id')

# Convert article_id to string for consistency
merged_df['article_id'] = merged_df['article_id'].astype(str)

merged_df.to_csv('merged_metrics.csv', index=False)
merged_df.to_json('merged_metrics.json', orient='records', indent=2)

# Download the results
files.download('merged_metrics.csv')
files.download('merged_metrics.json')

df = pd.read_csv('merged_metrics.csv')

# Prepare a new list of rows
rows = []

for idx, row in df.iterrows():
    article_id = row['article_id']

    # Collect human (ground truth) metrics
    human_row = {'article_id': article_id, 'source': 'human', 'label': 0}
    for col in df.columns:
        if col.startswith('ground_truth_'):
            base_col = col.replace('ground_truth_', '')
            human_row[base_col] = row[col]

    # Collect AI-generated metrics
    ai_row = {'article_id': article_id, 'source': 'ai', 'label': 1}
    for col in df.columns:
        if col.startswith('generated_'):
            base_col = col.replace('generated_', '')
            ai_row[base_col] = row[col]

    # Add to final list
    rows.extend([human_row, ai_row])

# Create a new DataFrame
long_df = pd.DataFrame(rows)

# Save to CSV
long_df.to_csv('text_classification_data.csv', index=False)

# Download it
from google.colab import files
files.download('text_classification_data.csv')

# random forest 
# Load the CSV file

df = pd.read_csv('text_classification_data.csv') # the file before

# Preprocess the data
# Ensure numeric columns are properly typed and handle missing/invalid values
numeric_columns = [
    'letters_per_word', 'words_per_sentence', 'words_per_text', 'sentences_per_text',
    'LD', 'LS1', 'LS2', 'VS1', 'VS2', 'CVS1', 'T', 'T50', 'T50S', 'T50W', 'TTR',
    'MSTTR', 'CTTR', 'RTTR', 'LogTTR', 'Uber', 'MATTR', 'HDD', 'MTLD', 'MTLD-MA',
    'MTLD-bii', 'LV', 'VV1', 'SVV1', 'CVV1', 'CVV2', 'NV', 'ADJV', 'ADVV', 'MODV',
    'MLS', 'MLT', 'MLC', 'C/S', 'VP/T', 'C/T', 'DC/C', 'DC/T', 'CT/T', 'CP/T',
    'CP/C', 'CN/T', 'CN/C', 'T/S'
]

# Convert columns to numeric, coercing errors to NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Prepare features and target
# Features: all metrics except article_id, source, and label
X = df[numeric_columns]
y = df['label']  # 0 for human, 1 for AI

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# Calculate AUC-ROC
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc_roc:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': numeric_columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head(5))

df = pd.read_csv('/content/text_classification_data.csv')

# Define numeric columns (same as original script)
numeric_columns = [
    'letters_per_word', 'words_per_sentence', 'words_per_text', 'sentences_per_text',
    'LD', 'LS1', 'LS2', 'VS1', 'VS2', 'CVS1', 'T', 'T50', 'T50S', 'T50W', 'TTR',
    'MSTTR', 'CTTR', 'RTTR', 'LogTTR', 'Uber', 'MATTR', 'HDD', 'MTLD', 'MTLD-MA',
    'MTLD-bii', 'LV', 'VV1', 'SVV1', 'CVV1', 'CVV2', 'NV', 'ADJV', 'ADVV', 'MODV',
    'MLS', 'MLT', 'MLC', 'C/S', 'VP/T', 'C/T', 'DC/C', 'DC/T', 'CT/T', 'CP/T',
    'CP/C', 'CN/T', 'CN/C', 'T/S'
]

# Convert columns to numeric, coercing errors to NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Prepare features and target
X = df[numeric_columns]
y = df['label']  # 0 for human, 1 for AI

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train original Random Forest model (for comparison)
original_model = RandomForestClassifier(random_state=42, n_estimators=100)
original_model.fit(X_train_scaled, y_train)
y_pred_original = original_model.predict(X_test_scaled)
print("Original Model Classification Report:")
print(classification_report(y_test, y_pred_original, target_names=['Human', 'AI']))
auc_roc_original = roc_auc_score(y_test, original_model.predict_proba(X_test_scaled)[:, 1])
print(f"Original Model AUC-ROC Score: {auc_roc_original:.3f}")

# Calculate correlation matrix and drop highly correlated features
corr_threshold = 0.8
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
print("\nFeatures to drop due to high correlation (>0.8):", to_drop)
X_reduced = X.drop(to_drop, axis=1)

# Update numeric columns after dropping correlated features
numeric_columns_reduced = X_reduced.columns.tolist()

# Split and scale reduced feature set
X_train_reduced = X_train[numeric_columns_reduced]
X_test_reduced = X_test[numeric_columns_reduced]
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

# Select top 10 features
model_for_rfe = RandomForestClassifier(random_state=42)
rfe = RFE(model_for_rfe, n_features_to_select=10)
rfe.fit(X_train_reduced_scaled, y_train)
selected_features = X_train_reduced.columns[rfe.support_].tolist()
print("\nSelected Features (Top 10):", selected_features)

# Prepare final feature set with selected features
X_train_selected = X_train_reduced[selected_features]
X_test_selected = X_test_reduced[selected_features]
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Train feature-selected Random Forest model (for comparison)
feature_selected_model = RandomForestClassifier(random_state=42, n_estimators=100)
feature_selected_model.fit(X_train_selected_scaled, y_train)
y_pred_feature_selected = feature_selected_model.predict(X_test_selected_scaled)
print("\nFeature-Selected Model Classification Report:")
print(classification_report(y_test, y_pred_feature_selected, target_names=['Human', 'AI']))
auc_roc_feature_selected = roc_auc_score(y_test, feature_selected_model.predict_proba(X_test_selected_scaled)[:, 1])
print(f"Feature-Selected Model AUC-ROC Score: {auc_roc_feature_selected:.3f}")

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform grid search on the selected features
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_selected_scaled, y_train)
print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)

# Train tuned model with best parameters
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test_selected_scaled)
print("\nTuned Model (with Feature Selection) Classification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=['Human', 'AI']))
auc_roc_tuned = roc_auc_score(y_test, tuned_model.predict_proba(X_test_selected_scaled)[:, 1])
print(f"Tuned Model AUC-ROC Score: {auc_roc_tuned:.3f}")

# Performance Comparison
print("\nPerformance Comparison:")
print(f"Original Model Accuracy: {classification_report(y_test, y_pred_original, output_dict=True)['accuracy']:.3f}")
print(f"Feature-Selected Model Accuracy: {classification_report(y_test, y_pred_feature_selected, output_dict=True)['accuracy']:.3f}")
print(f"Tuned Model Accuracy: {classification_report(y_test, y_pred_tuned, output_dict=True)['accuracy']:.3f}")
print(f"Original Model AUC-ROC: {auc_roc_original:.3f}")
print(f"Feature-Selected Model AUC-ROC: {auc_roc_feature_selected:.3f}")
print(f"Tuned Model AUC-ROC: {auc_roc_tuned:.3f}")

# EDA
df = pd.read_csv('/content/text_classification_data.csv')

# Define numeric columns (same as used in model training)
numeric_columns = [
    'letters_per_word', 'words_per_sentence', 'words_per_text', 'sentences_per_text',
    'LD', 'LS1', 'LS2', 'VS1', 'VS2', 'CVS1', 'T', 'T50', 'T50S', 'T50W', 'TTR',
    'MSTTR', 'CTTR', 'RTTR', 'LogTTR', 'Uber', 'MATTR', 'HDD', 'MTLD', 'MTLD-MA',
    'MTLD-bii', 'LV', 'VV1', 'SVV1', 'CVV1', 'CVV2', 'NV', 'ADJV', 'ADVV', 'MODV',
    'MLS', 'MLT', 'MLC', 'C/S', 'VP/T', 'C/T', 'DC/C', 'DC/T', 'CT/T', 'CP/T',
    'CP/C', 'CN/T', 'CN/C', 'T/S'
]

# Convert columns to numeric, coercing errors to NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Separate human and AI texts
human_df = df[df['label'] == 0]
ai_df = df[df['label'] == 1]

# Calculate summary statistics
summary_stats = []
for col in numeric_columns:
    human_stats = {
        'Metric': col,
        'Human_Mean': human_df[col].mean(),
        'Human_Median': human_df[col].median(),
        'Human_Std': human_df[col].std(),
        'AI_Mean': ai_df[col].mean(),
        'AI_Median': ai_df[col].median(),
        'AI_Std': ai_df[col].std()
    }
    # Perform t-test
    t_stat, p_value = ttest_ind(human_df[col], ai_df[col], nan_policy='omit')
    human_stats['t_stat'] = t_stat
    human_stats['p_value'] = p_value
    summary_stats.append(human_stats)

# Create DataFrame for summary statistics
summary_df = pd.DataFrame(summary_stats)
summary_df['Significant'] = summary_df['p_value'] < 0.05  # Significance threshold

# Save summary statistics to CSV
summary_df.to_csv('/content/summary_statistics.csv', index=False)
print("Summary Statistics (first 5 rows):")
print(summary_df.head())
print("\nSignificant Metrics (p < 0.05):")
print(summary_df[summary_df['Significant']][['Metric', 'p_value']])

numeric_columns = [
    'letters_per_word', 'words_per_sentence', 'words_per_text', 'sentences_per_text',
    'LD', 'LS1', 'LS2', 'VS1', 'VS2', 'CVS1', 'T', 'T50', 'T50S', 'T50W', 'TTR',
    'MSTTR', 'CTTR', 'RTTR', 'LogTTR', 'Uber', 'MATTR', 'HDD', 'MTLD', 'MTLD-MA',
    'MTLD-bii', 'LV', 'VV1', 'SVV1', 'CVV1', 'CVV2', 'NV', 'ADJV', 'ADVV', 'MODV',
    'MLS', 'MLT', 'MLC', 'C/S', 'VP/T', 'C/T', 'DC/C', 'DC/T', 'CT/T', 'CP/T',
    'CP/C', 'CN/T', 'CN/C', 'T/S'
]

# Convert columns to numeric, coercing errors to NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Separate human and AI texts
human_df = df[df['label'] == 0]
ai_df = df[df['label'] == 1]

# Calculate summary statistics for all metrics
summary_stats = []
for col in numeric_columns:
    human_stats = {
        'Metric': col,
        'Human_Mean': human_df[col].mean(),
        'Human_Median': human_df[col].median(),
        'Human_Std': human_df[col].std(),
        'AI_Mean': ai_df[col].mean(),
        'AI_Median': ai_df[col].median(),
        'AI_Std': ai_df[col].std()
    }
    # Perform t-test
    t_stat, p_value = ttest_ind(human_df[col], ai_df[col], nan_policy='omit')
    human_stats['t_stat'] = t_stat
    human_stats['p_value'] = p_value
    summary_stats.append(human_stats)

# Create DataFrame for summary statistics
summary_df = pd.DataFrame(summary_stats)

# Filter for significant metrics (p < 0.05)
significant_df = summary_df[summary_df['p_value'] < 0.05][[
    'Metric', 'Human_Mean', 'Human_Median', 'Human_Std',
    'AI_Mean', 'AI_Median', 'AI_Std', 't_stat', 'p_value'
]]

# Round numeric columns for readability
significant_df = significant_df.round(4)

# Print the table
print("Summary Statistics for Significant Metrics (p < 0.05):")
print(significant_df.to_string(index=False))

# Save to CSV
significant_df.to_csv('add_a path for downloaing the results')

# Download the CSV file
from google.colab import files
files.download('add_a path for downloaing the results')