import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib

matplotlib.use('Agg')

df = pd.read_excel('ML_rf_model.demo.xlsx')

X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessing = Pipeline([
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif))
])

pipeline_rf = Pipeline([
    ('preprocessing', preprocessing),
    ('model', RandomForestClassifier(class_weight='balanced'))  # 设置类权重为'balanced'
])

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

num_features = X_train.shape[1]
param_grid_rf = {
    'preprocessing__feature_selection__k': range(1, num_features+1),
    'model__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000],
    'model__max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'model__min_samples_split': [1, 3, 5, 7, 9],
    'model__min_samples_leaf': [1, 2, 3, 4, 5],
    'model__max_features': ['auto', 'sqrt', 'log2', None],
    'model__oob_score': [True]
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring=scoring, refit='accuracy', n_jobs=30)
grid_search_rf.fit(X_train, y_train)

print(f"Random Forest best parameters: {grid_search_rf.best_params_}")

best_estimator_rf = grid_search_rf.best_estimator_
predictions_rf = best_estimator_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf, average='weighted')
recall_rf = recall_score(y_test, predictions_rf, average='weighted')
f1_rf = f1_score(y_test, predictions_rf, average='weighted')
print(f"Random Forest - Test Accuracy: {accuracy_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1 Score: {f1_rf:.4f}")

joblib.dump(best_estimator_rf, 'best_estimator_rf.joblib')

conf_matrix_rf = confusion_matrix(y_test, predictions_rf)
conf_matrix_df_rf = pd.DataFrame(conf_matrix_rf, index=best_estimator_rf.classes_, columns=best_estimator_rf.classes_)
conf_matrix_df_rf.to_csv('Confusion_Matrix_RF.csv')
print("Confusion Matrix:")
print(conf_matrix_df_rf)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df_rf, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Confusion_Matrix_RF.pdf')

y_probas_rf = best_estimator_rf.predict_proba(X_test)
n_classes = len(best_estimator_rf.classes_)

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

plt.figure()
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_probas_rf[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {best_estimator_rf.classes_[i]}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for each class')
plt.legend(loc="best")
plt.savefig('combined_precision_recall_curve_rf.pdf')

plt.figure()
colors = ['blue', 'red', 'green', 'purple', 'orange']
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probas_rf[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {best_estimator_rf.classes_[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for each class')
plt.legend(loc="lower right")
plt.savefig('combined_roc_curve_rf.pdf')