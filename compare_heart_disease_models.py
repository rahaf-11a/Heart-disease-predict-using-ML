"""Compare logistic regression with RFE, random forest, and KNN on heart.csv.

This is a reproducible educational experiment, not a clinical risk model.
Duplicate rows are removed before the train/test split. All trainable
preprocessing and feature selection are fitted only on training data.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42


def load_data(path: Path):
    data = pd.read_csv(path)
    if 'target' not in data.columns:
        raise ValueError("Expected a 'target' column in the dataset")
    original_count = len(data)
    data = data.dropna(subset=['target']).drop_duplicates().reset_index(drop=True)
    if not set(data['target'].unique()).issubset({0, 1}):
        raise ValueError('Expected binary target values 0 and 1')
    print(f'Original rows: {original_count}; unique complete rows: {len(data)}; '
          f'duplicates removed: {original_count - len(data)}')
    X, y = data.drop(columns=['target']), data['target'].astype(int)
    # Reject an ambiguous dataset where identical inputs carry conflicting labels.
    conflict_count = data.groupby(list(X.columns), dropna=False)['target'].nunique().gt(1).sum()
    if conflict_count:
        raise ValueError(f'{conflict_count} identical feature rows have conflicting labels')
    return X, y


def make_models():
    return {
        'Logistic Regression + RFE': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rfe', RFE(LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                        n_features_to_select=7)),
            ('model', LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]),
        'KNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate three heart disease classifiers')
    parser.add_argument('--data', type=Path, default=Path('heart.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f'Train size: {len(X_train)}; test size: {len(X_test)}')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in make_models().items():
        # Only the training partition is used for cross-validation.
        cross_val = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring={'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'},
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})')
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=['No disease', 'Disease']).plot(
            cmap='Blues', values_format='d', colorbar=False,
        )
        plt.title(name)
        plt.tight_layout()
        plt.savefig(args.output / (name.lower().replace(' ', '_').replace('+', 'plus') + '_confusion.png'), dpi=160)
        plt.close()
        row = {
            'model': name,
            'cv_accuracy_mean': cross_val['test_accuracy'].mean(),
            'cv_f1_mean': cross_val['test_f1'].mean(),
            'cv_roc_auc_mean': cross_val['test_roc_auc'].mean(),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, proba),
        }
        rows.append(row)
        print(f'{name}: test accuracy={row["test_accuracy"]:.3f}, '
              f'F1={row["test_f1"]:.3f}, AUC={row["test_roc_auc"]:.3f}')
        if name == 'Logistic Regression + RFE':
            columns = X_train.columns[model.named_steps['rfe'].support_].tolist()
            print('Logistic Regression RFE selected features:', ', '.join(columns))
        if name == 'Random Forest':
            importances = pd.Series(model.named_steps['model'].feature_importances_, index=X.columns)
            importances.sort_values(ascending=False).to_csv(args.output / 'random_forest_feature_importance.csv', header=['importance'])

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Held-out test ROC curves')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(args.output / 'roc_comparison.png', dpi=160)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / 'comparison_metrics.csv', index=False)
    print('\n', summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\nSaved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
