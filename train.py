"""Train only on training folds; select by CV F1 before evaluating holdout."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score

ROOT = Path(__file__).resolve().parent
NUM = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
FEATURES = NUM + CAT

def load_data():
    path = ROOT / 'data/telco.csv'
    if not path.exists():
        raise SystemExit('Missing data/telco.csv. See README for the download link.')
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if not df['Churn'].isin(['Yes', 'No']).all():
        raise ValueError('Churn must contain Yes or No.')
    return df

def build_pipeline(model):
    preprocess = ColumnTransformer([
        ('numbers', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM),
        ('categories', OneHotEncoder(handle_unknown='ignore'), CAT)
    ])
    return Pipeline([('prepare', preprocess), ('model', model)])

def main():
    df = load_data()
    X, y = df[FEATURES], df.Churn.eq('Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    models = {'Logistic Regression': LogisticRegression(max_iter=2000),
              'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {name: float(cross_val_score(build_pipeline(model), X_train, y_train, cv=cv, scoring='f1').mean()) for name, model in models.items()}
    winner = max(scores, key=scores.get)
    results = {}
    for name, model in {**models, 'Majority baseline': DummyClassifier(strategy='most_frequent')}.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {'cv_f1': scores.get(name), 'test_report': classification_report(y_test, pred, output_dict=True, zero_division=0), 'test_roc_auc': float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))}
        if name == winner:
            joblib.dump(pipe, ROOT / 'model.joblib')
            ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['Stayed', 'Left'], cmap='Blues')
            plt.title(f'{winner} — unseen test customers')
            plt.tight_layout()
            plt.savefig(ROOT / 'reports/confusion_matrix.png', dpi=160)
            plt.close()
    summary = {'selected_model': winner, 'rows': len(df), 'train_rows': len(X_train), 'test_rows': len(X_test), 'threshold': .5, 'results': results}
    (ROOT / 'reports/metrics.json').write_text(json.dumps(summary, indent=2))
    rates = df.groupby('Contract').Churn.apply(lambda s: s.eq('Yes').mean()).sort_values()
    rates.plot.barh(color='#197b85', xlabel='Share of customers who left', title='Observed churn by contract (descriptive only)')
    plt.tight_layout()
    plt.savefig(ROOT / 'reports/churn_by_contract.png', dpi=160)
    plt.close()
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
