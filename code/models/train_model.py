import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns


def load_titanic_data():
    df = sns.load_dataset('titanic')
    features = ['sex', 'age', 'fare', 'sibsp', 'parch', 'survived']
    df = df[features].dropna()
    return df

def train_titanic_model():
    df = load_titanic_data()
    
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0
    
    X = df[['sex', 'age', 'fare', 'sibsp', 'parch']]
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    with open('models/titanic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("Model trained and saved successfully!")


if __name__ == "__main__":
    train_titanic_model()