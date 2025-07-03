import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

current_dir = Path(__file__).resolve().parent
csv_path = current_dir.parent / 'knowledge' / 'clasificadores' / 'datasets' / 'abandono_escolar.csv'

data = pd.read_csv(csv_path, delimiter=';')

x = data.drop('Target', axis=1)
y = data['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)    

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

def value_x_train():
    return x_train

def value_model():
    return model
