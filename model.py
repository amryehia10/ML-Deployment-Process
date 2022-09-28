import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("iris.csv")

#print(df.head())

X = df.drop('Class', axis=1)

y = df["Class"]

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
