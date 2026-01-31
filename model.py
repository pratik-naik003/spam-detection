import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load the data
df = pd.read_csv("dataset.csv")

# 2. Encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # spam=1, ham=0

# 3. Split the data
X = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize the text
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)

# 5. Train the model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# 6. Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved")
