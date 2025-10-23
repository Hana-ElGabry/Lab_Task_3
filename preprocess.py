import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load raw Titanic data
df = pd.read_csv('data/titanic_raw.csv')
print("Titanic dataset loaded successfully!")

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Select relevant features
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save preprocessed data
os.makedirs('data', exist_ok=True)
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Preprocessing complete. Files saved: train.csv and test.csv")
