import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the Titanic dataset (for example, from a CSV file)
titanic_data = pd.read_csv("titanic.csv")

# Data preprocessing: fill missing values, encode categorical features, and select relevant features
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=["Sex"], prefix=["Gender"])
titanic_data = titanic_data[["Gender_female", "Gender_male", "Age", "Pclass", 'Siblings/Spouses Aboard', 'Parents/Children Aboard', "Fare", "Survived"]]

# Create a SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
titanic_data = pd.DataFrame(imputer.fit_transform(titanic_data), columns=titanic_data.columns)

# Split the data into train and test sets
X = titanic_data.drop(columns=["Survived"])
y = titanic_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a HistGradientBoostingClassifier
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict classes on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of HistGradientBoostingClassifier for Titanic Survival Classification: {accuracy}")
