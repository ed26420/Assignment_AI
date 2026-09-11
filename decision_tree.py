import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

columns = [
    'buying',
    'maintenance',
    'doors',
    'persons',
    'lug_boot',
    'safety',
    'class'
]

data = pd.read_csv('car.data', names=columns)

X = data.drop('class', axis=1)
y = data['class']

# Convert categorical values into numerical values
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use X_test to make predictions
y_pred = model.predict(X_test)

results = X_test.copy()
results['Actual Class'] = y_test
results['Predicted Class'] = y_pred
print(results)