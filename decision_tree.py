import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a table showing actual and predicted results
results = X_test.copy()
results['Actual Class'] = y_test
results['Predicted Class'] = y_pred

print(results)
print("Accuracy:", accuracy)

# Display precision, recall and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix to show correct and incorrect predictions
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))