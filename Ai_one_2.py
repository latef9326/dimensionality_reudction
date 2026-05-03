import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error

df = pd.read_csv("Real estate.csv")

df = df.drop(columns=["No"]) # Remove the "No" column as it is just an index and does not contribute to the prediction

X = df.drop(columns=["Y house price of unit area"])
y = df["Y house price of unit area"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
max_err = max_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Maximum Error:", max_err)