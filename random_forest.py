import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

x_train = train.values[:,1:]
y_train = train.ix[:,0]
test = test.values

rf = RandomForestClassifier(n_estimators=50)
rf.fit(x_train, y_train)
prediction = rf.predict(test)

pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('rf_res.csv', index=False, header=True)
