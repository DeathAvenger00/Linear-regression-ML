import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style 
import matplotlib.pyplot as plt

data = pd.read_csv("student-mat.csv", sep=";")
data = data[
    ["G1", "G2", "G3", "failures", "studytime", "absences", "famrel", "freetime", "Dalc",  "health",
     "absences"]]
predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

p = "failures"
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()