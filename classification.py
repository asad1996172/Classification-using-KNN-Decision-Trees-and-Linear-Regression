 # Input variables:
# # bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
#
# Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
#
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


data = pd.read_csv("bank-new.csv")
data = data.iloc[np.random.permutation(len(data))]    ## shuffle data

data.loc[data["job"] == "admin.", "job"] =1
data.loc[data["job"] == "blue-collar", "job"] =2
data.loc[data["job"] == "entrepreneur", "job"] =3
data.loc[data["job"] == "housemaid", "job"] =4
data.loc[data["job"] == "management", "job"] =5
data.loc[data["job"] == "self-employed", "job"] =6
data.loc[data["job"] == "services", "job"] =7
data.loc[data["job"] == "student", "job"] =8
data.loc[data["job"] == "technician", "job"] =9
data.loc[data["job"] == "unemployed", "job"] =10
data.loc[data["job"] == "unknown", "job"] =0
data.loc[data["job"] == "retired", "job"] =11


data.loc[data["marital"] == "divorced", "marital"] =3
data.loc[data["marital"] == "married", "marital"] =2
data.loc[data["marital"] == "single", "marital"] =1
data.loc[data["marital"] == "unknown", "marital"] =0


data.loc[data["education"] == "tertiary", "education"] =3
data.loc[data["education"] == "secondary", "education"] =2
data.loc[data["education"] == "primary", "education"] =1
data.loc[data["education"] == "unknown", "education"] =0

data.loc[data["default"] == "no", "default"] =0
data.loc[data["default"] == "yes", "default"] =1

data.loc[data["housing"] == "no", "housing"] =0
data.loc[data["housing"] == "yes", "housing"] =1

data.loc[data["loan"] == "no", "loan"] =0
data.loc[data["loan"] == "yes", "loan"] =1


data.loc[data["poutcome"] == "success", "poutcome"] =3
data.loc[data["poutcome"] == "failure", "poutcome"] =2
data.loc[data["poutcome"] == "other", "poutcome"] =1
data.loc[data["poutcome"] == "unknown", "poutcome"] =0

data.loc[data["y"] == "no", "y"] =0
data.loc[data["y"] == "yes", "y"] =1

# print(data["y"].unique())



X = data.as_matrix(columns=["age","job","marital","education","default","balance","housing","loan","day","duration","campaign","pdays","previous","poutcome"])
# X = data.as_matrix(columns=["age","job","marital","default","balance","housing","loan","day","duration","campaign","pdays","previous","poutcome"])
y = np.array(data['y'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)

## Linear Regression
regr = linear_model.Ridge (alpha=0.5)
regr.fit(X_train, y_train)

print('<---------- Results with Linear Regression ------------->')
# print('Coefficients: \n', regr.coef_)
# print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
print('Variance % for Linear Regression : ' + str(regr.score(X_test, y_test)*100))
print('<------------------------------------------------------->\n')


## K Nearest neighbours
print('<---------- Results with K Nearest Neighbours ---------->')
y = np.asarray(data['y'], dtype="|S6")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
# y_test = np.asarray(y_test,dtype=int)
print('Variance % for KNN Neighbours with (K = 3) : ' + str(neigh.score(X_test, y_test)*100))
# print("Mean squared error: %.2f" % np.mean((neigh.predict(X_test) - y_test) ** 2))
print('<------------------------------------------------------->\n')


## Decision Trees
print('<---------- Results with Decision Trees----------------->')
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, y_train)
print('Variance % for Decision Trees : ' + str(dt.score(X_test, y_test)*100))
print('<------------------------------------------------------->\n')