import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;

train = pd.read_csv('../input/titanic/train.csv');
test = pd.read_csv('../input/titanic/test.csv');
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);

train.head(15)

train.isnull().sum()/train.shape[0]*100

test.isnull().sum()/test.shape[0]*100

meantrain = train['Age'].mean();
meantest = test['Age'].mean()

train.update(train['Age'].fillna(meantrain));
test.update(test['Age'].fillna(meantest))

train = train.drop(columns = ['Name', 'Ticket'], axis = 1);
test = test.drop(columns = ['Name', 'Ticket'], axis = 1)

e = {'S': 1, 'C': 2, 'Q': 3};
train['Embarked'] = train['Embarked'].map(e);
test['Embarked'] = test['Embarked'].map(e);

s = {'male': 1, 'female': 2};
train['Sex'] = train['Sex'].map(s);
test['Sex'] = test['Sex'].map(s);

plt.figure(figsize = (25,25))
fineTech_appData3 = train.drop(['Survived'], axis = 1) 
sns.barplot(fineTech_appData3.columns,fineTech_appData3.corrwith(train['Survived']))

train = train.drop(columns = ['PassengerId', 'Age', 'SibSp'], axis =1);
test = test.drop(columns = ['PassengerId', 'Age', 'SibSp'], axis =1)

train['Fare'] = train['Fare'].astype('int32')
test['Fare'] = test['Fare'].astype('int32')

m = train.drop(columns = ['Survived'], axis =1);
n = train['Survived']

mtrain, mtest, ntrain, ntest = train_test_split(m ,n, test_size = 0.2, random_state = 55)

xgb = XGBClassifier();
xgb.fit(mtrain, ntrain)

xgb.score(mtest, ntest)

from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(mtrain,ntrain)

lgb.score(mtest, ntest)

rfc = RandomForestClassifier();
rfc.fit(mtrain, ntrain)

rfc.score(mtest, ntest)
