# Import libraries
import pandas as pd
import numpy as np
#Features Extraction with Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

#Other libraries
import seaborn as sns


loans = pd.read_csv('processed_data_cleaned_loans_2007.csv')
print(loans.shape)
print(loans.head())

# - - - - - Methode 1 - - - -

array = loans.values
X = array[:,0:39]
Y = array[:,38]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:39,:])


# - - - - - Methode 2 - - - -

# load data
array = loans.values
X = array[:,0:39]
Y = array[:,38]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 6)
fit = rfe.fit(X, Y)
print("Nombre de features demandées :"+str(fit.n_features_))
print("Features sélectionnées : "+str(fit.support_))
print("Rang des features : "+str(fit.ranking_))

# - - - - - Methode 3 - - - -

# load data
array = loans.values
X = array[:,0:39]
Y = array[:,38]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

# - - - Après selection de nos features creation d'une nouvelle DataFrame - - -

finalDataFrame = pd.concat([loans['grade'],loans['term_36months'],loans['term_60months'],loans['verification_status_not_verified'],loans['loan_amnt'],loans['installment']])

finalDataFrame.to_csv("FinalV1.csv",index="False")