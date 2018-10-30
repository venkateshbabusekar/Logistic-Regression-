
# coding: utf-8

# # I360 Data Scientist Project Role 

# In[90]:


# Importing all the necessary package


# In[41]:


import pandas as pd
import category_encoders as ce
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[42]:


# Loading the data 
inF2 = input("Enter the  Independent variable file Name : ")
user_data = pd.read_csv(inF2)
inF1 = input("Enter the  Dependent variable file Name :")
user_response = pd.read_csv(inF1)
data = pd.merge(user_data, user_response, how='inner', on=['ID','State'])


# In[45]:


print("Starting the data cleaning process")


# # Handeling N/A Values

# In[46]:


data['f110'].fillna('MAR', inplace=True)
data['f108'].fillna('MAR', inplace=True)
data['f114'].fillna('MAR', inplace=True)
data['f118'].fillna('MAR', inplace=True)
data.loc[data['f12'].notnull(), 'f12'] = 1
data['f12'].fillna(0, inplace=True)

data.loc[data['f115'].notnull(), 'f115'] = 1
data['f115'].fillna(0, inplace=True)


data.loc[data['f119'].notnull(), 'f119'] = 1
data['f119'].fillna(0, inplace=True)

data.loc[data['f120'].notnull(), 'f120'] = 1
data['f120'].fillna(0, inplace=True)

data.loc[data['f121'].notnull(), 'f121'] = 1
data['f121'].fillna(0, inplace=True)

data.loc[data['f122'].notnull(), 'f122'] = 1
data['f122'].fillna(0, inplace=True)

data.loc[data['f126'].notnull(), 'f126'] = 1
data['f126'].fillna(0, inplace=True)

d = {'Spend to Improve Economy' : 0, 'Reduce National Debt and Deficit' : 1} 
data['SPENDINGRESPONSE'] = data['SPENDINGRESPONSE'].map(d) 

data.loc[data['f127'].notnull(), 'f127'] = 1
data['f127'].fillna(0, inplace=True)


data.loc[data['f147'].notnull(), 'f147'] = 1
data['f147'].fillna(0, inplace=True)

data = data.fillna(data.median())


# ### Changing the catergorical variable in contineous 

# In[51]:



cols = data.columns
num_cols = data._get_numeric_data().columns
categorical_columns= list(set(cols) - set(num_cols))

encoder = ce.BinaryEncoder(cols=categorical_columns)
df_binary = encoder.fit_transform(data)


# ## Removing the value with high VIF 

# In[52]:


df_binary = df_binary.drop(['f103_0','f110_0','f96_0','f95_0','f99_0','f97_0','f101_0','State_0','f3_0','f1_0','f118_0','f13_0','f114_0','f98_0'], axis=1)

df_binary = df_binary.drop(['f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83'], axis=1)

df_binary = df_binary.drop(['f88','f89','f90','f91','f92','f94','f109','f112','f115','f121','f4','f100_2','f1_1','State_1','f103_3'], axis=1)


df_binary = df_binary.drop(['f93','f135','f113','f105','f93','f5','f6','f7','f8','f9','f118_1','f3_3','f102_0','f1_2','State_2','f95_1'], axis=1)


df_binary = df_binary.drop(['f143','f136','f111'], axis=1)


df_binary = df_binary.drop(['ID'], axis=1)

print("Data cleaning process completed")


# ## Normalizing the Data

# In[60]:


print("Starting the data Normalization process")

x_temp = df_binary.loc[:, df_binary.columns != 'SPENDINGRESPONSE']
y = df_binary.loc[:, df_binary.columns == 'SPENDINGRESPONSE']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

not_normalize= x_temp.filter(['f127','f147','f103_1','f103_2','f103_4','f110_1','f110_2','f110_3','f110_4','f110_5','f108_0','f108_1','f108_2','f108_3','f108_4','f96_1','f96_2','f95_2','f95_3','f99_1','f99_2','f99_3','f99_4','f97_1','f97_2','f97_3','f97_4','f101_1','f101_2','f101_3','f101_4','State_3','State_4','State_5','State_6','f3_1','f3_2','f3_4','f3_5','f102_1','f102_2','f1_3','f1_4','f1_5','f1_6','f1_7','f1_8','f1_9','f118_2','f118_3','f118_4','f100_0','f100_1','f13_1','f114_1','f114_2','f114_3','f114_4','f114_5','f98_1','f98_2'], axis=1)

to_normalize= x_temp.filter(['f2','f10','f11','f12','f84','f85','f86','f87','f104','f106','f107','f116','f117','f119','f120','f122','f123','f124','f125','f126','f128','f129','f130','f131','f132','f133','f134','f137','f138','f139','f140','f141','f142','f144','f145','f146'], axis=1)
np_scaled = min_max_scaler.fit_transform(to_normalize)
df_normalized = pd.DataFrame(np_scaled)
names = to_normalize.columns.values 
df_normalized.columns = names
df_normalized.head(5)


x = pd.concat([not_normalize, df_normalized], axis=1, sort=False)

print("Data Normalization compeleted")


print("Building Logisting Regression Model ")


# ## Spliting of  Data (Train , test) and Traing the logistics regression 

# In[66]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
x2 = x[x.columns.difference(['f110_1','f110_2','f110_3','f110_5','f108_0','f108_1','f108_2','f108_3','f99_1','f99_4','f97_1','f97_2','f101_2','State_3','State_5','State_6','f1_4','f1_6','f1_8','f114_2','f114_3','f114_4','f114_5','f2','f12','f120','f124','f129','f130','f131','f138','f140'])]

X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y, test_size=0.3, random_state=1)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train1, y_train1)
print("Logisting Regression Model built success")


# # Loading the scored data, cleaning and predicting the output 

# In[68]:


inF = input("Enter the file scored file Name : ")
sample_file = pd.read_csv(inF)
data=sample_file


# In[71]:


print("Cleaning the scored file")
data['f3'].fillna("I", inplace=True)
data['f110'].fillna('MAR', inplace=True)
data['f108'].fillna('MAR', inplace=True)
data['f114'].fillna('MAR', inplace=True)
data['f118'].fillna('MAR', inplace=True)

data.loc[data['f12'].notnull(), 'f12'] = 1
data['f12'].fillna(0, inplace=True)

data.loc[data['f115'].notnull(), 'f115'] = 1
data['f115'].fillna(0, inplace=True)


data.loc[data['f119'].notnull(), 'f119'] = 1
data['f119'].fillna(0, inplace=True)

data.loc[data['f120'].notnull(), 'f120'] = 1
data['f120'].fillna(0, inplace=True)

data.loc[data['f121'].notnull(), 'f121'] = 1
data['f121'].fillna(0, inplace=True)

data.loc[data['f122'].notnull(), 'f122'] = 1
data['f122'].fillna(0, inplace=True)

data.loc[data['f126'].notnull(), 'f126'] = 1
data['f126'].fillna(0, inplace=True)


data.loc[data['f127'].notnull(), 'f127'] = 1
data['f127'].fillna(0, inplace=True)


data.loc[data['f147'].notnull(), 'f147'] = 1
data['f147'].fillna(0, inplace=True)

data = data.fillna(data.median())


cols = data.columns
num_cols = data._get_numeric_data().columns
categorical_columns= list(set(cols) - set(num_cols))



encoder = ce.BinaryEncoder(cols=categorical_columns)
df_binary = encoder.fit_transform(data)



df_binary = df_binary.drop(['f103_0','f110_0','f96_0','f95_0','f99_0','f97_0','f101_0','State_0','f3_0','f1_0','f118_0','f13_0','f114_0','f98_0'], axis=1)

df_binary = df_binary.drop(['f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81','f82','f83'], axis=1)

df_binary = df_binary.drop(['f88','f89','f90','f91','f92','f94','f109','f112','f115','f121','f4','f100_2','f1_1','State_1','f103_3'], axis=1)

df_binary = df_binary.drop(['f143','f136','f111'], axis=1)

df_binary = df_binary.drop(['ID'], axis=1)

df_binary = df_binary.drop(['f93','f135','f113','f105','f93','f5','f6','f7','f8','f9','f118_1','f3_3','f102_0','f1_2','State_2','f95_1'], axis=1)



x_temp = df_binary

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

not_normalize= x_temp.filter(['f127','f147','f103_1','f103_2','f103_4','f110_1','f110_2','f110_3','f110_4','f110_5','f108_0','f108_1','f108_2','f108_3','f108_4','f96_1','f96_2','f95_2','f95_3','f99_1','f99_2','f99_3','f99_4','f97_1','f97_2','f97_3','f97_4','f101_1','f101_2','f101_3','f101_4','State_3','State_4','State_5','State_6','f3_1','f3_2','f3_4','f3_5','f102_1','f102_2','f1_3','f1_4','f1_5','f1_6','f1_7','f1_8','f1_9','f118_2','f118_3','f118_4','f100_0','f100_1','f13_1','f114_1','f114_2','f114_3','f114_4','f114_5','f98_1','f98_2'], axis=1)

to_normalize= x_temp.filter(['f2','f10','f11','f12','f84','f85','f86','f87','f104','f106','f107','f116','f117','f119','f120','f122','f123','f124','f125','f126','f128','f129','f130','f131','f132','f133','f134','f137','f138','f139','f140','f141','f142','f144','f145','f146'], axis=1)
np_scaled = min_max_scaler.fit_transform(to_normalize)
df_normalized = pd.DataFrame(np_scaled)
names = to_normalize.columns.values 
df_normalized.columns = names
df_normalized.head(5)

x = pd.concat([not_normalize, df_normalized], axis=1, sort=False)


x2 = x[x.columns.difference(['f110_1','f110_2','f110_3','f110_5','f108_0','f108_1','f108_2','f108_3','f99_1','f99_4','f97_1','f97_2','f101_2','State_3','State_5','State_6','f1_4','f1_6','f1_8','f114_2','f114_3','f114_4','f114_5','f2','f12','f120','f124','f129','f130','f131','f138','f140'])]

print("Data cleaning process is completed")


# In[78]:


print("Predicting the voter support ")
y_pred1 = logreg.predict(x2)

y_pred2 = logreg.predict_proba(x2)

pred =pd.DataFrame(y_pred1)

probabilities =pd.DataFrame(y_pred2)

output = pd.concat([probabilities , pred], axis=1, sort=False)


output.columns = ['probabilities_Spend to Improve Economy', 'probabilities_Reduce National Debt ', 'Prediction']

final_df = pd.concat([sample_file , output], axis=1, sort=False)


final_df.to_csv("Probablity_Prediction_Output.csv",index=False)

print("Predicting has been completed successfully ")


print("Please check the file Probablity_Prediction_Output.csv for more details about the probabilities and the Prediction ")

