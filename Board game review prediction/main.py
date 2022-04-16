from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

games = pd.read_csv('Board Game Review Prediction/games.csv')
# print(games.columns)
# print(games.shape)
# plt.hist(games['average_rating'])
# plt.show()
print('Equal to Zero: ', games[games['average_rating'] == 0].iloc[0])
print('Greater than Zero: ', games[games['average_rating'] > 0].iloc[0])
games = games[games['users_rated'] > 0]
games = games.dropna(axis=0)
plt.hist(games['average_rating'])
# plt.show()
print('All Columns: ', games.columns)
corrmat = games.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
# plt.show()
columns = games.columns.tolist()
columns = [c for c in columns if c not in ['bayes_average_rating', 'average_rating', 'id', 'type', 'name']]
target = 'average_rating'

train = games.sample(frac=0.8, random_state=1)
tes = games.loc[~games.index.isin(train.index)]

print('Shape of Train Sample: ', train.shape)
print('Shape of Test Sample: ', tes.shape)


LR = LinearRegression()
LR.fit(train[columns], train[target])

predict = LR.predict(tes[columns])
error = mean_squared_error(predict, tes[target])
print('Mean Squared Error of LR: ', error)

RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
RFR.fit(train[columns], train[target])
predict = RFR.predict(tes[columns])
error = mean_squared_error(predict, tes[target])
print('Mean Squared Error of RFR: ', error)

print(tes[columns].iloc[0])

rating_LR = LR.predict(tes[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(tes[columns].iloc[0].values.reshape(1, -1))

print('Rating LR: ', rating_LR)
print('Rating RFR: ', rating_RFR)
print('Test of Target at Location 0: ', tes[target].iloc[0])