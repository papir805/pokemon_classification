# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
# %matplotlib inline

# %%
pkmn = pd.read_csv('Pokemon_with_correct_pkmn_numbers.csv')
pkmn.rename(columns=({'#':'Number', 'Total':'Total Stats'}), inplace=True)
# This way the index starts at 1 and we can correctly join pkmn and combats tables later
pkmn.index = pkmn.index + 1

combats = pd.read_csv("./pokemon_data/combats.csv")

# %%
# Note that while Number is not unique, as pokemon like Venusaur and Mega Venusaur both have Number 3, their row index is unique.  This is important bc our combats data keeps track of winners using a pokemon's row index, NOT its pokemon number.  
pkmn.head()

# %%
print(F"The pkmn df has row index starting at {pkmn.index.min()} and ending at {pkmn.index.max()}")
print(F"While the min pkmn.Number is {pkmn.Number.min()} and the max pkmn.Number is {pkmn.Number.max()}")

# %%
combats.head()

# %%
# Looking at our combats data, our max number for Winner is 800.  This is because a winning pokemon is identified by the row index in the pkmn df.  Winner DOES NOT correspond to pkmn Number.
combats['Winner'].describe()

# %% [markdown]
# # Joins

# %% [markdown]
# ## Identify names of pokemon in winning battles

# %%
# Join combats to pkmn table using the row indices (1, 800) for the pkmn table.  This ensures that a pokemon like Venusaur vs. Mega Venusaur will each have their own appropriate number of wins. 
combats_join = pd.merge(combats, pkmn[['Name']], left_on='Winner', right_index=True, how='left')
combats_join.rename(columns={'Name':"winner_name"}, inplace=True)
combats_join.head()

# %%
# Check to see if join was done correctly




# %% [markdown]
# ## Identifying number of wins for each pokemon from battles data

# %%
# Because we may have pokemon that weren't used in combat and we have no data on, we need to use a left join to preserve all 800 unique Pokemon from pkmn table after the join.  Any Pokemon that aren't found in the right df (winners) will get a NaN value.
winners = combats_join['Winner'].value_counts()
pkmn_join = pd.merge(pkmn, winners, how='left', left_index=True, right_index=True)
pkmn_join.rename(mapper={'Winner':'Wins'}, axis=1, inplace=True)
# If a pokemon has NaN for Wins, we have no data on combats for the pokemon and can be considered as having 0 wins.
pkmn_join['Wins'].fillna(value=0, inplace=True)
pkmn_join.sort_values('Wins', ascending=False)

# %%
# Check to see if join was done correctly
num_combats = len(combats)
total_wins = pkmn_join['Wins'].sum()

print(num_combats == total_wins)

# %% [markdown]
# # kNN classification - Predicting legendary status from pokemon stats (HP, Defense, ..., num_wins_in_combat)

# %% [markdown]
# ## Extra granularity - building a model without total stats, looking at each stat individually.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# %%
pkmn_join_copy = pkmn_join.copy(deep=True)
# Drop Total Stats column as we have more granularity if we look at each stat individually.  We can consider building a model that looks at total stats later on and compare performance to the model we build now.
pkmn_join_copy.drop('Total Stats', axis=1, inplace=True)

numeric_cols_labels = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Wins']

numeric_cols = pkmn_join_copy.loc[:, numeric_cols_labels]

# %% [markdown]
# ### Scale and transform the data

# %%
# kNN classification compares Euclidean distance between points when creating the model.  Some of our numeric values are on a larger scale than others, which will have an impact on Euclidean distance, and may disproportionately favor certain columns in the model as a result.  To overcome this issue, we transform our numeric data so all columns are on the same scale.
scaler = StandardScaler()
scaler.fit(numeric_cols)
pkmn_join_copy.loc[:, numeric_cols_labels] = scaler.transform(numeric_cols)

# kNN classification requires quantitative values as input.  For categorical data, we can convert to dummy variables, which are quantitative, and allow for the use of categorical data in the model. 
categorical_cols = pkmn_join_copy.loc[:, ['Generation', 'Type 1', 'Type 2']]
categorical_cols_labels = list(categorical_cols.columns)
scaled_with_dummies = pd.get_dummies(pkmn_join_copy.drop(['Number', 'Name', 'Legendary'], axis=1), columns=categorical_cols_labels)

# Lastly, we separate our target labels from the rest of the dataset
target_df = pkmn_join_copy['Legendary']

# %%
scaled_with_dummies.head()

# %%
scaled_with_dummies.columns

# %%

# %%

# %% [markdown]
# ### Train/Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies, target_df, test_size=0.2)

# %% [markdown]
# ### Creating the model

# %%
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# %% [markdown]
# ### Evaluating model performance

# %%
training_scores = []
k_values = np.arange(3, 21, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies, target_df, test_size=0.2)
training_scores = []
k_values = np.arange(3, 21, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# try knn again but take out wins column to see if accuracy goes down.  If it does, then this would be a good example of feature engineering because I added the wins column on my own.

# %%
pkmn_new = pkmn.copy(deep=True)
pkmn_new.drop_duplicates(subset='Number', keep='last', inplace=True)
pkmn_new.drop(labels='Number', axis=1, inplace=True)
pkmn_new

# %%
numeric_cols = pkmn_new.select_dtypes(include=[np.int, np.float]).drop('Generation', axis=1)
new_scaler = StandardScaler()
new_scaler.fit(numeric_cols)
new_cols = new_scaler.transform(numeric_cols)
pkmn_new[list(numeric_cols.columns)] = new_cols


target_df = pkmn_new['Legendary']
pkmn_new.drop(['Legendary', 'Name'], axis=1, inplace=True)



# %%
object_cols_labels = list(pkmn_new.select_dtypes(include=object).columns)
object_cols_labels.append('Generation')
scaled_with_dummies_2 = pd.get_dummies(pkmn_new, columns=object_cols_labels)
scaled_with_dummies_2.columns

# %%
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(scaled_with_dummies_2, target_df)
neigh.score(scaled_with_dummies_2, target_df)

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_2, target_df, test_size=0.2)
scores = []
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    scores.append(score)
plt.plot(k_values, scores)
plt.show()


preds = neigh.predict(X_test)
confusion_matrix(y_test, preds)

# %%
preds = neigh.predict(X_test)
confusion_matrix(y_test, preds)

# %%
dropped = scaled_with_dummies_2.drop(['Type 1_Electric', 'Generation_2', 'Type 2_Rock', 'Total'], axis=1)
dropped

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(dropped, target_df, test_size=0.35)
scores = []
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    scores.append(score)
plt.plot(k_values, scores)

# %%
scaled_with_dummies_2

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_2.drop(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def'], axis=1), target_df, test_size=0.3)
scores = []
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    scores.append(score)
plt.plot(k_values, scores)

# %%
target_df_int = target_df.astype(int)
target_df_int

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_2, target_df_int, test_size=0.3)
scores = []
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    scores.append(score)
plt.plot(k_values, scores)

# %%
new_list = numeric_cols_labels[:]
new_list.append('Wins')

X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies[new_list], target_df_int, test_size=0.3)
scores = []
for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    print(score)
    scores.append(score)
plt.plot(k_values, scores)


preds = neigh.predict(X_test)

# %%
y_test

# %%
new_list

# %%
scaled_with_dummies

# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
score = neigh.score(X_test, y_test)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, pos_label=None, average='weighted')
precision = precision_score(y_test, preds, pos_label=None, average='weighted')
recall = recall_score(y_test, preds, pos_label=None, average='weighted')

# %%
score, accuracy, f1, precision, recall

# %%
print(classification_report(y_test, preds))

# %%
confusion_matrix(y_test, preds)

# %%
from sklearn.metrics import plot_confusion_matrix

# %%
plot_confusion_matrix(neigh, X_test, y_test, display_labels=['Non-Legendary', 'Legendary'], cmap=plt.cm.Blues)

# %%
plot_confusion_matrix(neigh, X_test, y_test, display_labels=['Non-Legendary', 'Legendary'], cmap=plt.cm.Blues, normalize='true')

# %%

# %%

# %%

# %%
scaled_with_dummies

# %%

# %%

# %%
from sklearn.linear_model import LogisticRegression

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies, target_df, test_size=0.4)
lr = LogisticRegression().fit(X_train, y_train)
lr.score(X_test, y_test)

# %%
plot_confusion_matrix(lr, X_test, y_test, cmap=plt.cm.Blues)

# %%
lr_preds = lr.predict(X_test)
print(classification_report(y_test, lr_preds))

# %% [markdown]
# # Logistic Regression - Predicting Legendary Status

# %%
pkmn_join.corr()

# %%
plt.figure(figsize=(10,10))
sns.heatmap(pkmn_join.corr(), cmap=plt.get_cmap('gist_gray'))

# %% [markdown]
# Based on the correlation df and heatmap above, Legendary is most highly correlated with the following 3 quantitative variables:
#     1. Total
#     2. Sp. Atk
#     3. Sp. Def
# As total is the sum of all of the other stats, HP, Attack, Defense, Sp. Atk, Sp. Def, and Speed, I'd expect it to be collinear with the other variables.  Let's check:

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
pkmn_copy = pkmn_join.copy(deep=True)
pkmn_copy['Legendary'] = pkmn_copy['Legendary'].astype('int')

# %%
variance_inflation_factor(pkmn_copy.loc[:,['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']].values, 6)

# %%
pkmn_copy.loc[:,['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]

# %%
X


# %%
def calc_vif(X):
    """
    X: A pandas DataFrame object of numerical independent variables to be used in regression,
    Calculates the variance inflation factor of each independent variable in X
    against all of the other independent variables in X"""
    vif = pd.DataFrame()
    vif['Variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return(vif)


# %%
new_df = pkmn_copy.loc[:,['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']]
X = new_df.iloc[:,:-1]
calc_vif(X)

# %%
new_df = pkmn_copy.loc[:,['HP', 'Attack', 'Defense', 'Sp. Atk', 'Speed', 'Legendary']]
X = new_df.iloc[:,:-1]
calc_vif(X)

# %%
new_df = pkmn_copy.loc[:,['HP', 'Defense', 'Sp. Atk', 'Speed', 'Legendary']]
X = new_df.iloc[:,:-1]
calc_vif(X)

# %%
new_df = pkmn_copy.loc[:,['Defense', 'Sp. Atk', 'Speed', 'Legendary']]
X = new_df.iloc[:,:-1]
calc_vif(X)

# %%
new_df = pkmn_copy.loc[:,['Defense', 'Speed', 'Legendary']]
X = new_df.iloc[:,:-1]
calc_vif(X)

# %%

# %%

# %%
#scaled_with_dummies
new_df = scaled_with_dummies.loc[:,['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
X = new_df.iloc[:,:]
calc_vif(X)

# %%

# %%
lr = LogisticRegression(random_state=4)
lr.fit(pkmn_join.loc[:,['Defense', 'Speed']], target_df)
lr.coef_

# %%
lr = LogisticRegression(random_state=4)
lr.fit(scaled_with_dummies.loc[:,['Defense', 'Speed']], target_df)
lr.coef_

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Regression

# %%
pkmn_join.plot.scatter('Total', 'Wins')

# %%
pkmn_join.Wins

# %%
pkmn_join['Wins']
