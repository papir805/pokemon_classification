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

# %% [markdown]
# # Start

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

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# %%
pkmn_join_copy = pkmn_join.copy(deep=True)

numeric_cols_labels = ['Total Stats', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Wins']

numeric_cols = pkmn_join_copy.loc[:, numeric_cols_labels]

# %% [markdown]
# ## Scale and transform the data

# %%
# kNN classification compares Euclidean distance between points when creating the model.  Some of our numeric values are on a larger scale than others, which will have an impact on Euclidean distance, and may disproportionately favor certain columns in the model as a result.  To overcome this issue, we transform our numeric data so all columns are on the same scale.
scaler = StandardScaler()
scaler.fit(numeric_cols)
pkmn_join_copy.loc[:, numeric_cols_labels] = scaler.transform(numeric_cols)

# kNN classification requires quantitative values as input.  For categorical data, we can convert to dummy variables, which are quantitative, and allow for the use of categorical data in the model. 
# Note, we exclude the "Generation" column.  I don't think it's reasonable to know what generation a pokemon comes from when trying to classify it as legendary.
categorical_cols = pkmn_join_copy.loc[:, ['Type 1', 'Type 2']]
categorical_cols_labels = list(categorical_cols.columns)
scaled_with_dummies = pd.get_dummies(pkmn_join_copy.drop(['Number', 'Name', 'Legendary'], axis=1), columns=categorical_cols_labels)

# Lastly, we separate our target labels from the rest of the dataset
target_df = pkmn_join_copy['Legendary']

# %%
scaled_with_dummies.head()

# %%
scaled_with_dummies.columns

# %% [markdown]
# ## Which numerical features are most highly correlated with a pokemon being legendary?

# %%
pkmn_corr = pkmn_join_copy.corr()

# %%
sns.heatmap(pkmn_corr, cmap='Reds')

# %%
# Legendary seems most highly correlated with Total Stats, Sp. Atk, Sp. Def, Attack, Speed, and Wins
pkmn_corr['Legendary'].sort_values(ascending=False)

# %%
most_corr_num_features = pkmn_corr['Legendary'].sort_values(ascending=False)[1:7].index.values

# %% tags=[]
sns.pairplot(pkmn_join_copy[['Total Stats', 'Sp. Atk', 'Sp. Def', 'Attack', 'Speed', 'Wins', 'Legendary']], hue='Legendary')

# %% [markdown]
# For most, if not all plots, we see a tendency for Legendary pokemon to cluster in the upper right of each scatter plot, indicating that Legendary pokemon tend to have high stats as compared to non-legendary pokemon.  These features are probably going to be the most important for our model's performance.

# %%
fig, ax = plt.subplots(1,1)
pkmn_join_copy[pkmn_join_copy['Legendary']==True].hist(column='Total Stats', ax=ax)
pkmn_join_copy[pkmn_join_copy['Legendary']==False].hist(column='Total Stats', ax=ax, alpha=0.5)
plt.legend(['Legendary', 'Non-Legendary'])
plt.show();

# %% [markdown]
# Focusing on the aggregated total stats, we see legendary pokemon are towards the top.  This feature alone may be sufficient for our model.

# %% [markdown]
# ## Building Models

# %% [markdown]
# ### Creating Model using all features except total stats.
#
# First, I'm going to consider building a model that uses each stat.  The extra granularity here may help build a better model, but it will be more complex as a result.  I'll first start using all available features, but then see what happens when I narrow down to focusing on the numerical values that are most highly correlated with legendary: Sp. Atk, Sp. Def, Attack, Speed, and Wins

# %%
# Drop Total Stats column as we have more granularity if we look at each stat individually.  We can consider building a model that looks at total stats later on and compare performance to the model we build now.
scaled_with_dummies_no_total = scaled_with_dummies.drop('Total Stats', axis=1)

# %%
scaled_with_dummies_no_total.head()

# %% [markdown]
# #### Which n_neighbors is most optimal and what is the performance?

# %%
from sklearn.model_selection import GridSearchCV

# %%
knn_all_features_no_total = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1,101,2)}

knn_all_features_no_total_gscv = GridSearchCV(knn_all_features_no_total, param_grid, cv=5)
knn_all_features_no_total_gscv.fit(scaled_with_dummies_no_total, target_df)

print(F"Optimal n_neighbors for model: {knn_all_features_no_total_gscv.best_params_}")
print(F"Highest model performance: {knn_all_features_no_total_gscv.best_score_}")

# %% [markdown]
# This will establish our baseline.  Let's see if we can build a model that is less complex and performs better.

# %% [markdown]
# ### Creating Model with all features, except HP, Attack, Defense, Sp. Attack, and Sp. Defense are swapped for an aggregate called total_stats.  Our model might not need to know the individual stats to perform well.

# %%
stats_labels = numeric_cols_labels[1:]
scaled_with_dummies_total = scaled_with_dummies.drop(stats_labels, axis=1)

# %% [markdown]
# #### Which n_neighbors is most optimal and what is the performance?

# %%
knn_all_features_with_total = KNeighborsClassifier()

knn_all_features_with_total_gscv = GridSearchCV(knn_all_features_with_total, param_grid, cv=5)
knn_all_features_with_total_gscv.fit(scaled_with_dummies_total, target_df)

print(F"Optimal n_neighbors for model: {knn_all_features_with_total_gscv.best_params_}")
print(F"Highest model performance: {knn_all_features_with_total_gscv.best_score_}")

# %% [markdown]
# We see worse performance with this model.  It's more complex, with an optimal n_neighbors of 41 and performs slightly worse.

# %% [markdown]
# ### Creating model that focuses solely on numerical values that were highly correlated with legendary: Total Stats, Sp. Atk, Sp. Def, Attack, Speed, and Wins

# %% [markdown]
# #### Model using Sp. Atk, Sp. Def, Attack, Speed, and Wins

# %%
individual_stats_and_wins = scaled_with_dummies[['Sp. Atk', 'Sp. Def', 'Attack', 'Speed', 'Wins']]

# %%
knn_individual_stats_and_wins = KNeighborsClassifier()

knn_individual_stats_and_wins_gscv = GridSearchCV(knn_individual_stats_and_wins, param_grid, cv=5)
knn_individual_stats_and_wins_gscv.fit(individual_stats_and_wins, target_df)

print(F"Optimal n_neighbors for model: {knn_individual_stats_and_wins_gscv.best_params_}")
print(F"Highest model performance: {knn_individual_stats_and_wins_gscv.best_score_}")

# %% [markdown]
# We're starting to see better performance when we reduce the number of inputs to the model.  Now the optimal n_neighbors is only 15, with slightly better performance than our previous models.

# %% [markdown] tags=[]
# #### Model using only Total Stats and Wins

# %%
total_stats_and_wins = scaled_with_dummies[['Total Stats', 'Wins']]

# %% [markdown]
# ##### Which n_neighbors is most optimal and what is the performance?

# %%
knn_total_stats_and_wins = KNeighborsClassifier()

knn_total_stats_and_wins_gscv = GridSearchCV(knn_total_stats_and_wins, param_grid, cv=5)
knn_total_stats_and_wins_gscv.fit(total_stats_and_wins, target_df)

print(F"Optimal n_neighbors for model: {knn_total_stats_and_wins_gscv.best_params_}")
print(F"Highest model performance: {knn_total_stats_and_wins_gscv.best_score_}")

# %% [markdown]
# This model is much less complex, having an optimal n_neighbors of 3, and better performance than our previous models.  It also only requires two input variables.  Using total stats and wins is looking to be our best model so far.

# %% [markdown]
# #### Model using only total stats

# %%
total_stats = scaled_with_dummies.loc[:,'Total Stats']
total_stats = np.array(total_stats).reshape(-1,1)

# %%
knn_total_stats = KNeighborsClassifier()

knn_total_stats_gscv = GridSearchCV(knn_total_stats, param_grid, cv=5)
knn_total_stats_gscv.fit(total_stats, target_df)

print(F"Optimal n_neighbors for model: {knn_total_stats_gscv.best_params_}")
print(F"Highest model performance: {knn_total_stats_gscv.best_score_}")

# %% [markdown]
# Slightly better performance, with a slightly more complex n_neighbors of 5, but now we only have one input variable.  Using total stats, or total stats and wins, both seem like reasonable choices for our final model.  Let's investigate their performance a little further.

# %% [markdown]
# ## Further investigating performance of our two top models

# %% [markdown] tags=[]
# ### Model: Total stats and wins

# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

# %%
X_train, X_test, y_train, y_test = train_test_split(total_stats_and_wins, target_df, test_size=0.2, random_state=5)

# %%
knn_total_stats_and_wins.set_params(n_neighbors=3)
knn_total_stats_and_wins.fit(X_train, y_train)
y_preds = knn_total_stats_and_wins.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
knn_total_stats_and_wins_score = knn_total_stats_and_wins.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds, pos_label=None, average='weighted')
precision = precision_score(y_test, y_preds, pos_label=None, average='weighted')
recall = recall_score(y_test, y_preds, pos_label=None, average='weighted')

# %%
knn_total_stats_and_wins_score, accuracy, f1, precision, recall

# %%
print(classification_report(y_test, y_preds))

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_preds, cmap='Greens', display_labels=['Non-Legendary', 'Legendary'], colorbar=False);

# %% [markdown]
# Decent performance, but let's see how our other top model performs.

# %% [markdown]
# ### Model: Total stats

# %%
X_train, X_test, y_train, y_test = train_test_split(total_stats, target_df, test_size=0.2, random_state=6)

# %%
knn_total_stats.set_params(n_neighbors=5)
knn_total_stats.fit(X_train, y_train)
y_preds = knn_total_stats.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
knn_total_stats_score = knn_total_stats.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds, pos_label=None, average='weighted')
precision = precision_score(y_test, y_preds, pos_label=None, average='weighted')
recall = recall_score(y_test, y_preds, pos_label=None, average='weighted')

# %%
knn_total_stats_score, accuracy, f1, precision, recall

# %%
print(classification_report(y_test, y_preds))

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_preds, cmap='Greens', display_labels=['Non-Legendary', 'Legendary'], colorbar=False);

# %% [markdown]
# This model looks to perform quite well.  it has ~96% for accuracy, f1, precision, and recall, using only one input.  

# %%

# %%

# %%

# %% [markdown]
# # Logistic Regression - Predicting Legendary Status

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
