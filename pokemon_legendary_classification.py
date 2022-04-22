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
# #### Evaluating model performance - Which n_neighbors is most optimal?

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_no_total, target_df, test_size=0.2, random_state=5)
training_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
avg_cross_val_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train, y_train)
    cross_val_scores = cross_val_score(knn, scaled_with_dummies_no_total, target_df, cv=5)
    avg_cross_val_score = np.mean(cross_val_scores)
    avg_cross_val_scores.append(avg_cross_val_score)
plt.scatter(k_values, avg_cross_val_scores, color='red', marker='x')
plt.plot(k_values, avg_cross_val_scores)


# %% [markdown]
# If we create a model for values of k from 3 to 100, all using the same train/test split, then check the performance of each model, we see ~92% or better performance for most values of k, with peak performance around k=20.  However, since train/test split is a random process, it's hard to say if the model truly performs best when k=20, or if it just performs best for the particular training/testing data we used.  To overcome this obstacle, we can build many models with different training/testing data and identify which values of k work best most frequently.

# %% [markdown]
# ##### Creating a function

# %%
def bootstrap_knn(scaled_df, target_df, num_iterations=1, max_k=10, weight='uniform'):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    
    # To avoid potential ties, we only choose odd values for k
    k_values = np.arange(3, max_k, 2)
    scores_df = pd.DataFrame(index=k_values)
    
    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(scaled_df, target_df, test_size=0.2)
        training_scores = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_train, y_train)
            new_score = knn.score(X_test, y_test)
            training_scores.append(new_score)
        
        col_label = "iter" + str(i)
        scores_df[col_label] = training_scores
    
    scores_df = scores_df.rename_axis('k')
    return scores_df


# %%
def get_n_neighbors_counts(input_df):
    import pandas as pd
    min_k_vals = []
    for col in input_df.columns:
        # Get index of first entry that has a max value in input_df[col]
        min_k_val = input_df[col].idxmax()
        min_k_vals.append(min_k_val)
    k_counts = pd.Series(min_k_vals).value_counts()
    k_counts_sorted = k_counts.sort_index()
    return k_counts_sorted


# %% [markdown]
# #### Model using uniform weights

# %%
iterations = 50
high_k = 60

# %%
experiments_uniform = bootstrap_knn(scaled_with_dummies_no_total, target_df, num_iterations=iterations, max_k=high_k)

# %%
experiments_uniform.head()

# %%
get_n_neighbors_counts(experiments_uniform)

# %%
get_n_neighbors_counts(experiments_uniform).plot(kind='bar')

# %% [markdown]
# Because k=3 scored the best in regards to model performance for the vast majority of models that were built, we'll use k=3 in our final model.

# %% [markdown]
# #### Model using weighted distances

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_no_total, target_df, test_size=0.2, random_state=5)
training_scores = []
k_values = np.arange(3, 101, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
avg_cross_val_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #knn.fit(X_train, y_train)
    cross_val_scores = cross_val_score(knn, scaled_with_dummies_no_total, target_df, cv=5)
    avg_cross_val_score = np.mean(cross_val_scores)
    avg_cross_val_scores.append(avg_cross_val_score)
plt.scatter(k_values, avg_cross_val_scores, color='red', marker='x')
plt.plot(k_values, avg_cross_val_scores)

# %%
experiments_distance = bootstrap_knn(scaled_with_dummies, target_df, num_iterations=iterations, max_k=high_k, weight='distance')

# %%
get_n_neighbors_counts(experiments_distance)

# %%
get_n_neighbors_counts(experiments_distance).plot(kind='bar')

# %% [markdown]
# ### Creating Model with all features, except HP, Attack, Defense, Sp. Attack, and Sp. Defense are swapped for an aggregate called total_stats.  Our model might not need to know the individual stats to perform well.

# %%
stats_labels = numeric_cols_labels[1:]
scaled_with_dummies_total = scaled_with_dummies.drop(stats_labels, axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(scaled_with_dummies_total, target_df, test_size=0.2, random_state=5)
training_scores = []
k_values = np.arange(3, 101, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
avg_cross_val_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #knn.fit(X_train, y_train)
    cross_val_scores = cross_val_score(knn, scaled_with_dummies_total, target_df, cv=5)
    avg_cross_val_score = np.mean(cross_val_scores)
    avg_cross_val_scores.append(avg_cross_val_score)
plt.scatter(k_values, avg_cross_val_scores, color='red', marker='x')
plt.plot(k_values, avg_cross_val_scores)

# %%
total_stats_w_distance = bootstrap_knn(scaled_with_dummies_total, target_df, num_iterations=iterations, max_k=high_k, weight='distance')

# %%
get_n_neighbors_counts(total_stats_w_distance)

# %%
get_n_neighbors_counts(total_stats_w_distance).plot(kind='bar')

# %% [markdown]
# So far all signs point to k=3 being a number that will frequently produce a model that scores highest

# %% [markdown]
# ### Creating model that focuses solely on numerical values that were highly correlated with legendary: Total Stats, Sp. Atk, Sp. Def, Attack, Speed, and Wins

# %% [markdown]
# #### Model using Total Stats and Wins

# %%
total_stats_and_wins = scaled_with_dummies[['Total Stats', 'Wins']]

# %%
X_train, X_test, y_train, y_test = train_test_split(total_stats_and_wins, target_df, test_size=0.2, random_state=5)
training_scores = []
k_values = np.arange(3, 101, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
avg_cross_val_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #knn.fit(X_train, y_train)
    cross_val_scores = cross_val_score(knn, total_stats_and_wins, target_df, cv=5)
    avg_cross_val_score = np.mean(cross_val_scores)
    avg_cross_val_scores.append(avg_cross_val_score)
plt.scatter(k_values, avg_cross_val_scores, color='red', marker='x')
plt.plot(k_values, avg_cross_val_scores)

# %% [markdown]
# #### Model using Sp. Atk, Sp. Def, Attack, Speed, and Wins

# %%
individual_stats_and_wins = scaled_with_dummies[['Sp. Atk', 'Sp. Def', 'Attack', 'Speed', 'Wins']]

# %%
X_train, X_test, y_train, y_test = train_test_split(individual_stats_and_wins, target_df, test_size=0.2, random_state=5)
training_scores = []
k_values = np.arange(3, 101, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    new_score = knn.score(X_test, y_test)
    training_scores.append(new_score)
plt.scatter(k_values, training_scores, color='red', marker='x')
plt.plot(k_values, training_scores)

# %%
avg_cross_val_scores = []
# To avoid potential ties, we only choose odd values for k
k_values = np.arange(3, 100, 2)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    #knn.fit(X_train, y_train)
    cross_val_scores = cross_val_score(knn, individual_stats_and_wins, target_df, cv=5)
    avg_cross_val_score = np.mean(cross_val_scores)
    avg_cross_val_scores.append(avg_cross_val_score)
plt.scatter(k_values, avg_cross_val_scores, color='red', marker='x')
plt.plot(k_values, avg_cross_val_scores)

# %% [markdown]
# So far it's looking like we can get very good performance with low k (num_neighbors) and using as few as only two features, Total Stats and Wins

# %% [markdown]
# # GridSearchCV

# %%
from sklearn.model_selection import GridSearchCV

# %%
knn2 = KNeighborsClassifier()

# %%
param_grid = {'n_neighbors': np.arange(1,101,2)}

# %%
knn2_gscv = GridSearchCV(knn2, param_grid, cv=5)

# %%
knn2_gscv.fit(total_stats_and_wins, target_df)

# %%
knn2_gscv.best_params_, knn2_gscv.best_score_

# %%
knn3 = KNeighborsClassifier(weights='distance')
knn3_gscv = GridSearchCV(knn3, param_grid, cv=5)
knn3_gscv.fit(total_stats_and_wins, target_df)

# %%
knn3_gscv.best_params_, knn3_gscv.best_score_

# %% [markdown]
# Using n_neighors = 5 and weighting by distance appears to offer the best performance.

# %% [markdown]
# ## Picking a winner - Creating model using Total Stats and Wins as input features

# %%
X_train, X_test, y_train, y_test = train_test_split(total_stats_and_wins, target_df, test_size=0.2, random_state=6)

knn_final = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_final.fit(X_train, y_train)

# %% [markdown]
# ### Evaluating Performance

# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

# %%
y_preds = knn_final.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
knn_final_score = knn_final.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds, pos_label=None, average='weighted')
precision = precision_score(y_test, y_preds, pos_label=None, average='weighted')
recall = recall_score(y_test, y_preds, pos_label=None, average='weighted')

# %%
knn_final_score, accuracy, f1, precision, recall

# %%
print(classification_report(y_test, y_preds))

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_preds, cmap='Greens', display_labels=['Non-Legendary', 'Legendary'], colorbar=False);

# %%
total_stats = scaled_with_dummies.loc[:,'Total Stats']

# %% [markdown]
# # Final model using only total stats

# %%
total_stats = scaled_with_dummies.loc[:,'Total Stats']
total_stats = np.array(total_stats).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(total_stats, target_df, test_size=0.2, random_state=6)

knn_final2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_final2.fit(X_train, y_train)

# %% [markdown]
# ### Evaluating Performance

# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

# %%
y_preds = knn_final2.predict(X_test)
confusion_matrix(y_test, y_preds)

# %%
knn_final2_score = knn_final2.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds, pos_label=None, average='weighted')
precision = precision_score(y_test, y_preds, pos_label=None, average='weighted')
recall = recall_score(y_test, y_preds, pos_label=None, average='weighted')

# %%
knn_final2_score, accuracy, f1, precision, recall

# %%
print(classification_report(y_test, y_preds))

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_preds, cmap='Greens', display_labels=['Non-Legendary', 'Legendary'], colorbar=False);

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
