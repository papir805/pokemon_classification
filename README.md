# Pokémon Legendary Classifier
## Goal: Use Pokémon stats to predict whether a Pokémon is legendary, in order to practice classification algorithms like K Nearest Neighbors (kNN) or Logistic Regression.

### Pokémon stats (predictors):
* HP
* Attack
* Defense
* Special Attack
* Special Defense
* Speed
* Type 1
* Type 2
* Number of wins in battle

Important Python libraries used: `SKlearn`, `Pandas`, `Matplotlib`, `seaborn`

## How to use this repository:
To see the data that was used: [pokemon_table](https://github.com/papir805/pokemon_classification/blob/main/pokemon_data/Pokemon_with_correct_pkmn_numbers.csv) or [combats_table](https://github.com/papir805/pokemon_classification/blob/main/pokemon_data/combats.csv)

To see the code I wrote to analyze the data and build kNN and logistic regression models: [click here](https://github.com/papir805/pokemon_classification/blob/main/pkmn_legendary_classification.ipynb)

## Method
1) Left-Join Pokémon table and Combats table.
2) Check correlation between Legendary and all predictors.  Visualize the relationships using:
    - Heatmaps
    - Scatter Plots
    - Histograms
3) kNN
    - Transform numerical predictors such that they're all on the same scale.  
        - kNN classification compares Euclidean distance between points when classifying an observation.  Some of our numeric values are on a larger scale than others, which will have an impact on Euclidean distance, and may skew our understanding of the strength of a given predictor in the model.
    - Create models of varying complexity and use hyperparameter tuning to determine optimal number of neighors (k).
    - Train models using optimal k neighbors on generation 1-6 data and estimate testing performance using k-Fold Repeated Cross Validation.
    - Evaluate model performance on new data from Pokémon generation 7.
4) Logistic Regression
    - Train model on generation 1-6 data and estimate testing performance using k-Fold Repeated Cross Validation.
    - Evaluate model performance on new data from Pokémon generation 7.


# kNN Results:
Using hyperparameter tuning with five-fold cross validation, a model with k=5 neighbors produced the highest testing score (96%) and achieved a weighted average of ~0.96 for precision, recall, and on the f1-score for testing data.  The model's only predictor variable was an aggregate called `Total Stats`, which was the sum of `HP`, `Attack`, `Defense`, `Special Attack`, `Special Defense`, and `Speed`.

Despite having high predictive accuracy, given that such a small percentage of our population was actually legendary (roughly 8%), the model was heavily biased towards predicting a Pokémon as being non-legendary.  If one employed a strategy of only predicting non-legendary, one would achieve roughly 92% accuracy, so the kNN model is only performing slightly better that that. 

The model was trained and tested on data from Pokemon generation 1-6, but I was curious to see how it performed on data from a different generation.  Feeding the model generation 7 data resulted in a weighted average of precision of 0.85, recall 0.81, and f1-score 0.75.

When tested on generation 7 data, the model suffered a sizeable hit to accuracy, losing nearly 10%.  This is largely due to the fact that the kNN model is heavily biased towards predicting non-legendary, however in generation 7 there is a lower proportion of non-legendary Pokémon (81%) as compared to generation 1-6 (92%).  The proportion is roughly 10% lower, hence the 10% hit on accuracy.

All that being said, kNN doesn't look like a great model to use for predicting whether a Pokémon is legendary or not and will search for other models that might work better instead.

# Logistic Regression Results:
Fitting a logistic regression model on generation 1-6 data, using `Total Stats` as the only predictor achieved a ten-fold cross validation of roughly 93%.  This model was also heavily biased towards predicting non-legendary and when tested on generation 7 data, achieved 81% accuracy making the **exact same predictions** as were made by the kNN model.  

Unfortunately a logistic regression model doesn't seem to offer any advantage over the kNN model and I'm going to continue searching for something better.
