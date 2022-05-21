# Pokemon Legendary Classifier
## Goal: Use Pokemon stats to predict whether a Pokemon is legendary in order to practice classification algorithms like K Nearest Neighbors (kNN) or Logistic Regression.

### Pokemon stats (predictors):
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

To see the code I wrote to analyze the data and build kNN and logistic regression models: [click here](https://github.com/papir805/pokemon_classification/blob/main/pkmn_legendary_classification_knn.ipynb)

## Method
1) Left-Join Pokemon table and Combats table.
2) Transform numerical predictors such that they're all on the same scale.  
    - kNN classification compares Euclidean distance between points when classifying an observation.  Some of our numeric values are on a larger scale than others, which will have an impact on Euclidean distance, and may skew our understanding of the strength of a given predictor in the model. 
3) Check correlation between Legendary and all predictors.  Visualize the relationships using:
    - Heatmaps
    - Scatter Plots
    - Histograms
4) Create models of varying complexity and use hyperparameter tuning to determine optimal number of neighors (k).
5) Train models using optimal k neighbors and check performance on testing data.
6) Evaluate model performance on new data from Pokemon generation 8.


# kNN Results:
Using hyperparameter tuning with five-fold cross validation, a model with k=5 neighbors produced the highest testing score (~0.96) and achieved a weighted average of ~0.96 for precision, recall, and on the f1-score for testing data.  The model's only predictor variable was an aggregate called `Total Stats`, which was the sum of HP, Attack, Defense, Special Attack, Special Defense, and Speed.

The model was trained and tested on data from Pokemon generation 1-7, but I was curious to see how it performed on data from a different generation.  Feeding the model generation 8 data resulted in a weighted average of precision of 0.85, recall 0.81, and f1-score 0.75.

# Logistic Regression Results:
Coming Soon.