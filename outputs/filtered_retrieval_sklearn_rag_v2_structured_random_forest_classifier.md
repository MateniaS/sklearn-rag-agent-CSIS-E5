# Filtered retrieval test — sklearn_rag_v2_structured

Question: Which RandomForestClassifier parameters can control model complexity?

Metadata filter: topic = `random_forest_classifier`

Top 5 retrieved chunks:

## Result 1

- Score: 0.4761
- Chunk ID: D10_v2_005
- Doc ID: D10
- Title: RandomForestClassifier
- Topic: random_forest_classifier
- Section: RandomForestClassifier #
- URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Preview:

ding trees (if bootstrap=True ) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features ). See Glossary for details. verbose int, default=0 Controls the verbosity when fitting and predicting. warm_start bool, default=False When set to True , reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See Glossary and Fitting additional trees for details. class_weight {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None Weights associated with classes in the form {class_label: weight} . If not given, all classes are supposed to have weight one.

## Result 2

- Score: 0.4735
- Chunk ID: D10_v2_002
- Doc ID: D10
- Title: RandomForestClassifier
- Topic: random_forest_classifier
- Section: RandomForestClassifier #
- URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Preview:

A random forest classifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Trees in the forest use the best split strategy, i.e. equivalent to passing splitter="best" to the underlying DecisionTreeClassifier . The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree. For a comparison between tree-based ensemble models see the example Comparing Random Forests and Histogram Gradient Boosting models . This estimator has native support for missing value

## Result 3

- Score: 0.4639
- Chunk ID: D10_v2_030
- Doc ID: D10
- Title: RandomForestClassifier
- Topic: random_forest_classifier
- Section: RandomForestClassifier #
- URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Preview:

The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data, max_features=n_features and bootstrap=False , if the improvement of the criterion is identical for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during fitting, random_state has to be fixed. L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001. >>> from sklearn.ensemble import RandomForestClassifier >>> from sklearn.datasets import make_classification >>> X , y = make_classification ( n_samples = 1000 , n_features = 4 , ... n_informative = 2 , n_redundant = 0 , ... random_state = 0 , shuffle = Fa

## Result 4

- Score: 0.4597
- Chunk ID: D10_v2_001
- Doc ID: D10
- Title: RandomForestClassifier
- Topic: random_forest_classifier
- Section: RandomForestClassifier #
- URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Preview:

class sklearn.ensemble. RandomForestClassifier ( n_estimators = 100 , * , criterion = 'gini' , max_depth = None , min_samples_split = 2 , min_samples_leaf = 1 , min_weight_fraction_leaf = 0.0 , max_features = 'sqrt' , max_leaf_nodes = None , min_impurity_decrease = 0.0 , bootstrap = True , oob_score = False , n_jobs = None , random_state = None , verbose = 0 , warm_start = False , class_weight = None , ccp_alpha = 0.0 , max_samples = None , monotonic_cst = None ) [source] #

## Result 5

- Score: 0.4436
- Chunk ID: D10_v2_017
- Doc ID: D10
- Title: RandomForestClassifier
- Topic: random_forest_classifier
- Section: RandomForestClassifier #
- URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Preview:

proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)) The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown. For multi-output, the weights of each column of y will be multiplied. Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified. ccp_alpha non-negative float, default=0.0 Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Prunin
