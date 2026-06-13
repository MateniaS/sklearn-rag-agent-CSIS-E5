# Docker demo run

Qdrant endpoint: `qdrant:6333`
Collection: `sklearn_rag_v2_structured`

## Demo question 1

How can a complete classification workflow combine preprocessing, model training, cross-validation and evaluation?

### Agent decision

- Tool: `rag_retriever`
- Thought: The question combines multiple workflow stages, so full-corpus retrieval is more appropriate.

Arguments:

```json
{
  "question": "How can a complete classification workflow combine preprocessing, model training, cross-validation and evaluation?",
  "top_k": 5
}
```

### Answer

A complete classification workflow in scikit-learn can combine preprocessing, model training, cross-validation, and evaluation effectively by using Pipelines and cross-validation tools as follows:

1. **Preprocessing and Model Training**: Preprocessing steps such as standardization or feature selection should be learned from the training data and then applied to the test data to avoid data leakage. This can be done manually by fitting a transformer (e.g., `StandardScaler`) on the training set and then transforming both training and test sets before training a model (e.g., SVM).

2. **Using Pipelines**: To streamline this process and ensure that preprocessing is correctly applied during cross-validation, scikit-learn provides `Pipeline` objects. A Pipeline chains preprocessing steps and the estimator so that all steps are applied consistently during training and testing. For example, a pipeline can include a scaler and an SVM classifier.

3. **Cross-validation**: Cross-validation (e.g., k-fold CV) is used to evaluate the model's generalization performance without needing a separate validation set. The training data is split into k folds; the model is trained on k-1 folds and validated on the remaining fold, repeated k times. Using a Pipeline within cross-validation ensures that preprocessing is done inside each fold correctly.

4. **Evaluation**: Metrics such as accuracy can be computed on the validation folds during cross-validation using functions like `cross_val_score`. This provides an estimate of model performance that is less biased by overfitting.

Putting it all together, a typical workflow might look like this:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, ShuffleSplit

# Define a pipeline with preprocessing and classifier
clf = make_pipeline(StandardScaler(), SVC(C=1))

# Define cross-validation strategy
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# Evaluate the pipeline with cross-validation
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)
```

This approach ensures that preprocessing, model training, cross-validation, and evaluation are combined in a robust and reproducible manner.

### Sources
- Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html (Sections 3.1 and 3.1.1)
- Pipelines and composite estimators: https://scikit-learn.org/stable/modules/cross_validation.html (Section 3.1.1)

### Retrieved chunks

1. `D05` | Cross-validation | cross_validation | score=0.5393
   URL: https://scikit-learn.org/stable/modules/cross_validation.html
2. `D05` | Cross-validation | cross_validation | score=0.5172
   URL: https://scikit-learn.org/stable/modules/cross_validation.html
3. `D06` | Grid search | hyperparameter_tuning | score=0.5105
   URL: https://scikit-learn.org/stable/modules/grid_search.html
4. `D05` | Cross-validation | cross_validation | score=0.5078
   URL: https://scikit-learn.org/stable/modules/cross_validation.html
5. `D05` | Cross-validation | cross_validation | score=0.5069
   URL: https://scikit-learn.org/stable/modules/cross_validation.html

## Demo question 2

Which RandomForestClassifier parameters can control model complexity?

### Agent decision

- Tool: `metadata_filtered_retriever`
- Thought: The question asks about RandomForestClassifier parameters, so the API-specific topic should be used.

Arguments:

```json
{
  "question": "Which RandomForestClassifier parameters can control model complexity?",
  "topic_filter": "random_forest_classifier",
  "top_k": 5
}
```

### Answer

The parameters of RandomForestClassifier that can control model complexity include:

- **max_depth**: The maximum depth of the tree. Limiting the depth controls the complexity by restricting how deep each tree can grow.
- **min_samples_split**: The minimum number of samples required to split an internal node. Increasing this value can reduce complexity by preventing splits that create small nodes.
- **min_samples_leaf**: The minimum number of samples required to be at a leaf node. Larger values can smooth the model and reduce complexity.
- **max_features**: The number of features to consider when looking for the best split. Limiting features can reduce complexity.
- **max_leaf_nodes**: Limits the number of leaf nodes in the tree, controlling complexity by pruning the tree structure.
- **min_impurity_decrease**: A node will be split if this split induces a decrease of the impurity greater than or equal to this value, controlling complexity by pruning.
- **ccp_alpha**: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity smaller than ccp_alpha will be chosen, effectively pruning the tree to control complexity.

These parameters help to control overfitting and the overall complexity of the random forest model.

Sources:
- RandomForestClassifier — scikit-learn 1.4.0 documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### Retrieved chunks

1. `D10` | RandomForestClassifier | random_forest_classifier | score=0.4761
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2. `D10` | RandomForestClassifier | random_forest_classifier | score=0.4735
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
3. `D10` | RandomForestClassifier | random_forest_classifier | score=0.4639
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
4. `D10` | RandomForestClassifier | random_forest_classifier | score=0.4597
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
5. `D10` | RandomForestClassifier | random_forest_classifier | score=0.4436
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## Demo question 3

Who won the FIFA World Cup in 2022?

### Agent decision

- Tool: `rag_retriever`
- Thought: No specific topic filter was clearly required, so full-corpus retrieval is used.

Arguments:

```json
{
  "question": "Who won the FIFA World Cup in 2022?",
  "top_k": 5
}
```

### Answer

The available context does not contain enough information to answer this question.

Sources:
- Model evaluation - https://scikit-learn.org/stable/modules/model_evaluation.html
- Random forests - https://scikit-learn.org/stable/modules/ensemble.html#forest
- LogisticRegression - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Grid search - https://scikit-learn.org/stable/modules/grid_search.html

### Retrieved chunks

1. `D07` | Model evaluation | metrics | score=0.0996
   URL: https://scikit-learn.org/stable/modules/model_evaluation.html
2. `D07` | Model evaluation | metrics | score=0.0957
   URL: https://scikit-learn.org/stable/modules/model_evaluation.html
3. `D09` | Random forests | random_forest | score=0.0879
   URL: https://scikit-learn.org/stable/modules/ensemble.html#forest
4. `D08` | LogisticRegression | logistic_regression | score=0.0822
   URL: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
5. `D06` | Grid search | hyperparameter_tuning | score=0.0803
   URL: https://scikit-learn.org/stable/modules/grid_search.html
