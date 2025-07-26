import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import optuna

# Load datasets
train = pd.read_csv('../../data/fin_metrics_pipeline/fin_metrics_train.csv')
val = pd.read_csv('../../data/fin_metrics_pipeline/fin_metrics_val.csv')
test = pd.read_csv('../../data/fin_metrics_pipeline/fin_metrics_test.csv')

# Separate features and target variable
X_train = train.drop(columns=['Valuation Label', 'Name'])
y_train = train['Valuation Label']
X_val = val.drop(columns=['Valuation Label', 'Name'])
y_val = val['Valuation Label']
X_test = test.drop(columns=['Valuation Label', 'Name'])
y_test = test['Valuation Label']

# Ensure 'ICB Industry' is of type 'category'
X_train['ICB Industry'] = X_train['ICB Industry'].astype('category')
X_val['ICB Industry'] = X_val['ICB Industry'].astype('category')
X_test['ICB Industry'] = X_test['ICB Industry'].astype('category')

# Identify categorical columns
cat_features = ['ICB Industry']  # Specify the column as categorical
#
#
# # Objective function for Optuna optimization (with GPU)
# def objective(trial):
#     param = {
#         'iterations': trial.suggest_int('iterations', 500, 1500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
#         'border_count': trial.suggest_int('border_count', 32, 128),
#         'cat_features': cat_features,  # Specify categorical features
#         'task_type': 'GPU',  # Enable GPU acceleration
#         'verbose': 100,
#         'early_stopping_rounds': 50
#     }
#
#     # Initialize and train the model
#     model = CatBoostClassifier(**param)
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, cat_features=cat_features)
#
#     # Evaluate the model on the test set
#     y_pred = model.predict(X_test)
#     accuracy = classification_report(y_test, y_pred, output_dict=True)['accuracy']
#
#     return accuracy
#
#
# # Optuna optimization process
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
#
# # Output best trial details
# best_trial = study.best_trial
# print(f"Best Trial: {best_trial.params}")
#
# # Final model training with the best parameters
# best_model = CatBoostClassifier(**best_trial.params)
# best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, cat_features=cat_features)
#
# # Evaluate the final model on the test set
# y_pred = best_model.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# # Save the model after training
# best_model.save_model('catboost_model.cbm', format='cbm')

# Loading the model back if needed
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')

# Making predictions with the loaded model
predictions = loaded_model.predict(X_test)

# Evaluate the final model on the test set
y_pred = loaded_model.predict(X_test)
print(classification_report(y_test, y_pred))
