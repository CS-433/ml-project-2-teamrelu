import xgboost as xgb

def xgb_training(trainloader, testloader, num_boost_round, learning_rate, dropout, lambda_, verbose):

    evals_result={}
    num_class=15
    xgboost_params = {
        'learning_rate': learning_rate,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 1.0 - dropout,
        'colsample_bytree': 1.0 - dropout,
        'lambda': lambda_,
        'alpha': 2,
        'gamma': 0.5,
        'objective': 'multi:softmax',
        'num_class': num_class,
        'eval_metric': ['mlogloss', 'merror']
    }

    batch_train = next(iter(trainloader))
    _, X_train_batch, _, _, y_train_batch, _ = batch_train


    batch_test = next(iter(testloader))
    _, X_test_batch, _, _, y_test_batch, _ = batch_test

    dtrain = xgb.DMatrix(X_train_batch.numpy(), label=y_train_batch.numpy())
    dtest = xgb.DMatrix(X_test_batch.numpy(), label=y_test_batch.numpy())

    model = xgb.train(params=xgboost_params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    evals_result = evals_result,
                    verbose_eval=25 if verbose == True else False)

    train_loss = evals_result['train']['mlogloss']
    test_loss = evals_result['test']['mlogloss']
    train_accuracy = [1 - e for e in evals_result['train']['merror']]
    test_accuracy = [1 - e for e in evals_result['test']['merror']]

    return train_loss, train_accuracy, test_loss, test_accuracy, model


{   
    "file_path": "datasets/final_dataset_dyn_with_te.csv",
    "test_size": 0.2,
    "batch_size": 32,
    "static_model": {
      "model_type": 2,
      "num_epochs": 500,
      "patience": None,
      "dropout": 0.2,
      "intermediate_dim": None,
      "learning_rate": 0.001,
      "weight_decay": 9,
      "criterion": None,
      "scheduler": None,
      "long_embeddings": true
    },

    "dynamic_model": {
      "model_type": 0,
      "num_epochs": 100,
      "patience": 20,
      "dropout": 0.2,
      "intermediate_dim": 32,
      "learning_rate": 1e-3,
      "weight_decay": 1e-2,
      "lambda_penalty": 1e-4,
      "criterion": "CrossEntropyLoss",
      "scheduler": {
        "type": "StepLR",
        "step_size": 10,
        "gamma": 0.1,
        "T_max": 60,
        "eta_min": 1e-5
      },
      "no_concentration": false,
      "no_interactions": false,
      "no_static": false,
      "static_learnable": false
    }
  }