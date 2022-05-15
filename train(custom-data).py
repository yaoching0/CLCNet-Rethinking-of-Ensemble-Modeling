import pandas as pd
import numpy as np
import os
from CLCNet.tab_model import CLCNet
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('--cls-output', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','(named)effb4_output_ds_for_tabnet.csv'),
                        help='Path of classification model results for CLCNet training.')
parser.add_argument('--weight-save', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights'),
                        help='Path of CLCNet weight to save.')
parser.add_argument('--max-epochs', type=int, default=200,
                        help='Maximum stopping epoch for training CLCNet.')
parser.add_argument('--patience', type=int, default=50,
                        help='Number of consecutive non improving epoch before early stopping.')
parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size.')
parser.add_argument('--virtual_batch_size', type=int, default=128,
                        help='Batch size for Ghost Batch Normalization of TabNet (virtual_batch_size < batch_size).')
                        
cf = parser.parse_args()

if __name__ == '__main__':
    
    # Load previously generated datasets
    data = pd.read_csv(cf.cls_output,header=None)

    # Discard filename column
    data=data.iloc[:,1:]
    data.columns=list(range(data.shape[1]))
    data=data.reset_index(drop=True)

    training_data_dim=data.shape[1]-1

    # Randomly split training/validation/test sets
    data["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(data.shape[0],))

    train_indices = data[data.Set=="train"].index
    valid_indices = data[data.Set=="valid"].index
    test_indices = data[data.Set=="test"].index

    X_train = data[list(range(training_data_dim))].values[train_indices]
    y_train = data[training_data_dim].values[train_indices].reshape(-1, 1)

    X_valid = data[list(range(training_data_dim))].values[valid_indices]
    y_valid = data[training_data_dim].values[valid_indices].reshape(-1, 1)

    X_test = data[list(range(training_data_dim))].values[test_indices]
    y_test = data[training_data_dim].values[test_indices].reshape(-1, 1)

    # Give higher weight to negative samples
    weights=np.ones([y_train.size]) / y_train.size
    weights=pd.DataFrame(y_train)[0].map({0:1,1:0}).values*(1/y_train.size)+weights
    
    # Define the CLCNet model
    clf = CLCNet()

    # Train CLCNet
    clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=[ 'mse',],
    max_epochs=cf.max_epochs,
    weights=weights,
    patience=cf.patience,
    batch_size=cf.batch_size,
    virtual_batch_size=cf.virtual_batch_size, 
    num_workers=0,
    drop_last=False
    ) 

    # Compute MSE on the test set
    preds = clf.predict(X_test)
    test_score = mean_squared_error(y_pred=preds, y_true=y_test)

    # save weight
    save_path=os.path.join(cf.weight_save,f'(Custom-data)CLCNet(MSE_{round(test_score,4)})')
    saved_filepath = clf.save_model(save_path)