
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import os
import json
import argparse
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import pdb
import networkx as nx
import dgl
from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt
import pdb
from itertools import product
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from modules import Unfolding    

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]



def main(max_seeds: int = 5):

    parser = argparse.ArgumentParser(description='Train a AutoGluon with graph information',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datasets', '-s', default='ogbn-arxiv', type=str)
    parser.add_argument('--num_parts', type=int, default=2, help='num of partition')
    parser.add_argument('--alpha', type=float, default=0.95, help='alpha')
    parser.add_argument('--step', type=int, default=50, help='propagation step')
    parser.add_argument('--rep', type=int, default=0, help='repeat')


    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)

    ## propagation 
    propagation_DAD = Unfolding(alp=args.alpha, prop_step=args.step, kernel='DAD', clamp=True)
    propagation_AD = Unfolding(alp=args.alpha, prop_step=args.step, kernel='AD', clamp=True)
    propagation_DA = Unfolding(alp=args.alpha, prop_step=args.step, kernel='DA', clamp=True)



    data = DglNodePropPredDataset(name=args.datasets)
    graph, labels = data[0]
    splitted_idx = data.get_idx_split()
    train_id, val_id, test_id = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    evaluator = Evaluator(name=args.datasets)

    ## new feature from C&S
    # feat = torch.load('feat.pt')
    feat = graph.ndata["feat"]



    # # import pdb;pdb.set_trace()
    # if args.datasets == 'ogbn-arxiv':
    #     year = graph.ndata["year"][train_id].numpy()
    #     threshold = np.median(year)
    #     parts = np.multiply(1, year <  threshold)



    if args.datasets == 'ogbn-arxiv':
        # add reverse edges
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)





    X_Column_index = ['Column_X_{}'.format(i) for i in range(feat.shape[1])]
    X = pd.DataFrame(feat.numpy(), columns=X_Column_index)
    y = pd.DataFrame(labels.numpy(), columns=['class'])
    dataset = pd.concat([X, y], axis=1)

    # y_true = torch.from_numpy(y.to_numpy()).float()
    num_samples = X.shape[0]
    train_mask = torch.zeros(num_samples, 1)
    train_mask[train_id] = 1

    


    # smoothing the feature and cat with original feature
    # X_original = X
    # X = torch.from_numpy(X.to_numpy(copy=True)).float().squeeze()
    # X = propagation.forward(graph, X)
    # X = pd.DataFrame(X.numpy(), columns=X_original.columns)
    # dataset = pd.concat([X_original, y], axis=1)


    # parts = dgl.metis_partition_assignment(g=graph, k=args.num_parts, balance_ntypes=train_mask)
    # dataset['groups'] = parts
    # parts = dgl.metis_partition_assignment(g=graph, k=args.num_parts)



    ## label propagation
    graph = graph.to(device)




    ################################################  train AG from here ################################################
    train_data = dataset.iloc[train_id]
    test_data = dataset.iloc[test_id]
    val_data = dataset.iloc[val_id]



    # train AG with group index
    save_path = f'log/{args.datasets}/NN/phase_one'  # specifies folder to store trained models
    # base_model = {'CAT': {}, 'KNN': {}, 'XGB':{}}
    # base_model = {'NN': {}, 'CAT': {}, 'GBM': {}, 'KNN':{}}
    # base_model = {'CAT': {}}
    base_model = {'NN': {}}
    # predictor = TabularPredictor.load(save_path) 
    
    # train_data = pd.read_pickle('middle.pkl')
    # train_data['class'] = y.iloc[train_data.index]
    predictor = TabularPredictor(label='class', path=save_path, learner_kwargs={'label_count_threshold':1}).fit(train_data, hyperparameters=base_model, num_bag_folds=args.num_parts, num_bag_sets=1, num_stack_levels=2)



    # prediction_mid = predictor.predict_proba(dataset)
    # prediction_mid = torch.from_numpy(np.array(prediction_mid))

    # print(compute_acc(prediction_mid[train_id], labels[train_id], evaluator))
    # print(compute_acc(prediction_mid[val_id], labels[val_id], evaluator))
    # print(compute_acc(prediction_mid[test_id], labels[test_id], evaluator))


    

    train_data_stacking = train_data
    test_data_stacking = test_data
    val_data_stacking = val_data
    base_models = predictor.get_model_names(level=1)


    for _ in range(1):

        y_train_stacker = []
        y_test_stacker = []
        y_val_stacker = []
        oof_stacker = []

        # for model in base_models:

        #     y_train = predictor.predict_proba(train_data_stacking, model)
        #     y_train_stacker.append(y_train)

        #     y_test = predictor.predict_proba(test_data_stacking, model)
        #     y_test_stacker.append(y_test)

        #     y_val = predictor.predict_proba(val_data_stacking, model)
        #     y_val_stacker.append(y_val)

        #     oof = predictor.get_oof_pred_proba(train_data=train_data, model=model)
        #     oof_stacker.append(oof)



        y_train = predictor.predict_proba(train_data_stacking)
        y_train_stacker.append(y_train)

        y_test = predictor.predict_proba(test_data_stacking)
        y_test_stacker.append(y_test)

        y_val = predictor.predict_proba(val_data_stacking)
        y_val_stacker.append(y_val)

        oof = predictor.get_oof_pred_proba(train_data=train_data)
        oof_stacker.append(oof)



        # base_models_index = ['{}_{}'.format(i, j) for i,j in product(base_models, range(oof.shape[1]))]

        y_train_stacker = pd.concat(y_train_stacker, axis=1)
        y_test_stacker = pd.concat(y_test_stacker, axis=1)
        y_val_stacker = pd.concat(y_val_stacker, axis=1)
        oof_stacker = pd.concat(oof_stacker, axis=1)


        y_train_stacker_pd = pd.DataFrame(np.asarray(y_train_stacker))
        y_train_stacker_pd.set_index(train_data.index, inplace=True)


        y_test_stacker_pd = pd.DataFrame(np.asarray(y_test_stacker))
        y_test_stacker_pd.set_index(test_data.index, inplace=True)


        y_val_stacker_pd = pd.DataFrame(np.asarray(y_val_stacker))
        y_val_stacker_pd.set_index(val_data.index, inplace=True)


        oof_stacker_pd = pd.DataFrame(np.asarray(oof_stacker))
        oof_stacker_pd.set_index(train_data.index, inplace=True)



        prediction_all = pd.concat([oof_stacker_pd, y_test_stacker_pd, y_val_stacker_pd], axis=0).sort_index()

        
        prediction_all = torch.from_numpy(np.array(prediction_all))


        prediction_all = propagation_DA.forward(graph, prediction_all, cat=True)

        # graph propagation 
        prediction_all = torch.cat(prediction_all, 1).cpu()






        ## new data after graph propagation 
        oof_stacker_pd = pd.DataFrame(np.asarray(prediction_all[train_id]))
        oof_stacker_pd.set_index(train_data.index, inplace=True)

        y_test_stacker_pd = pd.DataFrame(np.asarray(prediction_all[test_id]))
        y_test_stacker_pd.set_index(test_data.index, inplace=True)

        y_val_stacker_pd = pd.DataFrame(np.asarray(prediction_all[val_id]))
        y_val_stacker_pd.set_index(val_data.index, inplace=True)



        ## concat with original features
        train_data_stacking = pd.concat([oof_stacker_pd, train_data], axis=1)
        test_data_stacking = pd.concat([y_test_stacker_pd, test_data], axis=1)
        val_data_stacking = pd.concat([y_val_stacker_pd, val_data], axis=1)


        # pdb.set_trace()
        ## new dataset
        dataset_stacking = pd.concat([train_data_stacking, test_data_stacking, val_data_stacking], axis=0).sort_index()

        save_path = f'log/{args.datasets}/NN/phase_two'  # specifies folder to store trained models
        base_model = {'NN': {}}
        predictor = TabularPredictor(label='class', path=save_path, learner_kwargs={'label_count_threshold':1}).fit(train_data_stacking, hyperparameters=base_model, tuning_data=val_data_stacking)
        # predictor = TabularPredictor(label='class', path=save_path, learner_kwargs={'label_count_threshold':1}).fit(train_data_stacking, hyperparameters=base_model, num_bag_folds=args.num_parts, num_bag_sets=1, num_stack_levels=0)

        # predictor = TabularPredictor(label='class', path=save_path, groups='groups').fit(train_data_stacking, hyperparameters=base_model)


        # performance = predictor.evaluate(train_data_stacking)

        prediction_final = predictor.predict_proba(dataset_stacking)



        # graph propagation 
        prediction_final = torch.from_numpy(np.array(prediction_final))


        torch.save(prediction_final, f'save_results/{args.datasets}/bagging_step_{args.step}_rep_{args.rep}.pt')



        print(compute_acc(prediction_final[train_id], labels[train_id], evaluator))
        print(compute_acc(prediction_final[val_id], labels[val_id], evaluator))
        print(compute_acc(prediction_final[test_id], labels[test_id], evaluator))







 
if __name__ == "__main__":
    main()



