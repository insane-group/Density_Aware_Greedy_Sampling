import os
import argparse
from datetime import datetime
from xgboost import XGBRegressor

import logging
from logger import Logger 
from active_learning import Active_Learning

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',   '--method' ,            help='Define the selection method as on of the following: dags, igs, qbc, rt, random, all.',      default="all")
    parser.add_argument('-d',   '--dataset',            help="""Select the dataset to use.
                                                                For synthetic datasets you can select: forrester, forrester_imb,
                                                                                                       jump_forrester, jump_forrester_imb, 
                                                                                                       gaussian, gaussian_imb, 
                                                                                                       exponential, exponential_imb, or all_synthetic. \n
                                                                For real datasets you can select: zifs_diffusivity, co2, co2/n2, o2, n2, ch4, h2, he, methane or all_real.
                                                             """,                                                                                         default="all_synthetic")
    parser.add_argument('-s',   '--save',               help='Define the path to save the results.',                                                      default="./ALresults")
    parser.add_argument('-i',   '--iter',               help='Define the number of experiments to perform.',                                              default=10)
    
    parsed_args = parser.parse_args()

    selection_method  = parsed_args.method
    dataset           = parsed_args.dataset
    save_path         = parsed_args.save
    iterations        = parsed_args.iter


    logger = Logger(name = 'dags-al_logger', level=logging.DEBUG, output="filestream",
                        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    real_dataset_names      = ["zifs_diffusivity", "co2", "co2/n2", "o2", "n2", "ch4", "h2", "he", "methane"]
    synthetic_dataset_names = ["forrester", "forrester_imb", "jump_forrester", "jump_forrester_imb", "gaussian", "gaussian_imb", "gaussian_imb_noise", "exponential", "exponential_imb"]

    if dataset == "all_real":
        save_path         = os.path.join(save_path, "Real")
        selected_datasets = real_dataset_names
    
    elif dataset == "all_synthetic":
        save_path         = os.path.join(save_path, "Synthetic")
        selected_datasets = synthetic_dataset_names

    else:
        dataset = [dataset]

    selection_methods_map = {"dags"  :  "DAGS",
                             "igs"   :  "iGS",
                             "qbc"   :  "QBC",
                             "rt"    :  "RT",
                             "random":  "Random"}

    if selection_method == "all":
        al_selection_methods = list(selection_methods_map.values())

    else:
        al_selection_methods = [selection_methods_map[selection_method]]

    al_process = Active_Learning(logger)
    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        random_state=6410)

    for dataset in selected_datasets:
        for method in al_selection_methods:
            for i in range(iterations):
                
                file_name      = dataset + "_150_" + method + "_" + str(i) + ".csv"
                curr_save_path = os.path.join(save_path, method, dataset)
                file_path      = os.path.join(curr_save_path, file_name)

                if dataset in real_dataset_names:
                    data, feature_names, target_names = al_process.get_real_dataset(dataset)
                
                else:
                    data, feature_names, target_names = al_process.get_synthetic_dataset(dataset)

                results, names, targets = al_process.run_process(XGBR, data, feature_names, target_names, designspace_thres=150, seedno=(i+1)*10 , exp_num=i, method=method)

                if not os.path.exists(curr_save_path):
                    os.makedirs(curr_save_path)

                results.to_csv(file_path, index=False)
