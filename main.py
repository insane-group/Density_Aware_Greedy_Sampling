import os
from datetime import datetime
from xgboost import XGBRegressor

import logging
from logger import Logger 
from active_learning import Active_Learning

if __name__ == "__main__":

    logger = Logger(name = 'dags-al_logger', level=logging.DEBUG, output="filestream",
                        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    #real_dataset_names = ["zifs_diffusivity", "co2", "co2/n2", "o2", "n2", "ch4", "h2", "he", "methane"]
    synthetic_dataset_names = ["Forrester", "forrester_imb", "jump_forrester", "jump_forrester_imb", "gaussian", "gaussian_imb", "gaussian_imb_noise", "exponential", "exponential_imb"]

    al_selection_methods = ["density", "igs", "qbc", "rt", "random"]

    al_process = Active_Learning(logger)
    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        random_state=6410)

    for dataset in synthetic_dataset_names:
        for method in al_selection_methods:
            for i in range(10):
                file_name = dataset + "_150_" + method + "_" + str(i) + ".csv"
                save_path = "./ALresults/Synthetic/" + method + "/" + dataset
                file_path = os.path.join(save_path, file_name)

                # data, feature_names, target_names = al_process.get_real_dataset("zifs_diffusivity")

                data, feature_names, target_names = al_process.get_synthetic_dataset(dataset)

                results, names, targets = al_process.run_process(XGBR, data, feature_names, target_names, designspace_thres=150, seedno=(i+1)*10 , exp_num=i, method=method)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                results.to_csv(file_path, index=False)
