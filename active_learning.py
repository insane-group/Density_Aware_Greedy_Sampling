import math
import numpy  as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold

from al_methods import *
from logger import Logger
from real_datasets import *
from synthetic_datasets import *

class Active_Learning():

    def __init__(self, logger : Logger):
        self.logger     = logger
        self.log_prefix = "Active_Learning"

    def run_process(self, model : any, materials : pd.DataFrame, X_featureNames : list, Y_featureNames : list , designspace_thres : int, seedno : int, exp_num : int, method : str) -> pd.DataFrame:

        np.random.seed(seedno)

        # Get unique materials
        unique_materials = materials.type.unique()

        self.logger.info(self.log_prefix, "Setting up the K-Fold process.")

        fold_num = 10
        kfold = KFold(n_splits=fold_num, shuffle=True, random_state=seedno)
        inner_round = 0 
        
        material_names              = []
        material_target             = []
        mae_per_train_size          = {}
        # best_perform_materials          = {}

        i = 0
        self.logger.info(self.log_prefix, "Starting K-Fold process.")
        for train_material_indicies, left_out_material_indicies in kfold.split(unique_materials):

            np.random.seed(seedno)
            inner_round += 1

            self.logger.info(self.log_prefix, "Getting names of train and test materials.")
            train_material_names = np.delete(unique_materials, left_out_material_indicies)
            test_material_name   = unique_materials[left_out_material_indicies]

            self.logger.info(self.log_prefix, "Getting actual train and test materials.")
            train_materials = materials[~materials['type'].isin(test_material_name)]
            test_materials  = materials[materials['type'].isin(test_material_name)]
    
            
            current_data_points   = pd.DataFrame()

            self.logger.info(self.log_prefix, "Getting indicies of initial materials.")
            init_indicies         = np.load(f'./Init_Index/start_{exp_num}.npy')


            for index in init_indicies:

                self.logger.info(self.log_prefix,f"Adding initial material with index {index} in the training dataset.")
                init_material_name  = train_materials.iloc[index]["type"]
                selected_materials  = train_materials[(train_materials['type'] == init_material_name)]
                        
                material_names.append(init_material_name)
                material_target.append(selected_materials[Y_featureNames])
                
                if method != "rt":
                    train_materials      = train_materials[(train_materials['type']) != init_material_name]
                    train_material_names = np.delete(train_material_names, np.where(train_material_names == init_material_name))

                current_data_points  = pd.concat([current_data_points, selected_materials], axis=0, ignore_index=True)

            self.logger.info(self.log_prefix, "Training the model on the initial materials.")
            x_trainAll = current_data_points[X_featureNames].to_numpy()
            y_trainAll = current_data_points[Y_featureNames].to_numpy()  

            model.fit(x_trainAll, y_trainAll.ravel())  

            if method == "RT":
                # Set-up the class Regression_Tree for Active Learning
                self.logger.info(self.log_prefix, "Setting up the Regression Tree method.")
                RT = Regression_Tree(self.logger, seed=seedno,min_samples_leaf = 2)
                
                X_train_init = train_materials[X_featureNames].to_numpy()
                Y_train      = train_materials[Y_featureNames].to_numpy()

                #Input all samples (X_train_init), indices of the labeled MOFs (range(label_init) here) and labels of this set of MOFs (Y_train[:label_init])
                RT.input_data(X_train_init, init_indicies, y_trainAll.ravel())
        
                #Fit the initial regression tree
                self.logger.info(self.log_prefix, "Training Regression Tree model.")
                RT.fit_tree()

            self.logger.info(self.log_prefix, "Starting the Active Learning process.")
            for train_materials_num in range(designspace_thres - len(init_indicies)):

                if method == "DAGS":
                    dags  = Density_Aware_Greedy_Sampling(model,x_trainAll, y_trainAll,train_materials,X_featureNames,self.logger)
                    select_index = dags.get_next_index()

                elif method == "iGS":
                    iGS = Improved_Greedy_Sampling(model,x_trainAll, y_trainAll,train_materials,X_featureNames,self.logger)
                    select_index, dist= iGS.get_next_index()

                elif method == "QBC":
                    qbc = Query_By_Commite(model,x_trainAll, y_trainAll,train_materials,X_featureNames,self.logger)
                    select_index, dist = qbc.get_next_index()

                elif method == "Random":
                    rs = Random_Sampling(train_materials,self.logger)
                    select_index = rs.get_next_index()

                elif method == "RT":
                    #Compute the proportion of unlabeled MOFs and variance of true labels of MOFs in each leaf
                    RT.al_calculate_leaf_proportions() 
                    
                    #Select the new MOFs from each leaf based on the sample size
                    select_index = RT.pick_new_points(num_samples = 1)
                    select_index = select_index[0]
                    
                    #Label the newly selected MOFs
                    RT.label_point(select_index, Y_train[select_index][0])

                    #Fit the regression tree with the updated training set to continue the cycle
                    RT.fit_tree()

                else:
                    self.logger.error(self.log_prefix, "Error. Provided active learning method not recognised.")
                    print("Error active learning method not recognised.")
                    exit()
                
                self.logger.info(self.log_prefix, f"Adding material with index {select_index} in the training dataset.")

                selected_name = train_materials.iloc[select_index]["type"]
                material_names.append(selected_name)
                selected_materials = train_materials[(train_materials['type'] == selected_name)]
                material_target.append(selected_materials[Y_featureNames])
                
                # Remove the sellected material from the list of available for training
                if method != "RT":
                    train_materials = train_materials[(train_materials['type']) != selected_name]
                    train_material_names = np.delete(train_material_names, np.where(train_material_names == selected_name))
                
                current_data_points = pd.concat([current_data_points, selected_materials], axis=0, ignore_index=True)

                
                self.logger.info(self.log_prefix, "Prepare model training the model on the updated dataset.")
                # Create feature matrices for all currently used data.
                x_trainAll = current_data_points[X_featureNames].to_numpy()
                y_trainAll = current_data_points[Y_featureNames].to_numpy()

                uniqe_current_data = current_data_points.type.unique()
                trainLength = len(uniqe_current_data)

                # if (train_materials_num + 1) not in best_perform_materials.keys():
                #         best_perform_materials[(train_materials_num + 1)] = []                
                # best_perform_materials[(train_materials_num + 1)].append(selected_materials)

                if(trainLength>=5):

                    self.logger.info(self.log_prefix, "Training model with current data.")

                    # Train XGboost with the updated dataset 
                    model.fit(x_trainAll, y_trainAll.ravel())
                    
                    # Prediction on outer leave one out test data
                    x_test  = test_materials[X_featureNames].to_numpy()
                    y_test  = test_materials[Y_featureNames].to_numpy()
                    
                    y_pred = model.predict(x_test)
                    y_pred_train = model.predict(x_trainAll)
                    mae_train = metrics.mean_absolute_error(y_trainAll,y_pred_train)
                    
                    mae = metrics.mean_absolute_error(y_test, y_pred)
                    
                    print("Round",inner_round,"train size",train_materials_num+1,"mae is",mae,"train_mae is",mae_train)
  

                    if (train_materials_num + 1) not in mae_per_train_size.keys():
                        mae_per_train_size[(train_materials_num + 1)] = []
                          
                    # Append mae to the corresponding dictionary list
                    mae_per_train_size[(train_materials_num + 1)].append(mae)
                    
        print(mae_per_train_size.keys())    

        self.logger.info(self.log_prefix, "Writting results to a file.")
        result_df = pd.DataFrame()
        result_df["sizeOfTrainingSet"]       = np.array([iCnt for iCnt in sorted(mae_per_train_size.keys()) ])
        result_df["averageError"]            = [ np.array(mae_per_train_size[iCnt]).mean() for iCnt in mae_per_train_size.keys() ]
        result_df["stdErrorOfMeanError"]     = [ np.array(mae_per_train_size[iCnt]).std() / math.sqrt(iCnt) for iCnt in mae_per_train_size.keys() ]
        result_df["stdDeviationOfMeanError"] = [ np.array(mae_per_train_size[iCnt]).std()  for iCnt in mae_per_train_size.keys() ]
        
        return result_df, material_names, material_target

    def get_synthetic_dataset(self, dataset_name):

        if dataset_name == "forrester":
            self.logger.info(self.log_prefix, "Getting Forrester dataset.")
            forrester_dataset = Forrester_Dataset()
            return forrester_dataset.get_dataset()

        elif dataset_name == "forrester_imb":
            self.logger.info(self.log_prefix, "Getting Forrester imbalance dataset.")
            forrester_dataset = Forrester_Dataset(homogenous = False)
            return forrester_dataset.get_dataset()

        elif dataset_name == "jump_forrester":
            self.logger.info(self.log_prefix, "Getting Jump Forrester dataset.")
            jump_forr_dataset = Jump_Forrester_Dataset()
            return jump_forr_dataset.get_dataset()

        elif dataset_name == "jump_forrester_imb":
            self.logger.info(self.log_prefix, "Getting Jump Forrester imbalance dataset.")
            jump_forr_dataset = Jump_Forrester_Dataset(homogenous = False)
            return jump_forr_dataset.get_dataset()

        elif dataset_name == "gaussian":
            self.logger.info(self.log_prefix, "Getting Gaussian dataset.")
            gaussian_dataset = Gaussian_Dataset()
            return gaussian_dataset.get_dataset()

        elif dataset_name == "gaussian_imb":
            self.logger.info(self.log_prefix, "Getting Gaussian imbalance dataset.")
            gaussian_dataset = Gaussian_Dataset(homogenous = False)
            return gaussian_dataset.get_dataset()

        elif dataset_name == "exponential":
            self.logger.info(self.log_prefix, "Getting Exponential dataset.")
            exponential_dataset = Exponential_Dataset()
            return exponential_dataset.get_dataset()

        elif dataset_name == "exponential_imb":
            self.logger.info(self.log_prefix, "Getting Exponential imbalance dataset.")
            exponential_dataset = Exponential_Dataset(homogenous = False)
            return exponential_dataset.get_dataset()

        else:
            print("Unknwon synthetic dataset given.")
            exit()
    
    def get_real_dataset(self, dataset_name, source_file):

        if dataset_name == "zifs_diffusivity":
            self.logger.info(self.log_prefix, "Getting ZIFS diffusivity dataset.")
            return data_preparation(research_data="zifs_diffusivity")

        elif dataset_name == "co2":
            self.logger.info(self.log_prefix, "Getting CO2 dataset.")
            return data_preparation(sourceFile=source_file, research_data="co2")

        elif dataset_name == "co2/n2":
            self.logger.info(self.log_prefix, "Getting CO2/N2 dataset.")
            return data_preparation(sourceFile=source_file, research_data="co2/n2")

        elif dataset_name == "o2":
            self.logger.info(self.log_prefix, "Getting O2 dataset.")
            return data_preparation(sourceFile=source_file, research_data="o2")

        elif dataset_name == "n2":
            self.logger.info(self.log_prefix, "Getting N2 dataset.")
            return data_preparation(sourceFile=source_file, research_data="n2")

        elif dataset_name == "ch4":
            self.logger.info(self.log_prefix, "Getting CH4 dataset.")
            return data_preparation(sourceFile=source_file, research_data="ch4")

        elif dataset_name == "h2":
            self.logger.info(self.log_prefix, "Getting H2 dataset.")
            return data_preparation(sourceFile=source_file, research_data="h2")

        elif dataset_name == "he":
            self.logger.info(self.log_prefix, "Getting He dataset.")
            return data_preparation(sourceFile=source_file, research_data="he")

        elif dataset_name == "methane":
            self.logger.info(self.log_prefix, "Getting methane dataset.")
            return data_preparation(sourceFile=source_file, research_data="methane")
        else:
            self.logger.error(self.log_prefix, "Unknown dataset given.")
            print("Unknown real dataset given.")
            exit()
