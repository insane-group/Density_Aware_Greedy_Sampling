import copy
import numpy as np
import var as var
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

from logger import Logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

class AL_Methods():

    def __init__(self, logger : Logger):
        self.logger      = logger
        self.log_prefix  = "AL_Methods"
        self.method_name = "generic_method"
    
    def get_next_index(self):
        return 0

    def get_method_name(self):
        return self.method_name

class Query_By_Commite(AL_Methods):
    
    def __init__(self,model,X_train,y_train,X_unlabeled,featureNames, logger : Logger):
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_unlabeled  = X_unlabeled
        self.featureNames = featureNames
        self.logger       = logger
        self.log_prefix   = "Query_By_Commite"
        self.method_name  = "qbc"

    def get_next_index(self):
        self.logger.info(self.log_prefix, "Using Query By Commitee selection method.")
        x_unlabeled = self.X_unlabeled[self.featureNames].to_numpy()
        
        # Train a GP model and get predictions
        gp_model = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0, alpha=1.5))
        gp_model.fit(self.X_train,self.y_train.ravel())
        y_pred1, y_std = gp_model.predict(x_unlabeled,return_std=True)

        # Train a Random forrest Regressor and get predictions
        forrest_model = RandomForestRegressor(max_depth=5, random_state=0)
        forrest_model.fit(self.X_train,self.y_train.ravel())
        y_pred2 = forrest_model.predict(x_unlabeled)

        # Get predictions from the currently used model.
        y_pred3 = self.model.predict(x_unlabeled)

        stacked_ypreds = np.column_stack((y_pred1, y_pred2, y_pred3))
        
        # Calculate the standard deviation along the first axis (row-wise)
        # Query the point with maximum variance max uncertainty
        res = np.var(stacked_ypreds, axis=1)

        index = np.argmax(res)
        
        self.logger.info(self.log_prefix, "Selected index: " + str(index))

        return index, res

    def get_model(self):
        return self.model
    
    def get_X_train(self):
        return self.X_train
    
    def get_y_train(self):
        return self.y_train
    
    def get_X_unlabeled(self):
        return self.X_unlabeled
    
    def get_featureNames(self):
        return self.featureNames


class Improved_Greedy_Sampling(AL_Methods):
    
    def __init__(self,model,X_train,y_train,X_unlabeled,featureNames, logger : Logger):
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_unlabeled  = X_unlabeled
        self.featureNames = featureNames
        self.logger       = logger
        self.log_prefix   = "Improved_Greedy_Sampling"
        self.method_name  = "iGS"
    
    def get_next_index(self):
        self.logger.info(self.log_prefix, "Using Improved Greedy Sampling selection method.")
        res_x,dist_x = self.Gsx()
        res_y,dist_y = self.Gsy()

        dist  = dist_x * dist_y
        index = np.argmax(dist)

        self.logger.info(self.log_prefix, "Selected index: " + str(index))

        return index, dist

    def Gsx(self):
        
        x_unlabeled = self.X_unlabeled[self.featureNames].to_numpy()
        
        # Compute the pairwise Euclidean distances between each unlabeled sample and all training samples
        distances = cdist(x_unlabeled, self.X_train, metric='euclidean')
        
        # Find the minimum distance and the corresponding index
        min_distances = np.min(distances, axis=1)
        index         = np.argmax(min_distances)
        
        return  index, min_distances

    def Gsy(self):

        x_unlabeled = self.X_unlabeled[self.featureNames].to_numpy()        
        y_pred  = self.model.predict(x_unlabeled)

        y_train = self.y_train.ravel()

        min_distances = []
        for pred in y_pred:
            
            distances = np.sqrt((y_train - pred) ** 2)
            min_distances.append(np.min(distances))
        
        min_distances = np.array(min_distances)
        index         = np.argmax(min_distances)

        return index, min_distances

    def get_model(self):
        return self.model
    
    def get_X_train(self):
        return self.X_train
    
    def get_y_train(self):
        return self.y_train
    
    def get_X_unlabeled(self):
        return self.X_unlabeled
    
    def get_featureNames(self):
        return self.featureNames


class Density_Aware_Greedy_Sampling(AL_Methods):
    def __init__(self,model,X_train,y_train,X_unlabeled,featureNames, logger : Logger):
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_unlabeled  = X_unlabeled
        self.featureNames = featureNames
        self.logger       = logger
        self.log_prefix   = "Density_Aware_Greedy_Sampling"
        self.method_name  = "dags"
    
    def get_next_index(self):
        self.logger.info(self.log_prefix, "Using Density Aware Greedy Sampling selection method.")
        
        x_unlabeled = self.X_unlabeled[self.featureNames].to_numpy()

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x_unlabeled)
        distances, indices = nbrs.kneighbors(x_unlabeled)
        density = 1 / distances.mean(axis=1)

        # Run the iGS method to produce uncertainties 
        iGS = Improved_Greedy_Sampling(self.model, self.X_train, self.y_train, self.X_unlabeled, self.featureNames,self.logger)
        index, uncertainty = iGS.get_next_index()
        
        result_index = np.argmax(uncertainty * density)

        self.logger.info(self.log_prefix, "Selected index: " + str(result_index))
        return result_index

    def get_model(self):
        return self.model
    
    def get_X_train(self):
        return self.X_train
    
    def get_y_train(self):
        return self.y_train
    
    def get_X_unlabeled(self):
        return self.X_unlabeled
    
    def get_featureNames(self):
        return self.featureNames

class Regression_Tree(AL_Methods):
        
    def __init__(self,logger : Logger, min_samples_leaf=None, seed=None):

        self.points          = None
        self.labels          = None
        self.labeled_indices = None
        self._num_points     = 0
        self._num_labeled    = 0
        self.logger          = logger
        self.log_prefix      = "Regression_Tree"

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        if min_samples_leaf is None:
            self.min_samples_leaf=1
        else:
            self.min_samples_leaf=min_samples_leaf

        self.tree = DecisionTreeRegressor(random_state=self.seed,min_samples_leaf=self.min_samples_leaf)
        self._leaf_indices = []
        self._leaf_marginal = []
        self._leaf_var = []
        self._al_proportions =[]

        self._leaf_statistics_up_to_date = False
        self._leaf_proportions_up_to_date = False

        self._verbose = False

    ''' Input all features (all_data), indices (labeled_indices) and labels of the points that are labeled '''
    
    def input_data(self, all_data, labeled_indices, labels, copy_data=True):
    
        if copy_data:
            all_data = copy.deepcopy(all_data)
            labeled_indices = copy.deepcopy(labeled_indices)
            labels = copy.deepcopy(labels)

        if len(all_data) < len(labeled_indices):
            raise ValueError('Cannot have more labeled indicies than points')

        if len(labeled_indices) != len(labels):
            raise ValueError('Labeled indicies list and labels list must be same length')

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labeled_indices)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labeled_indices to list internally')
            labeled_indices = labeled_indices.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)  
        self._num_labeled = len(labels)     

        ''' Making a label list, with None in for unlabeled points '''

        temp = [None] * self._num_points
        for i,ind in enumerate(labeled_indices):
            temp[ind] = labels[i]
        self.labels = temp
        self.labeled_indices = list(labeled_indices)

    ''' Fitting a regression tree with labeled points '''
    
    def fit_tree(self):
        self.tree.fit(np.array(self.points)[self.labeled_indices,:], 
            np.array(self.labels)[self.labeled_indices])
        self._leaf_indices = self.tree.apply(np.array(self.points)) #returns index of leaf for each point, labeled and unlabeled
        self._leaf_statistics_up_to_date = False
        
    def get_depth(self):
        return(self.tree.get_n_leaves())

    ''' Label the unlabeled point selected from the pool for labeling '''
    
    def label_point(self, index, value):

        if self.labels is None:
            raise RuntimeError('No data in the tree')

        if len(self.labels) <= index:
            raise ValueError('Index {} larger than size of data in tree'.format(index))

        value = copy.copy(value)
        index = copy.copy(index)

        self.labels[index] = value
        self.labeled_indices.append(index)
        self._num_labeled += 1

    def predict(self, new_points):
        return(self.tree.predict(new_points))
        
    ''' Compute the variance of true labels and the proportion of unlabeled samples in each leaf '''
    
    def calculate_leaf_statistics(self):
        temp = Counter(self._leaf_indices)                            #the number of points in different leaves
        self._leaf_marginal = []
        self._leaf_var = []
        for key in np.unique(self._leaf_indices):
            self._leaf_marginal.append(temp[key]/self._num_points)    #proportion of unlabeled points in each leaf
            temp_ind = [i for i,x in enumerate(self._leaf_indices) if x == key]
            temp_labels = [x for i,x in enumerate(self.labels) if x is not None and self._leaf_indices[i]==key]

            self._leaf_var.append(self.unbiased_variance(temp_labels))    #variance of true labels in each leaf
        self._leaf_statistics_up_to_date = True
        
    ''' Calculation of n_k*, for each leaf k '''
    
    def al_calculate_leaf_proportions(self):
        if not self._leaf_statistics_up_to_date:
            self.calculate_leaf_statistics()
        al_proportions = []
        for i, val in enumerate(self._leaf_var):
            al_proportions.append(np.sqrt(self._leaf_var[i] * self._leaf_marginal[i]))

        al_proportions = np.array(al_proportions)/sum(al_proportions)
        self._al_proportions = al_proportions
        self._leaf_proportions_up_to_date = True
        
    ''' Select the point to be labeled, based on n_k* '''
    
    def pick_new_points(self, num_samples = 1): 
        self.logger.info(self.log_prefix, "Using Regression Tree selection method.")
        if not self._leaf_proportions_up_to_date:
            self.al_calculate_leaf_proportions()
        temp = Counter(np.array(self._leaf_indices)[[x for x in range(self._num_points) if self.labels[x] is None]])
        point_proportions = {}

        for i,key in enumerate(np.unique(self._leaf_indices)):
            point_proportions[key] = self._al_proportions[i] / max(1,temp[key]) 

        temp_probs = np.array([point_proportions[key] for key in self._leaf_indices])
        temp_probs[self.labeled_indices] = 0
        temp_probs = temp_probs / sum(temp_probs)
        
        if 'NaN' in temp_probs:
            return(temp,temp_probs,sum(temp_probs))
        leaves_to_sample = np.random.choice(self._leaf_indices,num_samples, 
            p=temp_probs, replace = False)                     #leaves to be sampled from have been selected
        
        
        points_to_label = []
        for leaf in np.unique(leaves_to_sample):
            points = []
            for j in range(Counter(leaves_to_sample)[leaf]):
                possible_points = np.setdiff1d([x for i,x in enumerate(range(self._num_points)
                    ) if self._leaf_indices[i] ==leaf and self.labels[i] is None ], points)                                              
                point_to_label = np.random.choice(possible_points)
                points_to_label.append(point_to_label)
                points.append(point_to_label)  

        return(points_to_label)
    
    def unbiased_variance(self, labels):

        n = len(labels)

        if n < 2:
            return 0

        mean = sum(labels)/n

        tot = 0
        for val in labels:
            tot += (mean - val)**2

        return tot/(n-1)
    

class Random_Sampling(AL_Methods):
    
    def __init__(self,X_unlabeled , logger : Logger):
        self.logger       =  logger
        self.x_unlabeled  =  X_unlabeled
        self.log_prefix   =  "Random_Sampling"
        self.method_name  =  "random"
    
    def get_next_index(self):

        self.logger.info(self.log_prefix, "Using Random Sampling selection method.")
        return np.random.choice(np.arange(len(self.x_unlabeled["type"].to_numpy())))