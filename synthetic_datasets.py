import itertools
import numpy as np
import pandas as pd

class Synthetic_Datasets():
    
    def __init__(self):
        self.dataset_name = "generic_dataset"

    def get_dataset(self):
        
        data = pd.DataFrame()
        featureNames = ['x']
        targetNames = ['y']

        return data, featureNames, targetNames

    def get_dataset_name(self):
        return self.dataset_name


class Forrester_Dataset(Synthetic_Datasets):
    
    def __init__(self, homogenous = True):
        self.dataset_name = "forrester_dataset"
        self.homogeneity = homogenous

    def get_dataset(self):
    
        np.random.seed(10)

        if(self.homogeneity):
            # Create 1000 evenly spaced samples.
            x = np.linspace(0, 1, 1000)    

        else:
            # Generate a large number of x values in the range [0.6, 0.9) and ensure uniqueness
            x_60 = np.random.choice(np.linspace(0.6, 0.9, 10000), size=600, replace=False)

            # Generate a large number of x values in the range [0, 0.6) and ensure uniqueness
            x_20_1 = np.random.choice(np.linspace(0, 0.6, 10000), size=200, replace=False)

            # Generate a large number of x values in the range [0.9, 1] and ensure uniqueness
            x_20_2 = np.random.choice(np.linspace(0.9, 1, 10000), size=200, replace=False)

            # Combine the x values
            x = np.concatenate((x_60, x_20_1, x_20_2))

        
        # Sort x values 
        x = np.sort(x)

        # Calculate the corresponding y values using the given formula
        y = (6 * x - 2) ** 2 * np.sin(12 * x - 4)

        # Create unique names for each sample
        types = [f"Sample_{i+1}" for i in range(1000)]

        # Create a DataFrame with columns x, y, and type
        data = pd.DataFrame({'x': x, 'y': y, 'type': types})

        featureNames = ['x']
        targetNames = ['y']
        
        return data, featureNames, targetNames

class Jump_Forrester_Dataset(Synthetic_Datasets):
    
    def __init__(self, homogenous = True):
        self.dataset_name = "jump_forrester_dataset"
        self.homogeneity = homogenous
    
    def get_dataset(self):
        np.random.seed(10)

        if self.homogeneity:
            # Create 1000 evenly spaced samples with a jump on 500th.
            x1 = np.linspace(0, 0.5, 500)
            x2 = np.linspace(0.5, 1, 500)
        
        else:  

            x1 = np.random.choice(np.linspace(0, 0.5, 10000), size=300, replace=False)
            x2 = np.random.choice(np.linspace(0.5, 1, 10000), size=700, replace=False)
        
        x = np.concatenate((x1,x2))


        y1 = (6 * x1 - 2) ** 2 * np.sin(12 * x1 - 4)
        y2 = (6 * x2 - 2) ** 2 * np.sin(12 * x2 - 4) + 10
        y = np.concatenate((y1,y2))

        # Create unique names for each sample
        types = [f"Sample_{i+1}" for i in range(1000)]

        # Create a DataFrame with columns x, y, and type
        data = pd.DataFrame({'x': x, 'y': y, 'type': types})

        featureNames = ['x']
        targetNames = ['y']

        return data, featureNames, targetNames
    

class Gaussian_Dataset(Synthetic_Datasets):
    
    def __init__(self, homogenous = True):
        self.dataset_name = "gaussian_dataset"
        self.homogeneity = homogenous
    
    def get_gaussian_2d(self,x1, x2, sigma=1):
        return np.exp(-(x1**2 + x2**2) / (2 * sigma**2))
        

    def get_dataset(self):

        np.random.seed(10)

        selected_pairs = None

        if self.homogeneity:
            # Generate linspace for x1 and x2
            x1 = np.linspace(-3, 3, 100)
            x2 = np.linspace(-3, 3, 100)

            # Create all possible unique pairs using itertools.product
            all_pairs = np.array(list(itertools.product(x1, x2)))

            # Randomly select 2000 unique pairs
            selected_indices = np.random.choice(len(all_pairs), size=2000, replace=False)
            selected_pairs = all_pairs[selected_indices]

        else:
            # Generate linspace for x1 and x2
            x1_1 = np.linspace(-0.5, 0.5, 100)
            x2_1 = np.linspace(-0.5, 0.5, 100)

            # Create all possible unique pairs using itertools.product
            all_pairs1 = np.array(list(itertools.product(x1_1, x2_1)))

            # Randomly select 2000 unique pairs
            selected_indices = np.random.choice(len(all_pairs1), size=500, replace=False)
            selected_pairs1 = all_pairs1[selected_indices]

            # Generate linspace for x1 and x2
            x1_2 = np.linspace(-3, 3, 100)
            x2_2 = np.linspace(-3, 3, 100)

            # Create all possible unique pairs using itertools.product
            all_pairs = np.array(list(itertools.product(x1_2, x2_2)))

            # Find all pairs that do not belong in the central area
            outside_range_pairs = all_pairs[~((all_pairs[:, 0] >= -0.5) & (all_pairs[:, 0] <= 0.5) & 
                                            (all_pairs[:, 1] >= -0.5) & (all_pairs[:, 1] <= 0.5))]


            selected_outside_indices = np.random.choice(len(outside_range_pairs), size=1500, replace=False)
            selected_outside_pairs = outside_range_pairs[selected_outside_indices]


            selected_pairs = np.vstack([selected_pairs1, selected_outside_pairs])

        
        # Calculate y for each pair
        y = np.array([self.get_gaussian_2d(pair[0], pair[1]) for pair in selected_pairs])

        # Combine x1, x2, and y into a DataFrame
        data = pd.DataFrame(selected_pairs, columns=['x1', 'x2'])
        data['y'] = y

        types = [f"Sample_{i+1}" for i in range(2000)]
        data['type'] = types

        featureNames = ['x1','x2']
        targetNames = ['y']

        return data, featureNames, targetNames


class Exponential_Dataset(Synthetic_Datasets):

    def __init__(self, homogenous = True):
        self.dataset_name = "exponential_dataset"
        self.homogeneity = homogenous
    
    def get_exponential_y(self, x1, x2):
        return 1 - np.exp(-0.6 * ((x1 - 0.5)**2 + (x2 - 0.5)**2))

    def get_dataset(self):

        np.random.seed(10)

        selected_pairs = None
        if self.homogeneity:

            x1 = np.linspace(0, 1, 100)
            x2 = np.linspace(0, 1, 100)

            # Create all possible unique pairs using itertools.product
            all_pairs = np.array(list(itertools.product(x1, x2)))

            # Randomly select 2000 unique pairs
            selected_indices = np.random.choice(len(all_pairs), size=2000, replace=False)
            selected_pairs = all_pairs[selected_indices]

        else:
            # Generate linspace for x1 and x2
            x1_1 = np.linspace(0.2, 0.8, 100)
            x2_1 = np.linspace(0.2, 0.8, 100)

            # Create all possible unique pairs using itertools.product
            all_pairs_1 = np.array(list(itertools.product(x1_1, x2_1)))

            # Randomly select 2000 unique pairs
            selected_indices = np.random.choice(len(all_pairs_1), size=500, replace=False)
            selected_pairs_1 = all_pairs_1[selected_indices]

            # Generate linspace for x1 and x2
            x1_2 = np.linspace(0, 1, 100)
            x2_2 = np.linspace(0, 1, 100)

            all_pairs = np.array(list(itertools.product(x1_2, x2_2)))

            outside_range_pairs = all_pairs[~((all_pairs[:, 0] >= 0.2) & (all_pairs[:, 0] <= 0.8) & 
                                            (all_pairs[:, 1] >= 0.2) & (all_pairs[:, 1] <= 0.8))]


            selected_outside_indices = np.random.choice(len(outside_range_pairs), size=1500, replace=False)
            selected_outside_pairs = outside_range_pairs[selected_outside_indices]


            selected_pairs = np.vstack([selected_pairs_1, selected_outside_pairs])


        # Calculate y for each pair
        y = np.array([self.get_exponential_y(pair[0], pair[1]) for pair in selected_pairs])

        types = [f"Sample_{i+1}" for i in range(2000)]

        # Combine x1, x2, and y into a DataFrame
        np_data = pd.DataFrame(selected_pairs, columns=['x1', 'x2'])
        np_data['y'] = y
        np_data['type'] = types

        featureNames = ['x1','x2']
        targetNames = ['y']

        return np_data, featureNames, targetNames
