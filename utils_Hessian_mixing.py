import os
import numpy as np
import pandas as pd 
import urllib 
import scipy
from sklearn.datasets import fetch_openml

# Dataset-dependent suptitle
dataset_titles = {
    'autompg'           : "Autompg"             ,
    'energy'            : "Energy"              ,
    'elevators'         : "Elevators"           ,
    'concrete'          : "Concrete"            ,
    'airfoil'           : "Airfoil"             ,
    'breastcancer'      : "Breast Cancer"       ,
    'parkinsons'        : "Parkinsons"          ,
    'sml'               : "SML"                 ,
    'keggundirected'    : "KEGG (Undirected)"   ,
    'challenger'        : "Challenger"          ,
    'protein'           : "Protein"             ,
    'crime'             : "Communities & Crime" ,
    'housing'           : "Boston Housing"      ,
    'bike'              : "Bike Sharing"        ,
    'wine'              : "Wine"                ,
    'tecator'           : "Tecator"             ,
    'Gas'               : "Gas"                 ,
    'Concrete'          : "Concrete"            ,
    'tamielectric'      : "Tami Electric"       ,
    'keggdirected'      : "KEGG (Directed)"     ,
    'yacht'             : "Yacht"               ,
    'solar'             : "Solar"               ,
    '3droad'            : "3D Road"             ,
    'slice'             : "Slice"               ,
    'servo'             : "Servo"               ,
    'autos'             : "Autos"               ,
    'concreteslump'     : "Concrete Slump"      ,
    'fertility'         : "Fertility"           ,
    'forest'            : "Forest"              ,
    'houseelectric'     : "House Electric"      ,
    'kin40k'            : "Kin40K"              ,
    'machine'           : "Machine"             ,
    'pol'               : "Pol"                 ,
    'pendulum'          : "Pendulum"            ,
    'pumadyn32nm'       : "Pumadyn32nm"         ,
    'buzz'              : "Buzz"                
}
    
################# Analytic Gaussian Mechanism #################
"""
    Taken from the official repository of Balle and Wang, ICML'18 
    https://github.com/BorjaBalle/analytic-gaussian-mechanism
"""

from math import exp, sqrt
from scipy.special import erf

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

################# Analytic solver for the noise required for the GaussMix mechanism #################
"""
    Correspond to the different calculations done in: 
"""

# Find noise required for Gaussian mechanism with composition over T steps 
# by performing the exact conversion from RDP to DP 
def objective_Gaussian_composition(alpha, sigma, delta,C_max, T):
    if alpha <= 1:
        return np.inf  # Penalize out-of-bound values

    term1 = alpha * (C_max**2) / (2.0 * sigma**2)
    term2 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    return T * term1 + term2

def objective_func_full_composition(alpha, k, gamma, delta, inflation_norm, T):
    if alpha <= 1 or alpha >= gamma/(1 + inflation_norm):
        return np.inf  # Penalize out-of-bound values

    term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + inflation_norm) / (gamma))
    term2 = - (k / (2 * (alpha - 1))) * np.log(1 - (alpha*(1 + inflation_norm)) / gamma)
    term3 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    term4 = np.sqrt(2.0 * np.log(1.25/delta)) / (gamma/np.sqrt(k))
    return T * (term1 + term2) + term3 + term4

def objective_func_full(alpha, k, gamma, delta, inflation_norm):
    if alpha <= 1 or alpha >= gamma/(1+inflation_norm):
        return np.inf  # Penalize out-of-bound values

    term1 = (k * alpha) / (2 * (alpha - 1)) * np.log(1.0 - (1.0 + inflation_norm) / (gamma))
    term2 = - (k / (2 * (alpha - 1))) * np.log(1 - (alpha*(1 + inflation_norm)) / gamma)
    term3 = (np.log(1.0 / delta) + (alpha - 1)*np.log(1-1/alpha) - np.log(alpha)) / (alpha - 1)
    term4 = np.sqrt(2.0 * np.log(1.25/delta)) / (gamma/np.sqrt(k))
    return term1 + term2 + term3 + term4

def solve_gamma_renyi_full_composition(init_gamma, k, target_delta, target_epsilon, inflation_norm, T):   
    # Solve for the \gamma required for target \eps,\delta DP after composing T steps 
    # Define binary search bounds
    left, right = init_gamma / 500000.0, 500000.0*init_gamma
    best_gamma = right  # Default to upper bound in case no solution is found
    
    # The bound is for composition over 3 contributions for \delta.
    # Thus, we divide \delta by 3 
    internal_target_delta = target_delta/3.0
    while right - left > 1e-6:  # Precision threshold
        mid_gamma = (left + right) / 2
        
        # Solve for optimal alpha given the current gamma
        result = scipy.optimize.minimize_scalar(objective_func_full_composition, 
                                 bounds=(1 + 1e-5, mid_gamma - 1e-5), 
                                 args=(k, mid_gamma, internal_target_delta, inflation_norm, T), 
                                 method='bounded') # minimization is between 1 < \alpha < \gamma 
        
        if result.success and result.fun < target_epsilon:
            best_gamma = mid_gamma  # Update best found gamma
            right = mid_gamma  # Search for a smaller gamma
        else:
            left = mid_gamma  # Increase gamma to meet target_epsilon
    return best_gamma

# Solve the required noise for single sketching step by 
# performing the full conversion from RenyiDP to DP numerically
def solve_gamma_renyi_full(init_gamma, k, target_delta, target_epsilon, inflation_norm):   
    # Define binary search bounds
    left, right = init_gamma / 500000.0, 500000.0*init_gamma
    best_gamma = right  # Default to upper bound in case no solution is found
    
    # The bound is for composition over 3 contributions for \delta. Thus, we divide \delta by 3 
    internal_target_delta = target_delta/3.0
    while right - left > 1e-6:  # Precision threshold
        mid_gamma = (left + right) / 2
        # Solve for optimal alpha given the current gamma
        result = scipy.optimize.minimize_scalar(objective_func_full, 
                                 bounds=(1 + 1e-5, mid_gamma - 1e-5), 
                                 args=(k, mid_gamma, internal_target_delta, inflation_norm), 
                                 method='bounded') # minimization is between 1 < \alpha < \gamma 
        if result.success and result.fun < target_epsilon:
            best_gamma = mid_gamma  # Update best found gamma
            right = mid_gamma  # Search for a smaller gamma
        else:
            left = mid_gamma  # Increase gamma to meet target_epsilon
    return best_gamma

# Solve with full conversion from RenyiDP to DP for the Gaussian mechanism
def solve_Gaussian_composition(init_gamma, delta, target_epsilon, C_max, T):   
    # Solve for the \gamma required for target \eps,\delta DP after composing T steps 
    # Define binary search bounds
    left, right = init_gamma / 500000.0, 500000.0*init_gamma
    best_gamma = right  # Default to upper bound in case no solution is found
    while right - left > 1e-6:  # Precision threshold
        mid_gamma = (left + right) / 2
        
        # Solve for optimal alpha given the current gamma
        result = scipy.optimize.minimize_scalar(objective_Gaussian_composition, 
                                 bounds=(1 + 1e-5, 1e6), 
                                 args=(mid_gamma, delta,C_max, T), 
                                 method='bounded') # effectively unbounded by the upper limit being very large 
        
        if result.success and result.fun < target_epsilon:
            best_gamma = mid_gamma  # Update best found gamma
            right = mid_gamma  # Search for a smaller gamma
        else:
            left = mid_gamma  # Increase gamma to meet target_epsilon
    return best_gamma

# Load crimes dataset
def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', env_partition=0.05):
    
    # Directory of this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to "datasets" folder (same directory as this file)
    DATA_DIR = os.path.join(BASE_DIR, "datasets")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Full paths to the files
    data_path = os.path.join(DATA_DIR, "communities.data")
    names_path = os.path.join(DATA_DIR, "communities.names")

    if not os.path.isfile(data_path):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "communities.names")
    # create names
    names = []
    with open(names_path, 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])
    # load data
    data = pd.read_csv(data_path, names=names, na_values=['?'])
    data.drop(['state', 'county', 'community', 'fold', 'communityname'], axis=1, inplace=True)
    data = data.replace('?', np.nan)
    data['OtherPerCap'] = data['OtherPerCap'].fillna(data['OtherPerCap'].astype(float).mean())
    data = data.dropna(axis=1)
    data['OtherPerCap'] = data['OtherPerCap'].astype(float)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)
    to_drop = []
    y = data[label].values
    to_drop += [label]
    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]
    data.drop(to_drop + [label], axis=1, inplace=True)
    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()
    x = np.array(data.values)
    x = x[z >= env_partition]
    y = y[z >= env_partition]
    z = z[z >= env_partition]
    
    # Apply random permutation 
    m, n = x.shape
    p = np.random.permutation(m)
    x = x[p, :]
    y = y[p]
    
    # Split into training and testing sets
    train_size = int(0.8 * len(y))

    X_train = x[:train_size]
    y_train = y[:train_size]
    X_test = x[train_size:]
    y_test = y[train_size:]
    train_size = int(0.8 * len(y))

    norm_fact_y = np.max(np.abs(y_train))
    y_train /= norm_fact_y
    y_test  /= norm_fact_y
        
    return X_train,X_test,z,z,y_train,y_test

# Load datasets 
def GetDataset(dataset_name):
    
    # current directory of datasets files 
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, "datasets")
    
    # Boston housing 
    if dataset_name == 'housing':

        data = np.loadtxt(os.path.join(dataset_path, 'housing.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0

    # Elevators 
    elif dataset_name == 'elevators':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'elevators_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'elevators_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # sml 
    elif dataset_name == 'sml':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'sml_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'sml_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # protein 
    elif dataset_name == 'protein':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'protein_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'protein_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # keggle undirected 
    elif dataset_name == 'keggundirected':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggundirected_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggundirected_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # keggle directed 
    elif dataset_name == 'keggdirected':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggdirected_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'keggdirected_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0

    # 3droad
    elif dataset_name == '3droad':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, '3droad_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, '3droad_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # slice
    elif dataset_name == 'slice':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'slice_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'slice_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # servo
    elif dataset_name == 'servo':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'servo_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'servo_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # autos
    elif dataset_name == 'autos':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'autos_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'autos_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # concreteslump
    elif dataset_name == 'concreteslump':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'concreteslump_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'concreteslump_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # fertility
    elif dataset_name == 'fertility':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'fertility_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'fertility_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # forest
    elif dataset_name == 'forest':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'forest_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'forest_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # houseelectric
    elif dataset_name == 'houseelectric':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'houseelectric_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'houseelectric_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # yacht
    elif dataset_name == 'yacht':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'yacht_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'yacht_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # kin40k
    elif dataset_name == 'kin40k':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'kin40k_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'kin40k_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # machine
    elif dataset_name == 'machine':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'machine_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'machine_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
        
    # solar
    elif dataset_name == 'solar':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'solar_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'solar_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pol
    elif dataset_name == 'pol':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pol_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pol_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pendulum
    elif dataset_name == 'pendulum':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pendulum_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pendulum_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # pumadyn32nm
    elif dataset_name == 'pumadyn32nm':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'pumadyn32nm_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'pumadyn32nm_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
    
    # buzz
    elif dataset_name == 'buzz':
        test_mask = np.loadtxt(
                fname=os.path.join(dataset_path, 'buzz_mask.txt'),
                dtype=bool,
                delimiter=",",
            )
        data = np.loadtxt(
                fname=os.path.join(dataset_path, 'buzz_train.txt'),
                dtype=np.float64,
                delimiter=",",
            )
        # generate the train_mask
        # train_mask and test mask are opposite, i.e. test_mask = np.logical_not(train_mask)
        train_mask = np.logical_not(test_mask)

        # extract the inputs and reponse
        X_tot = data[:, :-1]
        y_tot = data[:, -1, None]
        
        # Apply train-test split  
        split = np.random.randint(10)
        X_test  = X_tot[test_mask[:, split], :]
        X_train = X_tot[train_mask[:, split], :]
        y_test  = y_tot[test_mask[:, split], :].reshape(-1)
        y_train = y_tot[train_mask[:, split], :].reshape(-1)
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe

        # Global response bound 
        C_max = 1.0
                                   
    # Breast Cancer dataset
    elif dataset_name == 'breastcancer':
        
        # Load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'breastcancer.txt'), delimiter=",")
        
        # Preprocess dataset 
        m, n = data.shape

        # Randomly permute the rows
        p = np.random.permutation(m)
        data = data[p, :]

        # Select columns [1:2, 4:end-1] (0-indexed: [0, 1] and [3:n-1])
        X = data[:, list(range(0, 2)) + list(range(3, n - 1))]
        y = data[:, 2]
        
        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # autompg 
    elif dataset_name == 'autompg':
        # load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'autompg.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
    
    # challenger 
    elif dataset_name == 'challenger':
        # chellenger dataset 
        challenger = pd.read_csv(os.path.join(dataset_path, "ChallengerDataset.csv"))

        # Convert to NumPy once
        data = challenger.to_numpy().astype(np.float64)
    
        # Split into features (X) and target (y)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]

        X_train = X
        y_train = y
        X_test = X
        y_test = y
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
    
    # parkinsons 
    elif dataset_name == 'parkinsons':
        # fetch dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'parkinsons.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # airfoil 
    elif dataset_name == 'airfoil':
        # load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'airfoil.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # concrete dataset 
    elif dataset_name == 'concrete':
        # load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'Concrete.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # Energy dataset 
    elif dataset_name == 'energy':
        # load dataset 
        data = np.loadtxt(os.path.join(dataset_path, 'energy.txt'))

        # 2. Shuffle the rows (similar to p = randperm(m)).
        m, n = data.shape
        p = np.random.permutation(m)
        data = data[p, :]

        # 3. Separate features (X) and target (y).
        X = data[:, :-1]
        y = data[:, -1]

        # apply train test split 
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
        
    # Bike sharing 
    elif dataset_name == 'bike':
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df=pd.read_csv(os.path.join(dataset_path, 'bike_train.csv'))

        # # seperating season as per values. this is bcoz this will enhance features.
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)

        # # # now can drop weather and season.
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})
    
        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values.astype(np.float64)
        y = df['count'].values.astype(np.float64)

        # Apply random permutation 
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        # Split into training and testing sets
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0 
    
    # Tecator 
    elif dataset_name == 'tecator':
        tecator = fetch_openml(name="Tecator", version=1, as_frame=True)

        # Extract the 100 absorbance features (cols are named like "Absorbance1", …)
        X = tecator.data.to_numpy()

        # Choose "Fat" as the regression target (you could also predict Moisture or Protein)
        y = tecator.target.astype(float).to_numpy()

        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max = 1.0
    
    # Wine dataset
    elif dataset_name == 'wine':
        wine = pd.read_csv(os.path.join(dataset_path, "winequality-red.csv"))

        # Drop duplicates 
        dub_wine=wine.copy()
        dub_wine.drop_duplicates(subset=None,inplace=True)

        y=dub_wine.pop('quality').to_numpy().astype(np.float64)
        X=dub_wine.to_numpy().astype(np.float64)
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0

    # SkillCraft dataset 
    elif dataset_name == 'SkillCraft':
        skillcraft = pd.read_csv(os.path.join(dataset_path, "SkillCraft.csv"))

        # Drop duplicates 
        dub_skillcraft=skillcraft.copy()
        dub_skillcraft.dropna(subset=None,inplace=True)

        y = dub_skillcraft['Age'].to_numpy().astype(np.float64)
        X = dub_skillcraft.drop(columns=['GameID', 'Age']).to_numpy()
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
    
    # Gas dataset 
    elif dataset_name == 'Gas':
        gas = pd.read_csv(os.path.join(dataset_path, "GasDataset.csv"))

        # Drop duplicate rows and missing values
        gas_cleaned = gas.drop_duplicates().dropna()

        # Convert to NumPy once
        data = gas_cleaned.to_numpy().astype(np.float64)
    
        # Split into features (X) and target (y)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        C_max       = 1.0
            
    elif dataset_name == 'crime':
        # load crime dataset, standardize features, and split into train and test
        X_train, X_test, z_train, z_test, y_train, y_test = read_crimes(env_partition=0.1)
        test_size = len(y_test)
        n_samples = len(y_train) + test_size

        C_max = 1.0 

    # Tamielectric dataset
    elif dataset_name == 'tamielectric':
        # load the tamielectric dataset 
        tamielectric = pd.read_csv(os.path.join(dataset_path, "tamielectric.csv"))
        tamielectric_numpy = tamielectric.to_numpy().astype(np.float64)
        
        # Drop duplicates 
        y=tamielectric_numpy[:, -1]
        X=tamielectric_numpy[:, :-1]
        
        # Apply random permutation and apply train test split
        m, n = X.shape
        p = np.random.permutation(m)
        X = X[p, :]
        y = y[p]
        
        train_size = int(0.8 * len(y))
        test_size = len(y) - train_size

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Normalize response variables
        norm_fact_y = np.max(np.abs(y_train))
        y_train /= norm_fact_y
        y_test  /= norm_fact_y
        
        # Standardize features
        feat_mean = X_train.mean(axis=0)
        feat_std  = X_train.std(axis=0, ddof=0)
        feat_std_safe = np.where(feat_std > 0, feat_std, 1.0)

        X_train = (X_train - feat_mean) / feat_std_safe
        X_test  = (X_test  - feat_mean) / feat_std_safe
        
        C_max       = 1.0

    # normalize dataset to have power 1 
    norm_fact = np.sqrt(np.max(np.sum(X_train**2, 1)))  # np.mean(np.mean(X_train**2))
    X_train = X_train/norm_fact
    X_test = X_test/norm_fact
    n, d = X_train.shape

    # Calculate minimum eigenvalue of X^T X and of (X,Y)^T (X,Y)
    lambda_max = np.max(np.linalg.eigvals(X_train.T @ X_train))
    lambda_min = np.min(np.linalg.eigvals(X_train.T @ X_train))
    y_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    XY = np.hstack((X_train, y_col))  
    lambda_min_XY = np.real(np.min(np.linalg.eigvals(XY.T @ XY)))
    
    dataset_title = dataset_titles[dataset_name]
    return C_max, X_train, y_train, X_test, y_test, np.real(lambda_min), np.real(lambda_max), np.real(lambda_min_XY), n, d, dataset_title
