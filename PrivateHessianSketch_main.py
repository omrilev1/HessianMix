import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm 
from utils_Hessian_mixing import * 
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")  # suppress all warnings from all modules

# Fix initial seed for reproducibility 
random.seed(50)
np.random.seed(50)

# Full set of datasets to run our algorithms: can select a partial list of these for current run  
datasets_run_list = ['housing', 'tecator', 'wine', 'bike', 'autompg', 'energy', 'concrete', 'elevators', 'Gas', 'crime', 'concreteslump', \
                     'airfoil', 'breastcancer', 'parkinsons', 'sml', 'keggundirected', 'pol', 'solar', \
                     'keggdirected', 'yacht', 'servo', 'autos', 'fertility', 'forest',\
                     'machine', 'protein', 'pendulum', 'slice', 'buzz', 'pumadyn32nm', '3droad', 'kin40k', 'tamielectric']


############# Hyperparameter T ############# 
iters_IHM = 3

############# Hyperparameter: k #############
# For IHM, overall k is going to be percentage_k_IHM * max(d, log(4T/varrho))
# For LinearMixing, overall k is going to be percentage_k_Mix * max(d, log(1/varrho))
percentage_k_IHM = 6  
percentage_k_Mix = 2.5 

############# Hyperparameter: learning rate for DP-GD #############
# Globl learning rate for all the different cases: calibrated offline to provide the best performance among all cases 
lr_dp_gd = 0.25

# Number of Monte-Carlo iterations
iters = 500 

# Which methods to plot: pick from the next list 
# 'AdaSSP'          : AdaSSP
# 'LinearMix'       : Linear Mixing
# 'HessianMix'      : Hessian mixing with iters_IHM iterations
# 'HessianMix_T_1'  : Hessian mixing with a single iteration, with the same k-value as Linear Mixing 
# 'DPGD'            : DP-GD     
methods_to_plot = ['AdaSSP', 'LinearMix', 'HessianMix']
add_legend_to_plot = False 

# epsilon values to run our algorithm 
epsilon_values = np.logspace(-1.0, 1.0, 6) # np.logspace(-1.0, 1.0, 6)

# Start looping over datasets 
for dataset_type in datasets_run_list:
    
    print('Running dataset ' + dataset_type)

    # Parse dataset 
    C_max, X_train, y_train,\
        X_test, y_test, \
        lambda_min, lambda_max, lambda_min_XY,\
        n, d, dataset_title = GetDataset(dataset_type)
        
    print('=============================')
    print('lambda min is ' + str(lambda_min)
          + ', lambda min XY is ' + str(lambda_min_XY))
    print('n is ' + str(n) + ', d is ' + str(d))
    print('=============================')

    # Delta and target varrho 
    delta_DP = 1/(n**2)
    target_varrho = delta_DP/10

    # Hyerparameter: tau - eigenvalue threshold constant 
    tau = np.sqrt(2.0 * np.log(np.max((3.0/delta_DP, 2.0/target_varrho))))
    tau_mix = np.sqrt(2.0 * np.log(np.max((4.0/delta_DP, 4.0/target_varrho))))
    
    # Baseline: OLS
    lambda_baseline = 1e-16 
    ols_model = LinearRegression(fit_intercept=False)  # alpha is the regularization strength
    ols_model.fit(X_train, y_train)
    ridge_y_pred = ols_model.predict(X_test)
    baseline_test_mse = np.sum((y_test - ridge_y_pred)**2)
    ridge_y_pred_train = ols_model.predict(X_train)
    baseline_train_mse = np.sum((y_train - ridge_y_pred_train)**2)

    print('============================')
    print('baseline test mse is  ' + str(baseline_test_mse/n))
    print('baseline train mse is ' + str(baseline_train_mse/n))
    print('============================')

    # Set k_value 
    k_val_LinMix = np.max((int(percentage_k_Mix * d), int(percentage_k_Mix * np.log(2.0/target_varrho))))  # target_rho is the failure probability of the random projection
    k_val_IHM    = np.max((int(percentage_k_IHM * d), \
        int(percentage_k_IHM * np.log(4.0*iters_IHM/target_varrho))))  # target_rho is the failure probability of the random projection

    # Prepare a list to store MSE for each sigma
    mse_for_sigmas                    = []
    mse_for_sigmas_std                = []
    mse_for_sigmas_adassp             = []
    mse_for_sigmas_std_adassp         = []
    mse_for_sigmas_IHM                = []
    mse_for_sigmas_std_IHM            = []
    mse_for_sigmas_IHM_T_1            = []
    mse_for_sigmas_std_IHM_T_1        = []
    mse_for_sigmas_DPGD               = []
    mse_for_sigmas_std_DPGD           = []
    
    train_mse_for_sigmas              = []
    train_mse_for_sigmas_std          = []
    train_mse_for_sigmas_adassp       = []
    train_mse_for_sigmas_std_adassp   = []
    train_mse_for_sigmas_IHM          = []
    train_mse_for_sigmas_std_IHM      = []
    train_mse_for_sigmas_IHM_T_1      = []
    train_mse_for_sigmas_std_IHM_T_1  = []
    train_mse_for_sigmas_DPGD         = []
    train_mse_for_sigmas_std_DPGD     = []
    
    # Iterate over different epsilons
    for eps_idx, eps in tqdm(enumerate(epsilon_values)):
        print('Started eps = ' + str(eps))
        
        # Initiate arrays for results of current \eps 
        curr_test_mse_linear_mixing  = []
        curr_test_mse_adassp         = [] 
        curr_test_mse_IHM            = [] 
        curr_test_mse_IHM_T_1        = [] 
        curr_test_mse_DP_GD          = [] 
        
        curr_train_mse_linear_mixing = []
        curr_train_mse_adassp        = [] 
        curr_train_mse_IHM           = [] 
        curr_train_mse_IHM_T_1       = []
        curr_train_mse_DP_GD         = []
        
        ######################## Compute Noise Values ##########################
        # Baseline for numeric calculations: (\eps, \delta)-DP with classical Gaussian mechanism 
        sigma_eps_delta_DP_Gaussian            = 2.0 * np.log(1.25/delta_DP) / (eps**2)
        
        ######################## Linear Mixing ########################
        # Linear mixing: noise for sketching required for target (\eps,\delta)-DP with inflation of (C_max)**2
        gamma_matrix        = solve_gamma_renyi_full(init_gamma=sigma_eps_delta_DP_Gaussian, k=k_val_LinMix, target_delta=delta_DP,\
                                                        target_epsilon=eps, inflation_norm=C_max**2)
        
        # we mutiply by \sqrt{1 + C_max**2} since the sensitivity of the 
        # minimal eigenvalue lambda_min_XY is (1 + C_max**2), and \sqrt{\gamma} (and also sigma_eigenval) has units of \sqrt{1.0 + C_max**2} 
        sigma_eigenval      = np.sqrt(1.0 + C_max**2) * gamma_matrix/np.sqrt(k_val_LinMix)
        
        ######################## IHM with T=1 ########################
        # Hessian mixing with T=1: noise for sketching required for target (\eps/2,\delta/2)-DP with no norm inflation
        sigma_IHM_T_1_X     = solve_gamma_renyi_full(init_gamma=sigma_eps_delta_DP_Gaussian, k=k_val_LinMix, target_delta=3.0*delta_DP/4.0,\
                                                        target_epsilon=eps/2.0, inflation_norm=0.0)
        sigma_eigenval_IHM_T_1 =  sigma_IHM_T_1_X/np.sqrt(k_val_LinMix)
        sigma_IHM_T_1_XTY   = calibrateAnalyticGaussianMechanism(eps/2.0, delta_DP/4.0, C_max, tol = 1.e-12)
        
        ######################## IHM with T=iters_IHM ########################
        # Hessian mixing with T=iters_IHM: noise for sketching required for target (\eps/2,\delta/2)-DP with no norm inflation
        gamma_matrix_half_iter = solve_gamma_renyi_full_composition(init_gamma=sigma_eps_delta_DP_Gaussian, k=k_val_IHM, \
                                            target_delta=3.0*delta_DP/4.0, target_epsilon=eps/2.0, inflation_norm=0.0, T=iters_IHM)
        sigma_eigenval_half_iter =  gamma_matrix_half_iter/np.sqrt(k_val_IHM)            
        sigma_IHM_XTY = calibrateAnalyticGaussianMechanism(eps/2.0, delta_DP/4.0, C_max, tol = 1.e-12) 
        sigma_IHM_XTY *= np.sqrt(iters_IHM)    
        # sigma_IHM_XTY_analytic_conversion_RDP_to_DP = solve_Gaussian_composition(sigma_eps_delta_DP_Gaussian, delta_DP/4.0, \
        #             eps/2.0, C_max, iters_IHM)
        
        ######################## AdaSSP ########################
        # AdaSSP base noise value: Gaussian mechanism for target of (\eps/3, \delta/3)
        sigma_base_AdaSSP_XTX   = calibrateAnalyticGaussianMechanism(eps/3.0, delta_DP/3.0, 1.0, tol = 1.e-12) 
        sigma_base_AdaSSP_XTY   = calibrateAnalyticGaussianMechanism(eps/3.0, delta_DP/3.0, C_max, tol = 1.e-12) 
        
        ######################## DP-GD ########################
        # DP-GD: Similar number of epochs as the IHM; Noise values taken from [Brown '24]
        rho_dpgd    = (np.sqrt(eps + np.log(1.0/delta_DP)) - np.sqrt(np.log(1.0/delta_DP)))**2 
        sigma_dp_gd = np.sqrt(2.0 * iters_IHM * C_max**2 / rho_dpgd / (n**2))

        # In case the current dataset is very large, we simulate with a smaller number of iterations to allow for a reasonable runtime 
        if n * d > 500000:
            curr_iters = 20
        else:
            curr_iters = iters 
        
        # Monte-carlo run start here 
        for iter in tqdm(range(curr_iters)):

            ############################### Linear Mixing ###############################
            y_train_vec = y_train.reshape(-1, 1)  # (n,1)
            
            # Sample sketch and noises 
            S = np.random.randn(k_val_LinMix, n) 
            N_ours = np.random.randn(k_val_LinMix, d)
            N_ours_y = np.random.randn(k_val_LinMix)
            
            # Calculate minimal eigenvalue and noise level
            gamma_tilde = np.max((0, lambda_min_XY - sigma_eigenval * (tau - np.random.randn())))
            gamma_tilde = np.sqrt(np.max((0, gamma_matrix - gamma_tilde)))
            
            # Sketch X and Y 
            X_train_full_PR = S @ X_train + gamma_tilde * N_ours  
            y_PR = (S @ y_train_vec).ravel() + gamma_tilde * N_ours_y

            # Solve the regression 
            theta_LinMix = np.linalg.inv(X_train_full_PR.T @ X_train_full_PR) @ X_train_full_PR.T @ y_PR
            
            ############################### IHM with T = 1 ###############################
            
            # Sample sketch and noises 
            S_iter = np.random.randn(k_val_IHM, n) 
            N_ours_iter = np.random.randn(k_val_IHM, d)
            N_ours_y_iter = np.random.randn(k_val_IHM)
            
            # Calculate minimal eigenvalue and noise level 
            gamma_tilde = np.max((0, lambda_min - sigma_eigenval_IHM_T_1 * (tau_mix - np.random.randn())))
            gamma_tilde = np.sqrt(np.max((0, sigma_IHM_T_1_X - gamma_tilde)))
            
            # Sketch X and add noise to XTY 
            X_train_full_PR_hessian = S @ X_train + gamma_tilde * N_ours
            H_hat = (X_train_full_PR_hessian.T @ X_train_full_PR_hessian) / k_val_LinMix  # ≈ X^T X
            XY_Pr = X_train.T @ y_train + sigma_IHM_T_1_XTY * np.random.randn(d)
            
            theta_IHM_T_1 = np.linalg.solve(H_hat, XY_Pr)
            
            ############################### IHM with T = iters_IHM ###############################            
            
            # Calculate minimal eigenvalue and noise level 
            gamma_tilde = np.max((0, lambda_min - sigma_eigenval_half_iter * (tau_mix - np.random.randn())))
            gamma_final = np.sqrt(np.max((0, gamma_matrix_half_iter - gamma_tilde)))

            # Iterative loop    
            theta_IHM_T_iter = np.zeros(d) 
            for iter_hess in range(iters_IHM):
                
                # Sketch X and add noise to XTY 
                S = np.random.randn(k_val_IHM, n) 
                N_ours = np.random.randn(k_val_IHM, d)
                X_train_full_PR_hessian = S @ X_train + gamma_final * N_ours 

                H_hat = (X_train_full_PR_hessian.T @ X_train_full_PR_hessian) / k_val_IHM  # ≈ X^T X
                XY_Pr = X_train.T @ np.clip(y_train - X_train @ theta_IHM_T_iter, -C_max, C_max) +\
                    sigma_IHM_XTY * np.random.randn(d)
                    
                # Update solution 
                theta_IHM_T_iter += np.linalg.solve(H_hat, XY_Pr)

            ############################### AdaSSP ###############################
            
            # Calculate minimal eigenvalue and noise level 
            lambda_min_tilde = np.max((0, lambda_min + sigma_base_AdaSSP_XTX * np.random.randn() - sigma_base_AdaSSP_XTX**2))
            lambda_adassp = np.max((0, np.sqrt(d * np.log(2.0*(d**2)/(target_varrho)))*sigma_base_AdaSSP_XTX - lambda_min_tilde))
            
            # Generate solution for AdaSSP
            N_upper = np.random.randn(d, d)
            N = np.triu(N_upper)  # Upper triangular part
            N_sym = N + N.T - np.diag(np.diag(N))  # Make symmetric
            XTX_noisy = X_train.T @ X_train + sigma_base_AdaSSP_XTX * N_sym  
            XTY_noisy = X_train.T @ y_train + sigma_base_AdaSSP_XTY * np.random.randn(d)  
            theta_adassp = np.linalg.inv(XTX_noisy + lambda_adassp*np.eye(d)) @ XTY_noisy

            ############################### DP_GD ###############################
            theta_DPGD = np.zeros(d)
            
            # Run DP-GD iterations 
            for iter in range(iters_IHM):
                gt_clip = -1.0 * (1/n) * X_train.T @ np.clip(y_train - X_train @ theta_DPGD, -C_max, C_max)
                theta_DPGD += lr_dp_gd * (sigma_dp_gd * np.random.randn(d) - gt_clip)
            
            ############################### Evaluate ###############################
            # Evaluate on original test data
            y_test_pred = X_test @ theta_LinMix
            curr_test_mse_linear_mixing.append(np.mean((y_test - y_test_pred)**2))
            y_train_pred = X_train @ theta_LinMix
            curr_train_mse_linear_mixing.append(np.mean((y_train - y_train_pred)**2))          
            
            y_test_pred = X_test @ theta_IHM_T_1
            curr_test_mse_IHM_T_1.append(np.mean((y_test - y_test_pred)**2))
            y_train_pred = X_train @ theta_IHM_T_1
            curr_train_mse_IHM_T_1.append(np.mean((y_train - y_train_pred)**2))
            
            y_test_pred = X_test @ theta_IHM_T_iter
            curr_test_mse_IHM.append(np.mean((y_test - y_test_pred)**2))
            y_train_pred = X_train @ theta_IHM_T_iter
            curr_train_mse_IHM.append(np.mean((y_train - y_train_pred)**2))
            
            y_test_pred = X_test @ theta_DPGD
            curr_test_mse_DP_GD.append(np.mean((y_test - y_test_pred)**2))
            y_train_pred = X_train @ theta_DPGD
            curr_train_mse_DP_GD.append(np.mean((y_train - y_train_pred)**2))
            
            y_test_pred = X_test @ theta_adassp
            curr_test_mse_adassp.append(np.mean((y_test - y_test_pred)**2))
            y_train_pred = X_train @ theta_adassp
            curr_train_mse_adassp.append(np.mean((y_train - y_train_pred)**2))

        # MSEs and confidence intervals 
        mse_for_sigmas.append(np.mean(curr_test_mse_linear_mixing))
        mse_for_sigmas_std.append(1.96 * np.std(curr_test_mse_linear_mixing)/np.sqrt(curr_iters))
        mse_for_sigmas_adassp.append(np.mean(curr_test_mse_adassp))
        mse_for_sigmas_std_adassp.append(1.96 * np.std(curr_test_mse_adassp)/np.sqrt(curr_iters))
        mse_for_sigmas_IHM_T_1.append(np.mean(curr_test_mse_IHM_T_1))
        mse_for_sigmas_std_IHM_T_1.append(1.96 * np.std(curr_test_mse_IHM_T_1)/np.sqrt(curr_iters))
        mse_for_sigmas_IHM.append(np.mean(curr_test_mse_IHM))
        mse_for_sigmas_std_IHM.append(1.96 * np.std(curr_test_mse_IHM)/np.sqrt(curr_iters))
        mse_for_sigmas_DPGD.append(np.mean(curr_test_mse_DP_GD))
        mse_for_sigmas_std_DPGD.append(1.96 * np.std(curr_test_mse_DP_GD)/np.sqrt(curr_iters))
        
        train_mse_for_sigmas.append(np.mean(curr_train_mse_linear_mixing))
        train_mse_for_sigmas_std.append(1.96 * np.std(curr_train_mse_linear_mixing)/np.sqrt(curr_iters))
        train_mse_for_sigmas_adassp.append(np.mean(curr_train_mse_adassp))
        train_mse_for_sigmas_std_adassp.append(1.96 * np.std(curr_train_mse_adassp)/np.sqrt(curr_iters))
        train_mse_for_sigmas_IHM_T_1.append(np.mean(curr_train_mse_IHM_T_1))
        train_mse_for_sigmas_std_IHM_T_1.append(1.96 * np.std(curr_train_mse_IHM_T_1)/np.sqrt(curr_iters))
        train_mse_for_sigmas_IHM.append(np.mean(curr_train_mse_IHM))
        train_mse_for_sigmas_std_IHM.append(1.96 * np.std(curr_train_mse_IHM)/np.sqrt(curr_iters))
        train_mse_for_sigmas_DPGD.append(np.mean(curr_train_mse_DP_GD))
        train_mse_for_sigmas_std_DPGD.append(1.96 * np.std(curr_train_mse_DP_GD)/np.sqrt(curr_iters))        

        print('====================')
        print('Test MSE: '          + str(np.mean(curr_test_mse_linear_mixing)))
        print('Test MSE ADASSP: '   + str(np.mean(curr_test_mse_adassp)))
        print('Test MSE Hessian T=1: '  + str(np.mean(curr_test_mse_IHM_T_1)))
        print('Test MSE Hessian '    + str(iters_IHM) + ' iters: ' + str(np.mean(curr_test_mse_IHM)))
        print('Test MSE DP GD: '    + str(np.mean(curr_test_mse_DP_GD)))

        print('====================')
        print('Train MSE: '         + str(np.mean(curr_train_mse_linear_mixing)))
        print('Train MSE ADASSP: '  + str(np.mean(curr_train_mse_adassp)))
        print('Train MSE Hessian: ' + str(np.mean(curr_train_mse_IHM_T_1)))
        print('Train MSE Hessian '   + str(iters_IHM) + ' iters: ' + str(np.mean(curr_train_mse_IHM)))
        print('Train MSE DP GD: '   + str(np.mean(curr_train_mse_DP_GD)))
        
    ################################# Generate Plots #################################
    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Extract the single k value
    label_suffix = r"$\frac{k}{\max\{d,\log(\frac{1}{\varrho})\}} = " + f"{percentage_k_Mix:.1f}$"
    label_suffix_iter = r"$\frac{k}{\max\{d,\log(\frac{4T}{\varrho})\}} = " + f"{percentage_k_IHM:.1f}$"
    
    # Define colors and markers for consistency
    colors = {
        'adassp': 'blue', 
        'linear_mixing': 'orangered',
        'hessian_mixing': 'green',
        'hessian_mixing_T_1': 'purple',
        'DPGD': 'black',
    }

    markers = {
        'adassp': '*',
        'linear_mixing': 'o',
        'hessian_mixing': 'D',
        'hessian_mixing_T_1': 'P',
        'DPGD': 'X',
    }
 
    eps = np.asarray(epsilon_values, dtype=float)

    # Right subplot: Test MSE
    if 'AdaSSP' in methods_to_plot:
        plt.plot(epsilon_values, train_mse_for_sigmas_adassp,
                 color=colors['adassp'], marker=markers['adassp'], markersize=16,
                 label="AdaSSP [Wang '18]")
        plt.fill_between(epsilon_values,
                         np.array(train_mse_for_sigmas_adassp) - np.array(train_mse_for_sigmas_std_adassp),
                         np.array(train_mse_for_sigmas_adassp) + np.array(train_mse_for_sigmas_std_adassp),
                         color=colors['adassp'], alpha=0.2)
    
    if 'LinearMix' in methods_to_plot:
        plt.plot(epsilon_values, train_mse_for_sigmas,
                 color=colors['linear_mixing'], marker=markers['linear_mixing'], markersize=16,
                 label=fr"Linear mixing [Lev et al. '25], {label_suffix}")
        plt.fill_between(epsilon_values,
                         np.array(train_mse_for_sigmas) - np.array(train_mse_for_sigmas_std),
                         np.array(train_mse_for_sigmas) + np.array(train_mse_for_sigmas_std),
                         color=colors['linear_mixing'], alpha=0.2)
    
    if 'DPGD' in methods_to_plot: 
        plt.plot(epsilon_values, train_mse_for_sigmas_DPGD,
                 color=colors['DPGD'], marker=markers['DPGD'], markersize=16,
                 label=fr"DP-GD [Brown et al. '24]")
        plt.fill_between(epsilon_values,
                         np.array(train_mse_for_sigmas_DPGD) - np.array(train_mse_for_sigmas_std_DPGD),
                         np.array(train_mse_for_sigmas_DPGD) + np.array(train_mse_for_sigmas_std_DPGD),
                         color=colors['DPGD'], alpha=0.2)
    
    if 'HessianMix' in methods_to_plot: 

            # line (same color & linestyle), different marker per iteration
            label_iterative = (fr"IHM (ours), {iters_IHM} iters, {label_suffix_iter}")
            plt.plot(eps, train_mse_for_sigmas_IHM,
                     linestyle='--',
                     color=colors['hessian_mixing'],
                     marker=markers['hessian_mixing'],
                     markersize=16,
                     label=label_iterative)

            plt.fill_between(epsilon_values,
                         np.array(train_mse_for_sigmas_IHM) - np.array(train_mse_for_sigmas_std_IHM),
                         np.array(train_mse_for_sigmas_IHM) + np.array(train_mse_for_sigmas_std_IHM),
                         color=colors['hessian_mixing'], alpha=0.2)
    
    if 'HessianMix_T_1' in methods_to_plot: 

            # line (same color & linestyle), different marker per iteration
            label_iterative = (fr"IHM (ours): single iteration, {label_suffix}")
            plt.plot(eps, train_mse_for_sigmas_IHM_T_1,
                     linestyle='--',
                     color=colors['hessian_mixing_T_1'],
                     marker=markers['hessian_mixing_T_1'],
                     markersize=16,
                     label=label_iterative)

            plt.fill_between(epsilon_values,
                         np.array(train_mse_for_sigmas_IHM_T_1) - np.array(train_mse_for_sigmas_std_IHM_T_1),
                         np.array(train_mse_for_sigmas_IHM_T_1) + np.array(train_mse_for_sigmas_std_IHM_T_1),
                         color=colors['hessian_mixing_T_1'], alpha=0.2)


    # Configure both subplots
    plt.xlabel(r"$\epsilon_{\mathrm{DP}}$", fontsize=38)
    plt.ylabel('Train MSE', fontsize=36)
    plt.xscale('log')
    plt.ticklabel_format(axis='y', scilimits=(-1, 1), useMathText=True)
    plt.tick_params(axis='both', which='both', labelsize=36)  
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(36)

    plt.tight_layout()
    
    if add_legend_to_plot:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), bbox_transform=plt.gcf().transFigure,  # anchor to figure
                        ncol=3, fontsize=22, frameon=False)
        
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/hessian_mixing_manuscript_{dataset_type}_n_{n}_d_{d}_k_{percentage_k_Mix}.pdf", bbox_inches='tight')
