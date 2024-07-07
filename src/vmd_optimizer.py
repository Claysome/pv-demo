import itertools
import numpy as np
import pandas as pd
from vmdpy import VMD
from tqdm import tqdm
from pathlib import Path
from scipy.signal import hilbert
# from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt


class VmdOptimizer:
    varbound = np.array([[0, 3000], [1, 10]])
    alpha_values = np.arange(0, 3001, 1000)
    k_values = np.arange(3, 6)     
    
    def __init__(self, data):
        self.data = data

    def get_decomposed_signal(self, alpha, K):
        v = VmdOptimizer.vmd_decompose(self.data, alpha, K)
        df_vmd = pd.DataFrame(v.T)
        df_vmd.columns = ['imf'+str(i) for i in range(K)]
        return df_vmd

    def get_decomposed_signal_by_opt(self):
        opt_params = self.optimize()
        alpha, K = opt_params[0], int(opt_params[1])
        v = VmdOptimizer.vmd_decompose(self.data, alpha, K)
        df_vmd = pd.DataFrame(v.T)
        df_vmd.columns = ['imf'+str(i) for i in range(K)]
        return df_vmd

    def optimize(self):
        # def loss_function(x):
        #     return VmdOptimizer.loss_function(self.data, x)
        
        # model = ga(
        #     function=loss_function,
        #     dimension=2,
        #     variable_type='real',
        #     variable_boundaries=self.varbound,
        #     algorithm_parameters={
        #         'max_num_iteration': 100,
        #         'population_size': 50,
        #         'mutation_probability': 0.1,
        #         'elit_ratio': 0.01,
        #         'crossover_probability': 0.5,
        #         'parents_portion': 0.3,
        #         'crossover_type': 'uniform',
        #         'max_iteration_without_improv': None
        #     },
        #     function_timeout=60  # 增加超时时间到60秒
        # )
        # model.run()
        # opt_params = model.output_dict['variable']
        best_loss, opt_params = np.inf, None

        for alpha, k in tqdm(itertools.product(self.alpha_values, self.k_values)):
            loss = VmdOptimizer.loss_function(self.data, (alpha, k))
            print(f"Alpha: {alpha}, K: {k}, Loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                opt_params = (alpha, k)
        print(f"Optimal parameters: Alpha: {opt_params[0]}, K: {int(opt_params[1])}, Loss: {best_loss}")
        return opt_params
    
    @staticmethod
    def plot_decomposed_signal(signal, decomposed_signal):
        plt.figure()
        num_components = len(decomposed_signal)
        plt.subplot(num_components + 1, 1, 1)
        plt.plot(signal)
        plt.title("Original Signal")
        for i in range(num_components):
            plt.subplot(num_components + 1, 1, i + 2)
            plt.plot(decomposed_signal[i])
            plt.title(f"VMD Component {i + 1}")
        plt.tight_layout()
        plt.savefig(Path(__file__).parent.parent / "images" / "vmd_decomposition.png")
        plt.show()

    @staticmethod
    def calculate_envelope_entropy(imfs):
        entropies = []
        for imf in imfs:
            analytic_signal = hilbert(imf)
            amptitude_envelope = np.abs(analytic_signal)
            normalized_envelope = amptitude_envelope / np.sum(amptitude_envelope)
            envelope_entropy = - np.sum(normalized_envelope * np.log(normalized_envelope + 1e-12))
            entropies.append(envelope_entropy)
        return np.sum(entropies)
    
    @staticmethod
    def loss_function(signal, x):
        alpha, K = x[0], int(x[1])
        v = VmdOptimizer.vmd_decompose(signal, alpha, K)
        loss = VmdOptimizer.calculate_envelope_entropy(v)
        return loss
    
    @staticmethod
    def vmd_decompose(signal, alpha, K):
        tau = 0.0
        DC = 0
        init = 1      
        tol = 1e-7
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        return u
    
