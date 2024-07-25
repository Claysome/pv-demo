from vmdpy import VMD
import matplotlib.pyplot as plt


class VMDecomposition:

    @classmethod
    def get_imfs(cls, target_values, k=6):
        u, u_hat, omega = VMD(target_values, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
        return u, omega
    
    @classmethod
    def plot_imfs(cls, imfs):
        plt.figure(figsize=(16, 10))
        for i, imf in enumerate(imfs):
            plt.subplot(len(imfs), 1, i+1)
            plt.plot(imf)
        plt.show()


