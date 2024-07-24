import pandas as pd
from vmdpy import VMD


data = pd.read_csv('data/cleaned_pv.csv')


class VMDecomposition:

    def __init__(self, data, target="Active_Power"):
        self.data = data
        self.target_values = self.data[target].values

    def get_imfs(self, k=7):
        u, u_hat, omega = VMD(
            f=self.target_values,
            alpha=2000,
            tau=0,
            K=k,
            DC=0,
            init="nndsvda",
            tol=1e-7,
        )
        return u, omega
    

if __name__ == '__main__':
    vmd = VMDecomposition(data)
    imfs, omega = vmd.get_imfs()
    print("Finished Decomposition")
    
    sum_imfs = imfs.sum(axis=0)
    sum_omega = omega.sum(axis=0)
    print(sum_imfs.shape)
    print(sum_imfs)
    print(sum_omega)


