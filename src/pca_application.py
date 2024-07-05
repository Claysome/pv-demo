import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCAApplication:
    components_bound = np.arange(2, 9, 1)

    @classmethod
    def apply(cls, data, n_components):
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(data)
        return pca_features
    
    @classmethod
    def plot_explained_variance(cls, data):
        pca = PCA().fit(data)
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')
        plt.title('Explained Variance')
        plt.grid(True)
        plt.savefig(Path(__file__).parent.parent / "images" / "pca_explained_var.png")
        plt.show()
