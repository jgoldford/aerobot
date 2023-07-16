import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# define asset path
asset_path,filename = os.path.split(os.path.abspath(__file__))
asset_path = asset_path + '/assets'


def run_pca(X,n_components=2,normalize=True):
	results = {}
	pca = PCA(n_components=n_components)
	if normalize:
		Xn = StandardScaler().fit_transform(X.values)
	else:
		Xn = X.values
	Xpca = pca.fit_transform(Xn)

	labels = ["PC{x} ({y}%)".format(x=x,y=round(y*100,2)) for x,y in list(zip(range(1,n_components+1),pca.explained_variance_ratio_))]
	pdf = pd.DataFrame(Xpca,index=X.index,columns =labels)
	results["pdf"] = pdf
	results["pca"] = pca
	return results