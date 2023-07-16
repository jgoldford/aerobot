import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# define asset path
asset_path,filename = os.path.split(os.path.abspath(__file__))
asset_path = asset_path + '/assets'


def process_data(df_train_features, df_train_labels, df_test_features, df_test_labels):
    # Merge feature and label dataframes based on index
    train_data = pd.merge(df_train_features, df_train_labels, left_index=True, right_index=True)
    test_data = pd.merge(df_test_features, df_test_labels, left_index=True, right_index=True)

    # Ensure the columns in the features for both train and test data match
    train_data, test_data = train_data.align(test_data, axis=1, join='inner')

    # Save column labels (excluding the last column which is the label)
    column_labels = list(train_data.columns[:-1])

    # Save row labels for train and test
    train_row_labels = list(train_data.index)
    test_row_labels = list(test_data.index)

    # Convert dataframe to numpy array for sklearn compatibility
    X_train = train_data.iloc[:, :-1].values  # All columns except the last
    y_train = train_data.iloc[:, -1].values  # Only the last column
    X_test = test_data.iloc[:, :-1].values  # All columns except the last
    y_test = test_data.iloc[:, -1].values  # Only the last column

    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_row_labels': train_row_labels,
        'test_row_labels': test_row_labels,
        'column_labels': column_labels
    }

    return data_dict



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