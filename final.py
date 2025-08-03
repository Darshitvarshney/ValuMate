import pandas as pd
from sklearn.model_selection  import train_test_split
import numpy as np
import matplotlib.pyplot as plot
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder ,OneHotEncoder,MinMaxScaler,StandardScaler,FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
housing = pd.read_csv(r"C:\Users\darsh\Python\House_Price_Pridiction\Experiments\files\housing.csv")

# Checking distribution first
# housing.hist(bins=50, figsize=(12, 8))


# removing upper caping
housing.housing_median_age.max()
i = housing.housing_median_age != 52.0
df_new  = housing[i]
df_new = df_new[df_new.median_house_value < 500001.0]
df_new.shape

# handling missing values and scaling by creating pipeline

# num pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),('scale', StandardScaler(with_mean=True, with_std=True)),
    ])
# cat pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

# ration transformer
def ratio(X):
    return X[0] / X[:,[1]]
def column_name(function_transformer,get_feature_in):
    return ["ratio"]
ratio_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy='median')),("ratio", FunctionTransformer(func = ratio, feature_names_out=column_name)),("StandardScaler", StandardScaler(with_mean=True))])


# cluster similarity


class Similarity4Cluster(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=0.1, random_state=None):
        # Initialize the settings
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, sample_weight=None, y=None):
        # Create and fit the KMeans model
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # Return similarity of each sample to each cluster center
        return rbf_kernel(X, self.kmeans.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, name=None):
        # Generate feature names like 'similarity with 1 cluster', etc.
        return [f"similarity with {i+1} cluster" for i in range(self.n_clusters)]


# log transformer
log_pipeline = Pipeline([
    ("SimpleImputer", SimpleImputer(strategy='median')),
    ("log", FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),
    ("StandardScaler", StandardScaler(with_mean=True))])

# Define feature name output      multimode distribution

def similarity_clms(function_transformer, get_features_in):
    return ["Similarity With Housing age: 35"]

# Build the pipeline
simil = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("similarity", FunctionTransformer(func=rbf_kernel, 
                                       kw_args=dict(Y=[[35]], gamma=0.1), 
                                       feature_names_out=similarity_clms)),
    ("standardscaler", StandardScaler())
])


# final preprocessing 
# final Preprocessing
preprocessings = ColumnTransformer([
    ("bedrooms", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline, ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline, ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", Similarity4Cluster(), ["latitude", "longitude"]),
    ("cat", cat_pipeline, ["ocean_proximity"]),
    ("simil", simil, ["housing_median_age"]) #("pass","passthrough",["median_house_value"]
    # ("drops", "drop", ["median_house_value"])
], 
remainder = num_pipeline)








