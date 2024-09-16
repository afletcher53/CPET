"""
SHAP values intprate the impact of having a certain value for a given feature in conmparison to the preidction wee would make if that feature took some baseline value.
"""

# Load up the mortality@30 data
from data_preparation import load_and_prepare_data


X_bxb, X_cat, y = load_and_prepare_data(30)

print(X_bxb.shape)