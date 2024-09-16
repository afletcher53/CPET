import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
import torch
from models.models import load_and_preprocess_data, DNNModel

# Load and preprocess data
days = [30] 
directory = f"data/ml_inputs/mortality_{days[0]}"
(
    X_bxb_train, X_cat_train, y_train,
    X_bxb_val, X_cat_val, y_val,
    X_bxb_test, X_cat_test, y_test,
) = load_and_preprocess_data(directory=directory)

# Load up the saved DNN model
dnn_model = DNNModel(X_cat_train.shape[1], [64,32], 1)
dnn_model.load_state_dict(torch.load(f"saved_models/mortality/{days[0]}/dnn_model.pth"))
dnn_model.eval()



# Assuming previous code for data loading and model setup remains the same

# Prepare the data
background = torch.tensor(X_cat_train[:100], dtype=torch.float32)
test_samples = torch.tensor(X_cat_val, dtype=torch.float32)

# Create the SHAP explainer
e = shap.DeepExplainer(dnn_model, background)

# Compute SHAP values
shap_values = e.shap_values(test_samples)

# Get feature names
cat_feature_names = [
    "OUES", "VE_VO2_SLOPE", "VO2_WORK_SLOPE", "CHRONOTROPIC_INDEX",
    "EXPROTOCOL", "HEIGHT", "WEIGHT", "SEX", "BSA", "EXERCISE_TIME",
    "ETHNICITY", "IMD_SCORE", "PLANNEDOPTYPE"
]

# Reshape shap_values if necessary
shap_values = np.array(shap_values).squeeze()

# Get feature importance
feature_importance = np.abs(shap_values).mean(0)
feature_importance_df = pd.DataFrame({
    'feature': cat_feature_names,
    'importance': feature_importance
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print("shap_values shape:", shap_values.shape)
print("X_cat_val shape:", X_cat_val.shape)
print("feature_importance shape:", feature_importance.shape)


shap.summary_plot(shap_values, X_cat_val, feature_names=cat_feature_names)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import shap
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from models.models import load_and_preprocess_data, SuperLearner

# # Load and preprocess data
# days = [30] 
# directory = f"data/ml_inputs/mortality_{days[0]}"
# (
#     X_bxb_train, X_cat_train, y_train,
#     X_bxb_val, X_cat_val, y_val,
#     X_bxb_test, X_cat_test, y_test,
# ) = load_and_preprocess_data(directory=directory)

# # Limit to 100 training examples
# X_cat_train = X_cat_train[:100]
# y_train = y_train[:100]

# # Create base models
# base_models = [
#     make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
#     make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42)),
#     make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
#     make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
# ]

# # Create SuperLearner
# superlearner = SuperLearner(base_models, meta_model=LogisticRegression())
# superlearner.fit(X_cat_train, y_train)

# # Get feature names
# cat_feature_names = [
#     "OUES", "VE_VO2_SLOPE", "VO2_WORK_SLOPE", "CHRONOTROPIC_INDEX",
#     "EXPROTOCOL", "HEIGHT", "WEIGHT", "SEX", "BSA", "EXERCISE_TIME",
#     "ETHNICITY", "IMD_SCORE", "PLANNEDOPTYPE"
# ]

# # Create base models
# base_models = [
#     make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
#     make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42)),
#     make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
#     make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
# ]

# # Create and train SuperLearner
# superlearner = SuperLearner(base_models, meta_model=LogisticRegression())
# superlearner.fit(X_cat_train, y_train)

# # Define a custom prediction function for the SuperLearner
# def superlearner_predict(X):
#     base_predictions = []
#     for model in superlearner.base_models:
#         # Get predictions from each base model
#         base_pred = model.predict_proba(X)[:, 1]  # Probability of the positive class
#         base_predictions.append(base_pred)
#     # Stack base model predictions as input features for the meta-model
#     meta_features = np.column_stack(base_predictions)
#     # Get final predictions from the meta-model
#     final_pred = superlearner.meta_model.predict_proba(meta_features)[:, 1]
#     return final_pred

# # Initialize SHAP KernelExplainer with a background sample
# background_sample = shap.sample(X_cat_train, 50)  # Smaller sample for efficiency
# explainer = shap.KernelExplainer(superlearner_predict, background_sample)

# # Compute SHAP values for the validation set
# shap_values = explainer.shap_values(X_cat_val)

# # Plot SHAP summary plot
# shap.summary_plot(shap_values, X_cat_val, feature_names=cat_feature_names)