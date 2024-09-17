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

# Prepare the data
background = torch.tensor(X_cat_train[:100], dtype=torch.float32)
test_samples = torch.tensor(X_cat_val, dtype=torch.float32)

# Create the SHAP explainer
e = shap.DeepExplainer(dnn_model, background)

# Compute SHAP values
shap_values = e.shap_values(test_samples)

# # Get feature names
cat_feature_names = [
    "OUES", "VE_VO2_SLOPE", "VO2_WORK_SLOPE", "CHRONOTROPIC_INDEX",
    "EXPROTOCOL", "HEIGHT", "WEIGHT", "SEX", "BSA", "EXERCISE_TIME",
    "ETHNICITY", "IMD_SCORE", "DATE_OF_CPET_TEST","DATE_OF_OPERATION",
    "OPERATION_SPECIALTY", "OPERATION_SUBCATEGORY", "HAEMOGLOBIN", "WBC",
    "PLATELETS", "SODIUM", "POTASSIUM", "UREA", "CREATININE", "TOTAL PROTEIN",
    "ALBUMIN", "TOTAL BILIRUBIN", "ALP", "ALT", "CALCIUM", "ADJUSTED_CALCIUM",
    "EGFR", "CC_BOOKED_AT_LISTING",
    "CPET_SEX", "AGE", "MI", "IHD", "ANGINA", "CABG_PCI", "HF", "CVD",
    "DIABETES", "CRI", "DYSRHYTHMIA", "HYPERTENSION", "COPD", "ASTHMA",
    "OTHER_LUNG_DISEASE", "BETA_BLOCKER", "ACE_INHIBITORS", "CALCIUM_BLOCKER",
    "DIURETIC", "OTHER_ANTI_HYPERTENSIVE", "ASPIRIN", "CLOPIDOGREL",
    "OTHER_ANTICOAGULANT", "STATIN", "DIGOXIN", "INSULIN", "ORAL_HYPOGLYCAEMIC",

    "INHALERS", "SYSTEMIC_STEROID", "AT", "PEAK_VO2", "VE_VCO2_AT",
    "VO2_HR_AT", "OXYGEN_PULSE", "OXYGVO2_WORKRATE"

],

# Reshape shap_values if necessary
shap_values = np.array(shap_values).squeeze()

print("shap_values shape:", shap_values.shape)
print("X_cat_val shape:", X_cat_val.shape)




def plot_shap_values(shap_values, features, feature_names, max_display=20,  outlier_threshold=None):
    plt.figure(figsize=(12, 8))
    
    # Ensure shap_values is a 2D array
    if len(shap_values.shape) == 3:
        # For multi-class models, we typically want to look at the magnitude of shap values across all classes
        shap_values = np.abs(shap_values).sum(axis=0)
    elif len(shap_values.shape) == 1:
        # If it's a 1D array, reshape it to 2D
        shap_values = shap_values.reshape(1, -1)
    
    if outlier_threshold is not None:
        # Calculate the threshold for outliers
        threshold = np.percentile(np.abs(shap_values), outlier_threshold)
        
        # Create a mask for non-outlier values
        mask = np.abs(shap_values) <= threshold
        
        # Apply the mask to shap_values and features
        shap_values_filtered = np.where(mask, shap_values, threshold * np.sign(shap_values))
        features_filtered = features
    else:
        shap_values_filtered = shap_values
        features_filtered = features

    shap.summary_plot(
        shap_values_filtered, 
        features_filtered, 
        feature_names=feature_names, 
        max_display=max_display, 
        show=False
    )
    
    plt.title("SHAP Summary Plot (Outliers Handled)" if outlier_threshold else "SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()  # Close the plot to free up memory

    print(f"SHAP summary plot saved as 'shap_summary_plot.png'")

# plot_shap_values(shap_values, X_cat_val, feature_names=cat_feature_names, outlier_threshold=95)

# Calculate and print feature importance
# feature_importance = np.abs(shap_values).mean(0)
# feature_importance_df = pd.DataFrame({
#     'feature': cat_feature_names,
#     'importance': feature_importance
# })

# # Sort features by importance
# feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
# print(feature_importance_df)


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
# X_cat_train = X_cat_train[:300]
# y_train = y_train[:300]

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
# # cat_feature_names = [
# #     "OUES", "VE_VO2_SLOPE", "VO2_WORK_SLOPE", "CHRONOTROPIC_INDEX",
# #     "EXPROTOCOL", "HEIGHT", "WEIGHT", "SEX", "BSA", "EXERCISE_TIME",
# #     "ETHNICITY", "IMD_SCORE", "PLANNEDOPTYPE"
# # ]

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
# # shap.summary_plot(shap_values, X_cat_val, feature_names=cat_feature_names)

# plot_shap_values(shap_values, X_cat_val, feature_names=cat_feature_names, outlier_threshold=95)

# # Calculate and print feature importance
# feature_importance = np.abs(shap_values).mean(0)
# feature_importance_df = pd.DataFrame({
#     'feature': cat_feature_names,
#     'importance': feature_importance
# })

# # Sort features by importance
# feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
# print(feature_importance_df)