import numpy as np
from dl_dataset_manager import DLManager, MortalityRecord
from sklearn.preprocessing import StandardScaler

def prepare_data(records):
    # Prepare lists to hold our data
    X_bxb_list = []
    X_cat_list = []
    y_list = []
    
    # Process each record
    for record in records:
        # Process breath-by-breath data for LSTM
        X_bxb = record.x_bxb.values
        X_bxb_list.append(X_bxb)
        
        # Process categorical data for deep network
        X_cat = record.x_cat.values.flatten()  # Flatten in case it's 2D
        X_cat_list.append(X_cat)
        
        # Add the outcome
        y_list.append(record.y)
    
    # Convert lists to numpy arrays
    X_bxb_array = np.array(X_bxb_list)
    X_cat_array = np.array(X_cat_list)
    y_array = np.array(y_list)
    
    # Process categorical data
    # EXPROTOCOL: just get the first two characters
    X_cat_array[:, 4] = np.array([str(x)[:2] for x in X_cat_array[:, 4]])
    
    # Sex: convert to binary (assuming 'Male' = 1, 'Female' = 0)
    X_cat_array[:, 7] = np.array([1 if str(x).lower() == 'Male' else 0 for x in X_cat_array[:, 7]])
    
    # Convert to float for scaling
    X_cat_array = X_cat_array.astype(float)
    
    scaler = StandardScaler()
    X_cat_normalized = scaler.fit_transform(X_cat_array)
    
    # Reshape X_bxb_array if necessary
    # Assuming each breath-by-breath record has the same number of time steps
    if len(X_bxb_array.shape) == 3:
        num_samples, num_timesteps, num_features = X_bxb_array.shape
    else:
        num_samples, num_features = X_bxb_array.shape
        num_timesteps = 1
        X_bxb_array = X_bxb_array.reshape(num_samples, num_timesteps, num_features)
    
    return X_bxb_array, X_cat_normalized, y_array

# Usage
mortality_manager = DLManager[MortalityRecord](MortalityRecord)
records = mortality_manager.load_data()
X_bxb, X_cat, y = prepare_data(records)
print(f"Breath-by-breath data shape: {X_bxb.shape}")
print(f"Categorical data shape: {X_cat.shape}")
print(f"Output shape: {y.shape}")

# split the data into train test validation 

from sklearn.model_selection import train_test_split

X_bxb_train, X_bxb_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(X_bxb, X_cat, y, test_size=0.2, random_state=42)

X_bxb_train, X_bxb_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_bxb_train, X_cat_train, y_train, test_size=0.2, random_state=42)

# save the data

np.save('X_bxb_train.npy', X_bxb_train)
np.save('X_bxb_test.npy', X_bxb_test)

np.save('X_cat_train.npy', X_cat_train)
np.save('X_cat_test.npy', X_cat_test)

np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

np.save('X_bxb_val.npy', X_bxb_val)
np.save('X_cat_val.npy', X_cat_val)

np.save('y_val.npy', y_val)
