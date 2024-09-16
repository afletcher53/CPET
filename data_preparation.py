import logging
import os
import numpy as np
from sklearn.calibration import LabelEncoder
from Classes.dl_dataset_manager import DLManager, MortalityRecord
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./logs/main.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def prepare_data(records):
    X_bxb_list = []
    X_cat_list = []
    y_list = []

    for record in records:
        X_bxb = record.x_bxb.values
        X_bxb_list.append(X_bxb)

        X_cat = record.x_cat.values.flatten()
        X_cat_list.append(X_cat)

        y_list.append(record.y)

    X_bxb_array = np.array(X_bxb_list)
    X_cat_array = np.array(X_cat_list)
    y_array = np.array(y_list)

    X_cat_array[:, 4] = np.array([str(x)[:2] for x in X_cat_array[:, 4]])
    X_cat_array[:, 7] = np.array(
        [1 if str(x).lower() == "Male" else 0 for x in X_cat_array[:, 7]]
    )

    le_ethnicity = LabelEncoder()
    le_surgery = LabelEncoder()

    X_cat_array[:, 10] = le_ethnicity.fit_transform(X_cat_array[:, 10])
    X_cat_array[:, 12] = le_surgery.fit_transform(X_cat_array[:, 12])
    

    X_cat_array = X_cat_array.astype(float)
    normalizer_cat = MinMaxScaler()

    scaler = StandardScaler()
    X_cat_normalized = scaler.fit_transform(X_cat_array)
    X_cat_normalized = normalizer_cat.fit_transform(X_cat_array)


    if len(X_bxb_array.shape) == 3:
        num_samples, num_timesteps, num_features = X_bxb_array.shape
    else:
        num_samples, num_features = X_bxb_array.shape
        num_timesteps = 1
        X_bxb_array = X_bxb_array.reshape(num_samples, num_timesteps, num_features)

    return X_bxb_array, X_cat_normalized, y_array

def load_and_prepare_data(days):
    mortality_manager = DLManager[MortalityRecord](MortalityRecord, days=days)
    records = mortality_manager.load_data()
    return prepare_data(records)

def split_and_save_data(X_bxb, X_cat, y, output_folder, test_size=0.2, val_size=0.2, random_state=42):
    X_bxb_train, X_bxb_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_bxb, X_cat, y, test_size=test_size, random_state=random_state
    )

    X_bxb_train, X_bxb_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_bxb_train, X_cat_train, y_train, test_size=val_size, random_state=random_state
    )

    np.save(os.path.join(output_folder, "X_bxb_train.npy"), X_bxb_train)
    np.save(os.path.join(output_folder, "X_bxb_test.npy"), X_bxb_test)
    np.save(os.path.join(output_folder, "X_cat_train.npy"), X_cat_train)
    np.save(os.path.join(output_folder, "X_cat_test.npy"), X_cat_test)
    np.save(os.path.join(output_folder, "y_train.npy"), y_train)
    np.save(os.path.join(output_folder, "y_test.npy"), y_test)
    np.save(os.path.join(output_folder, "X_bxb_val.npy"), X_bxb_val)
    np.save(os.path.join(output_folder, "X_cat_val.npy"), X_cat_val)
    np.save(os.path.join(output_folder, "y_val.npy"), y_val)

    return X_bxb_train, X_bxb_test, X_cat_train, X_cat_test, y_train, y_test, X_bxb_val, X_cat_val, y_val

def generate_mortality_data(days=365, output_folder=None):
    X_bxb, X_cat, y = load_and_prepare_data(days)
    logger.info(f"Breath-by-breath data shape: {X_bxb.shape}")
    logger.info(f"Categorical data shape: {X_cat.shape}")
    logger.info(f"Output shape: {y.shape}")
    
    if output_folder is None:
        output_folder = os.path.join("data", "ml_inputs", f"mortality_{days}")
    os.makedirs(output_folder, exist_ok=True)
    
    split_data = split_and_save_data(X_bxb, X_cat, y, output_folder)
    logger.info(f"Data has been split and saved in {output_folder}.")


    
    return split_data

if __name__ == "__main__":
    generate_mortality_data()

