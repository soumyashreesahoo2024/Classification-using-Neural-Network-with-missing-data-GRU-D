import tensorflow as tf
import math
import pandas as pd
import numpy as np
from tensorflow.keras.metrics import SparseCategoricalAccuracy
# import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, GRU, Dense, Masking, TimeDistributed
import os
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import resample
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages


from GRUD_without_temporaldecay import GRUD      # as per actual GRUD temporal decay factors are important  but this is without using temporal decay                   
#from GRUD_with_temporaldecay import GRUD        # Use this if want to incorporate temporal decay for GRU-D







# Mask: A binary matrix where 1 indicates that the value is observed (not NaN), and 0 indicates a missing value.

# Delta: A matrix where each entry represents the time gap since the last observed value. If a value is observed,
#  the delta is 1. If a value is missing, the delta is incremented by the delta of the previous time step.

#=========================================================================== 


# # Define functions for cleaning and padding
# def clean_time_steps(time_steps):
#     cleaned = []
#     for ts in time_steps:
#         if isinstance(ts, list):
#             # Replace NaNs with None and filter out invalid values
#             # cleaned_ts = [x if not pd.isna(x) else np.nan for x in ts]
#             cleaned_ts = [0 if pd.isna(x) else x for x in ts]
#             cleaned.append(cleaned_ts)
#     return cleaned



def pad_sequences(sequences, max_length):
    num_samples = len(sequences)
    padded_sequences = np.full((num_samples, 7, max_length), np.nan)
    
    for i, seq in enumerate(sequences):
        if isinstance(seq, list) or isinstance(seq, np.ndarray):
            seq = np.array(seq)
            seq_length = len(seq)
            # Pad the sequence to fit into the (max_length, max_length) shape
            if seq_length > max_length:
                seq = seq[:max_length]
                seq_length = max_length
            
            # Create a 2D representation (if seq is 1D)
            seq_2d = np.zeros((max_length, max_length))
            seq_2d[:seq_length, :seq_length] = np.expand_dims(seq, axis=-1)  # Expand dims if needed
            
            padded_sequences[i] = seq_2d
    
    return padded_sequences



# def pad_time_steps_v0(time_steps, max_length):
#     padded_time_steps = np.full((len(time_steps), max_length), np.nan)
#     for i, ts in enumerate(time_steps):
#         if isinstance(ts, list):
#             if len(ts) > max_length:
#                 ts = ts[:max_length]
#             padded_time_steps[i, :len(ts)] = ts
#     return padded_time_steps


# def pad_time_steps(time_steps):
#     padded_time_steps = np.full((len(time_steps), max_length), np.nan)
#     for i, ts in enumerate(time_steps):
#         if isinstance(ts, list):
#             if len(ts) > max_length:
#                 ts = ts[:max_length]
#             padded_time_steps[i, :len(ts)] = ts
#     return padded_time_steps


def generate_mask(padded_sequences):
    mask = np.isnan(padded_sequences)
    mask = ~mask  # Invert mask: 1 for valid elements, 0 for NaNs
    return mask.astype(int)


# def create_padded_time_steps_v0(padded_features):
#     # Initialize padded_time_steps with zeros
#     padded_time_steps = np.zeros_like(padded_features)
    
#     # Iterate through each user sequence
#     for i in range(padded_features.shape[0]):
#         # Start with 0 for the first time step
#         padded_time_steps[i, 0] = 0
        
#         # Iterate through each time step
#         for j in range(1, padded_features.shape[1]):
#             if j<7:
#                 padded_time_steps[i, j] = padded_time_steps[i, j - 1] + 1
#     return padded_time_steps




def create_padded_time_steps(padded_features):
    num_samples, time_steps_length, _ = padded_features.shape
    padded_time_steps = np.zeros((num_samples, time_steps_length, 1))
    for i in range(num_samples):
        padded_time_steps[i, 0, 0] = 0
        for j in range(1, time_steps_length):
            padded_time_steps[i, j] = padded_time_steps[i, j - 1] + 1
    return padded_time_steps




# def generate_delta_v0(padded_sequences, mask, padded_time_steps):
#     # print("Padded Sequences Shape:", padded_sequences.shape)
#     # print("Mask Shape:", mask.shape)
#     # print("Padded Time Steps Shape:", padded_time_steps.shape)
    
#     delta = np.full_like(padded_sequences, np.nan, dtype=float)    
#     for i in range(padded_sequences.shape[0]):  # Iterate over each user
#         delta[i, 0] = 0  # Set the first delta to 0

#         for j in range(1, padded_sequences.shape[1]):  # Iterate over each time step starting from the second one
#           if j<7:
#             if mask[i, j]:  # Current value is observed (mask = 1)
#                 delta[i, j] = padded_time_steps[i, j] - padded_time_steps[i, j - 1]
#             else:  # Current value is missing (mask = 0)
#                 delta[i, j] = padded_time_steps[i, j] - padded_time_steps[i, j - 1] + delta[i, j - 1]
#           else:
#             continue
#     return delta


def generate_delta(padded_sequences, mask, padded_time_steps):
    num_samples, time_steps_length, _ = padded_sequences.shape
    delta = np.full_like(padded_sequences, np.nan, dtype=float)
    
    for i in range(num_samples):
        delta[i, 0] = 0  # Set the first delta to 0
        
        for j in range(1, time_steps_length):
            if mask[i, j, 0]:  # Ensure mask[i, j] is a scalar
                delta[i, j, 0] = padded_time_steps[i, j, 0] - padded_time_steps[i, j - 1, 0]
            else:
                delta[i, j, 0] = padded_time_steps[i, j, 0] - padded_time_steps[i, j - 1, 0] + delta[i, j - 1, 0]
    
    delta = np.nan_to_num(delta, nan=0.0)
    return delta




#============================================================================================

# Load your data
dfactual = pd.read_csv('vectors_Qv_vlen1_updated_raw_mood_mean.csv')

label_counts = dfactual['QIDSlabel'].value_counts()
print("Class distribution before upsampling:")
print(label_counts)

df_majority = dfactual[dfactual['QIDSlabel'] == 'not improved']
df_minority = dfactual[dfactual['QIDSlabel'] == 'improved']
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42)  # for reproducibility

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print("Class distribution after upsampling:")
print(df_upsampled['QIDSlabel'].value_counts())
df=df_upsampled


## if not upsampling data
df=dfactual



#==========================================================================================
dmood_columns = ['dmood0', 'dmood1', 'dmood2', 'dmood3', 'dmood4', 'dmood5', 'dmood6']
df[dmood_columns] = df[dmood_columns].replace('ok', np.nan)
df[dmood_columns] = df[dmood_columns].apply(pd.to_numeric, errors='coerce')

# Add new column 'timestep' with the difference between dates 'D0'
df['D0'] = pd.to_datetime(df['D0'], errors='coerce')
df['D0'] = pd.to_datetime(df['D0'], format='%m/%d/%Y')
df = df.sort_values(by=['userid', 'D0'])
df['timestep'] = df.groupby('userid')['D0'].diff().dt.days
df['timestep'] = df['timestep'].fillna(0)

# Group by userid and extract features, masks, and deltas
user_data = df.groupby('userid')
time_steps = df.groupby('userid')['timestep'].apply(lambda x: x.tolist()).tolist()

# max_length = max(len(ts) for ts in time_steps)# Determine max length for padding


#==============================================================


def process_user_data_1(user_data, max_length):
    # time_steps = user_data['timestep'].tolist()
    features = user_data[dmood_columns].values.tolist()
    time_steps = len(features[0])
    
    n_samples = len(user_data)  # Number of samples in your dataset
    n_steps = time_steps  # Number of time steps in each sample
    n_features = 1  # Number of features
    
    # Clean and pad features and time_steps
    X=user_data[dmood_columns]
    X_raw = X.values[:, :, np.newaxis]

    
    # X_filledLOCF_df = X.fillna(method='ffill', axis=1)
    X_filledLOCF_df = X.ffill(axis=1)  # Forward fill along the rows
    X_filledLOCF = X_filledLOCF_df.values[:, :, np.newaxis]  # correct
    empirical_mean = np.mean(X.values, axis=1, keepdims=True)
    
    padded_features = pad_sequences(features, max_length)
    padded_time_steps = create_padded_time_steps(padded_features)
    mask = generate_mask(padded_features)
    delta = generate_delta(padded_features, mask, padded_time_steps)
    
    
    # print("In process_user_data==={}, {} ,{}".format(padded_features.shape,mask.shape,delta.shape))
    return padded_features, mask, delta, X_raw, X_filledLOCF, empirical_mean




#================================================================
def fillnan(X_train):
    # Forward fill along the time steps (axis 1)
    X_train_filled = np.where(np.isnan(X_train), np.nan, X_train)
    
    # Iterate over each sample and apply forward fill
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if np.isnan(X_train_filled[i, j]):
                X_train_filled[i, j] = X_train_filled[i, j - 1] if j > 0 else 0  # Use 0 or a designated fill value
    
    # Handle any NaNs that were at the beginning (optional)
    X_train_filled = np.where(np.isnan(X_train_filled), np.nanmean(X_train_filled, axis=1)[:, np.newaxis], X_train_filled)    
    return X_train_filled



def fillnan_LOCF(X_filledLOCF_train):
    # Mean imputation across the time steps (axis 1)
    X_filledLOCF_train_filled = np.where(np.isnan(X_filledLOCF_train), np.nan, X_filledLOCF_train)
    
    # Replace NaNs with the mean of each feature
    mean_values = np.nanmean(X_filledLOCF_train_filled, axis=1)
    mean_values = np.nan_to_num(mean_values, nan=1)
    for i in range(X_filledLOCF_train_filled.shape[0]):
        for j in range(X_filledLOCF_train_filled.shape[1]):
            if np.isnan(X_filledLOCF_train_filled[i, j]):
                X_filledLOCF_train_filled[i, j] = mean_values[i]
    
    return X_filledLOCF_train_filled



# def fillnan_LOCF(X_filledLOCF_train):
#     X_filledLOCF_train_filled = np.where(np.isnan(X_filledLOCF_train), np.nan, X_filledLOCF_train)

#     # Check which rows are all NaN
#     all_nan_rows = np.isnan(X_filledLOCF_train_filled).all(axis=(1,2))
    
#     # Expand dimensions of all_nan_rows to match the shape of X_filledLOCF_train_filled
#     all_nan_rows_expanded = all_nan_rows[:, np.newaxis, np.newaxis]
    
#     # Use the boolean mask to set all NaN rows to a default value
#     if all_nan_rows.any():
#         print("Warning: There are rows with all NaN values. Filling with default value 0.")
#         # Convert boolean mask to 3D boolean mask
#         mask = all_nan_rows_expanded
#         # Set all rows that are all NaN to zero or another default value
#         X_filledLOCF_train_filled[mask] = 0
    
#     # Replace NaNs with the mean of each feature
#     mean_values = np.nanmean(X_filledLOCF_train_filled, axis=1)
#     mean_values = np.nan_to_num(mean_values, nan=1)
#     for i in range(X_filledLOCF_train_filled.shape[0]):
#         for j in range(X_filledLOCF_train_filled.shape[1]):
#             if np.isnan(X_filledLOCF_train_filled[i, j]):
#                 X_filledLOCF_train_filled[i, j] = mean_values[i]
    
#     return X_filledLOCF_train_filled
#=====================================================================

data=df

max_length=1   # per sequence i have 1 fetaure

output_size = len(np.unique(df['QIDSlabel']))
# input_shape = (max_length, len(dmood_columns))
input_shape=7
n_steps= len(dmood_columns)
n_features=1

epochs = [1]  # List of epochs for demonstration
lr=[0.001]
batch_sizes = [8]
hidden_sizes=[8]
num_layers = 2
dropout_rate = 0.1


predicted=[]
actual=[]
f1_scores = []
prec=[]
rec=[]
acc=[]
tp_list = []
tn_list = []
fp_list = []
fn_list = []


# df_metrics = pd.DataFrame(columns=['epoch','F1','Precision','Recall','Accuracy'])

for hidden_size in hidden_sizes:
    for batch_size in batch_sizes:
        for lrn in lr:
            # model = GRUD(input_size=input_shape, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout_rate)                                 
            for e in epochs:
                model = GRUD(n_steps=n_steps, n_features=n_features, n_classes=2, rnn_hidden_size=hidden_size, batch_size=batch_size, epochs=e)
                for user in data['userid'].unique():
                    print(f"Processing user: {user}")
                    
                    # Split data into training and testing sets
                    train_data = data[data['userid'] != user]
                    test_data = data[data['userid'] == user]
                    # print(f"Training data size: {train_data.shape}")
                    # print(f"Testing data size: {test_data.shape}")

                    
                    train_features_user, train_masks_user, train_deltas_user, X_train, X_filledLOCF_train, empirical_mean_train = process_user_data_1(train_data, max_length)
                    test_features_user, test_masks_user, test_deltas_user, X_test, X_filledLOCF_test, empirical_mean_test = process_user_data_1(test_data, max_length)
                    
                    train_features_user = np.array(train_features_user)
                    train_masks_user = np.array(train_masks_user)
                    train_deltas_user = np.array(train_deltas_user)
                    
                    test_features_user = np.array(test_features_user)
                    test_masks_user = np.array(test_masks_user)
                    test_deltas_user = np.array(test_deltas_user)
                                        
                    # Debugging: Print shapes and types
                    # print(f"train_features_user shape: {train_features_user.shape}, dtype: {train_features_user.dtype}")
                    # print(f"train_masks_user shape: {train_masks_user.shape}, dtype: {train_masks_user.dtype}")
                    # print(f"train_deltas_user shape: {train_deltas_user.shape}, dtype: {train_deltas_user.dtype}")
                    
                    # print(f"\n\ntest_features_user shape: {test_features_user.shape}, dtype: {test_features_user.dtype}")
                    # print(f"test_masks_user shape: {test_masks_user.shape}, dtype: {test_masks_user.dtype}")
                    # print(f"test_deltas_user shape: {test_deltas_user.shape}, dtype: {test_deltas_user.dtype}")

                    label_mapping = {'not improved': 0, 'improved': 1}

                    train_labels = [label_mapping[label] for label in train_data['QIDSlabel'].tolist()]
                    test_labels = [label_mapping[label] for label in test_data['QIDSlabel'].tolist()]
                    ## Integer labels for binary or multi-class classification
                    # train_labels_int = np.array([label_mapping[label] for label in train_data['QIDSlabel'].tolist()], dtype=np.int64)
                    # test_labels_int = np.array([label_mapping[label] for label in test_data['QIDSlabel'].tolist()], dtype=np.int64)
                    train_labels_int = np.expand_dims(np.array([label_mapping[label] for label in train_data['QIDSlabel'].tolist()]), axis=-1)
                    test_labels_int = np.expand_dims(np.array([label_mapping[label] for label in test_data['QIDSlabel'].tolist()]), axis=-1)

                    # Combine the data into tuples for input to model
                    input_train = (X_train, X_filledLOCF_train, train_masks_user, train_deltas_user, empirical_mean_train)
                    input_test = (X_test, X_filledLOCF_test, test_masks_user, test_deltas_user, empirical_mean_test)
                    
                    input_val = input_test
                    test_labels_val = test_labels_int
                    
                    # train_data = (input_train, train_labels_int)
                    # val_data = (input_val, test_labels_val)
                    
                    ## imputation of nan  in any type of input================================================                    
                    X_train_filled=fillnan(X_train)
                    X_filledLOCF_train_filled =fillnan_LOCF(X_filledLOCF_train)
                    
                    
                    input_train = (X_train_filled, X_filledLOCF_train_filled, train_masks_user, train_deltas_user, empirical_mean_train)
                    input_test = (X_test, X_filledLOCF_test, test_masks_user, test_deltas_user, empirical_mean_test)
                    #======================================================================


                    # Train the model
                    print("Starting model training...")    
                    
                    # Debugging: Print shapes and types
                    # Print shapes and types for input_train
                    # print("input_train:")
                    # print(f"X_train_filled shape: {X_train_filled.shape}, type: {type(X_train_filled)}")
                    # print(f"X_filledLOCF_train_filled shape: {X_filledLOCF_train_filled.shape}, type: {type(X_filledLOCF_train_filled)}")
                    # print(f"train_masks_user shape: {train_masks_user.shape}, type: {type(train_masks_user)}")
                    # print(f"train_deltas_user shape: {train_deltas_user.shape}, type: {type(train_deltas_user)}")
                    # print(f"empirical_mean_train shape: {empirical_mean_train.shape}, type: {type(empirical_mean_train)}")
                    
                    # # Print shapes and types for input_test
                    # print("input_test:")
                    # print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
                    # print(f"X_filledLOCF_test shape: {X_filledLOCF_test.shape}, type: {type(X_filledLOCF_test)}")
                    # print(f"test_masks_user shape: {test_masks_user.shape}, type: {type(test_masks_user)}")
                    # print(f"test_deltas_user shape: {test_deltas_user.shape}, type: {type(test_deltas_user)}")
                    # print(f"empirical_mean_test shape: {empirical_mean_test.shape}, type: {type(empirical_mean_test)}")

                    


                    ### If you have validation data, pass it like this:
                    history = model.fit(input_train, train_labels_int, input_val, test_labels_val)

                    # Predict on the test data
                    test_predictions = model.predict(input_test)
                    
                    # Classify the test data
                    predicted_labels = model.classify((input_test))  # predicted labels
                    actual_labels = test_labels_int

                    # Calculate F1 Score, Accuracy, Precision, Recall, and Classification Report
                    f1 = f1_score(actual_labels, predicted_labels, average='binary')  # Use 'binary' for binary classification
                    
                    print("F1 Score: ", f1)

                    predicted.append(predicted_labels.flatten())
                    actual.append(actual_labels.flatten())
                    # predicted = np.concatenate(predicted)
                    # actual = np.concatenate(actual)
                    
                    f1_scores.append(f1)


                mean_f1_score = np.mean(f1_scores)
                print("Mean F1 Score over all users: ", mean_f1_score)
                
                predicted = np.concatenate(predicted).flatten()
                actual = np.concatenate(actual).flatten()
                
                precision = precision_score(actual, predicted)
                recall = recall_score(actual, predicted)
                accuracy = accuracy_score(actual, predicted)
                tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
                specificity = tn / (tn+fp)
                rocauc=roc_auc_score(actual, predicted)

                df_metrics = pd.read_csv('result_metrics.csv')
                df_len = len(df_metrics)
                
                df_metrics.loc[df_len, 'RNN']=hidden_size
                df_metrics.loc[df_len, 'batch']=batch_size
                df_metrics.loc[df_len, 'lr']=lrn
                df_metrics.loc[df_len, 'epoch']=e
                df_metrics.loc[df_len, 'f1']=mean_f1_score
                df_metrics.loc[df_len, 'precision']=precision
                df_metrics.loc[df_len, 'recall']=recall
                df_metrics.loc[df_len, 'accuracy']=accuracy
                df_metrics.loc[df_len, 'specificity']=specificity
                df_metrics.loc[df_len, 'roc_auc']=rocauc
                df_metrics.loc[df_len, 'TN']=tn
                df_metrics.loc[df_len, 'TP']=tp
                df_metrics.loc[df_len, 'FN']=fn
                df_metrics.loc[df_len, 'FP']=fp
                df_metrics.loc[df_len, 'Total']=tn+fp+fn+tp
                
                
                df_metrics.to_csv("result_metrics.csv", index=False)
