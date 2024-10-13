import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# 1. Load and Preprocess the Dataset
def load_and_preprocess_data(csv_path):
    # Load the dataset
    data = pd.read_csv(csv_path)  # Adjust with your path
    
    # Separate features and labels
    features = data.filter(regex=r'landmark_\d+_(x|y|z|visibility)').values
    labels = data['dance_name'].values
    sample_numbers = data['sample_number'].values
    frame_numbers = data['frame_number'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Group features and labels by sample_number
    grouped_features, grouped_labels, grouped_sample_numbers = group_samples(features_scaled, labels_encoded, sample_numbers)
    
    # Pad sequences to have the same length
    max_seq_length = max([seq.shape[0] for seq in grouped_features])
    grouped_features_padded = pad_sequences(grouped_features, maxlen=max_seq_length, dtype='float32', padding='post', value=0.0)
    
    # Compute variance features for each sample
    variance_features = compute_variance_features(grouped_features, max_seq_length)
    
    return grouped_features_padded, variance_features, grouped_labels, label_encoder, scaler, max_seq_length

def group_samples(features, labels, sample_numbers):
    grouped_features = []
    grouped_labels = []
    grouped_sample_numbers = []
    
    unique_samples = np.unique(sample_numbers)
    for sample in unique_samples:
        indices = np.where(sample_numbers == sample)[0]
        grouped_features.append(features[indices])
        grouped_labels.append(labels[indices][0])  # Assuming one label per sample
        grouped_sample_numbers.append(sample)
    
    return grouped_features, np.array(grouped_labels), np.array(grouped_sample_numbers)

def compute_variance_features(grouped_features, max_seq_length):
    variance_list = []
    for seq in grouped_features:
        # If sequence is shorter than max_seq_length, pad with zeros for variance computation
        if seq.shape[0] < max_seq_length:
            padded_seq = np.vstack([seq, np.zeros((max_seq_length - seq.shape[0], seq.shape[1]))])
        else:
            padded_seq = seq
        # Compute variance across the time axis (frames)
        variance = np.var(padded_seq, axis=0)
        variance_list.append(variance)
    return np.array(variance_list)

# 2. Define the Model with Additional Variance Inputs
def create_lstm_model(input_shape, variance_shape, num_classes, dropout_rate=0.3):
    # Sequence Input
    sequence_input = Input(shape=input_shape, name='sequence_input')
    masked = Masking(mask_value=0.0)(sequence_input)
    lstm_out = LSTM(64, return_sequences=True, activation='relu')(masked)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = LSTM(32, activation='relu')(lstm_out)
    
    # Variance Input
    variance_input = Input(shape=variance_shape, name='variance_input')
    dense_variance = Dense(64, activation='relu')(variance_input)
    dense_variance = Dropout(dropout_rate)(dense_variance)
    
    # Concatenate LSTM and Variance Outputs
    concatenated = concatenate([lstm_out, dense_variance])
    dense_final = Dense(64, activation='relu')(concatenated)
    dense_final = Dropout(dropout_rate)(dense_final)
    output = Dense(num_classes, activation='softmax')(dense_final)
    
    # Define the Model
    model = Model(inputs=[sequence_input, variance_input], outputs=output)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the Model
def train_model(model, X_seq, X_var, y, epochs=20, batch_size=8):
    history = model.fit(
        [X_seq, X_var],
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    return history

# 4. Save the Model and Preprocessing Objects
def save_artifacts(model, scaler, label_encoder, max_seq_length):
    model.save('dance_model.keras')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(max_seq_length, 'max_seq_length.pkl')
    print("Model and preprocessing objects saved successfully.")

# 5. Load the Saved Model and Preprocessors for Predictions
def load_model_and_tools():
    loaded_model = tf.keras.models.load_model('dance_model.keras')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_encoder = joblib.load('label_encoder.pkl')
    max_seq_length = joblib.load('max_seq_length.pkl')
    return loaded_model, loaded_scaler, loaded_encoder, max_seq_length

# 6. Prediction Function for New Samples
def predict_new_sample(model, scaler, encoder, max_seq_length, new_sample_frames):
    """
    Args:
        model: Loaded Keras model.
        scaler: Loaded StandardScaler.
        encoder: Loaded LabelEncoder.
        max_seq_length: Maximum sequence length used during training.
        new_sample_frames: A list or array of frames, where each frame contains landmark features.
                           Shape should be (num_frames, num_features).
    
    Returns:
        Predicted dance label.
    """
    # Convert to numpy array
    new_sample_frames = np.array(new_sample_frames)
    
    # Scale the features
    sample_scaled = scaler.transform(new_sample_frames)
    
    # Pad the sequence
    sample_padded = pad_sequences([sample_scaled], maxlen=max_seq_length, dtype='float32', padding='post', value=0.0)
    
    # Compute variance features
    variance = np.var(sample_scaled, axis=0)
    # If the sequence was padded, append zeros to match the original variance computation
    if new_sample_frames.shape[0] < max_seq_length:
        padding = np.zeros(max_seq_length - new_sample_frames.shape[0], dtype='float32')
        # Not needed here since variance is computed only on actual frames
    variance = variance.reshape(1, -1)  # Reshape for prediction
    
    # Make prediction
    prediction = model.predict([sample_padded, variance])
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# 7. Main Execution Flow
def main():
    # File path to your dataset
    csv_path = 'dance_dataset/all_samples.csv'  # Adjust with your path
    
    # Load and preprocess data
    X_seq, X_var, y, label_encoder, scaler, max_seq_length = load_and_preprocess_data(csv_path)
    
    # Define model parameters
    input_shape = (X_seq.shape[1], X_seq.shape[2])  # (frames, features per frame)
    variance_shape = (X_var.shape[1],)  # Variance features shape
    num_classes = len(np.unique(y))
    
    # Create the model
    model = create_lstm_model(input_shape, variance_shape, num_classes)
    model.summary()
    
    # Train the model
    history = train_model(model, X_seq, X_var, y, epochs=20, batch_size=8)
    
    # Save the model and preprocessors
    save_artifacts(model, scaler, label_encoder, max_seq_length)
    
    # Example Prediction (Uncomment and replace 'new_sample_frames' with actual data)
    """
    # Load the saved model and tools
    loaded_model, loaded_scaler, loaded_encoder, loaded_max_seq_length = load_model_and_tools()
    
    # Define a new sample: list of frames, each frame is a list of landmark features
    # Example: new_sample_frames = [[x0, y0, z0, v0, x1, y1, z1, v1, ..., x31, y31, z31, v31], ...]
    new_sample_frames = np.random.rand(30, 128)  # Replace with actual data (30 frames, 128 features)
    
    # Predict the dance name
    predicted_dance = predict_new_sample(loaded_model, loaded_scaler, loaded_encoder, loaded_max_seq_length, new_sample_frames)
    print(f"Predicted Dance: {predicted_dance}")
    """

if __name__ == "__main__":
    main()
