import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Configurations
BatchSize = 10
NumEpoch = 10

# Paths to datasets
datasets = {
    "SA": {
        "train": 'custom_train_SA.csv',
        "test": 'custom_test_SA.csv'
    },
    "SB": {
        "train": 'custom_train_SB.csv',
        "test": 'custom_test_SB.csv'
    },
    "SC": {
        "train": 'custom_train_SC.csv',
        "test": 'custom_test_SC.csv'
    }
}

def load_and_preprocess_data(train_path, test_path):
    # Load datasets
    print(f"Loading training data from {train_path}")
    print(f"Loading testing data from {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and labels
    X_train = train_data.iloc[:, 0:-2].values
    y_train = np.where(train_data.iloc[:, -2].values == 'normal', 0, 1)
    
    X_test = test_data.iloc[:, 0:-2].values
    y_test = np.where(test_data.iloc[:, -2].values == 'normal', 0, 1)
    
    # One-hot encoding
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [1, 2, 3])],
        remainder='passthrough'
    )
    X_train = np.array(ct.fit_transform(X_train), dtype=float)
    X_test = np.array(ct.transform(X_test), dtype=float)
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_and_train_fnn(X_train, y_train):
    # Initialize the ANN
    classifier = Sequential()
    
    # Add input and first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    
    # Add second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    
    # Add output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    
    # Compile the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print(f"Training model with {NumEpoch} epochs and batch size {BatchSize}")
    classifier_history = classifier.fit(X_train, y_train, batch_size=BatchSize, epochs=NumEpoch, verbose=1)
    
    return classifier, classifier_history

def evaluate_model(classifier, X_test, y_test):
    print("Evaluating model...")
    y_pred = (classifier.predict(X_test) > 0.9).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    loss, accuracy = classifier.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy, cm

def plot_metrics(history, scenario):
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.title(f'{scenario} - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(f'{scenario}_accuracy.png')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.title(f'{scenario} - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(f'{scenario}_loss.png')
    plt.show()

def analyze_confusion_matrix(cm):
    print("Print the Confusion Matrix:")
    print("[ TN, FP ]")
    print("[ FN, TP ]=")
    print(cm)
    
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

results = {}

# Train and evaluate models for each scenario
for scenario, paths in datasets.items():
    print(f"\nScenario {scenario} - Loading and Preprocessing Data")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(paths['train'], paths['test'])
    classifier, classifier_history = build_and_train_fnn(X_train, y_train)
    loss, accuracy, cm = evaluate_model(classifier, X_test, y_test)
    
    results[scenario] = {
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }
    
    plot_metrics(classifier_history, scenario)

# Display results
for scenario, result in results.items():
    print(f"\nScenario {scenario} - Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}")
    analyze_confusion_matrix(result['confusion_matrix'])

# Compare accuracy
accuracies = {scenario: result['accuracy'] for scenario, result in results.items()}
best_scenario = max(accuracies, key=accuracies.get)

print("\nAccuracy Comparison:")
for scenario, accuracy in accuracies.items():
    print(f"{scenario}: {accuracy:.4f}")

print(f"\nThe scenario with the highest accuracy is {best_scenario} with an accuracy of {accuracies[best_scenario]:.4f}")
