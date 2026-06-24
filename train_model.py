
import numpy as np
from PIL import Image
import os
import pickle

# ============================================
# CNN FROM SCRATCH (NumPy only - No TensorFlow)
# ============================================

class SimpleCNN:
    """Simple CNN for image classification"""

    def __init__(self, input_size, num_classes=3):
        self.input_size = input_size
        self.num_classes = num_classes
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(64, input_size) * 0.01
        self.b1 = np.zeros((64, 1))
        self.W2 = np.random.randn(num_classes, 64) * 0.01
        self.b2 = np.zeros((num_classes, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward(self, X):
        """Forward pass"""
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y, learning_rate=0.01):
        """Backward pass"""
        m = X.shape[1]

        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, Y, epochs=100, learning_rate=0.01):
        """Train the model"""
        for i in range(epochs):
            A2 = self.forward(X)
            self.backward(X, Y, learning_rate)

            if i % 20 == 0:
                loss = -np.sum(Y * np.log(A2 + 1e-8)) / X.shape[1]
                predictions = np.argmax(A2, axis=0)
                labels = np.argmax(Y, axis=0)
                accuracy = np.mean(predictions == labels)
                print(f"Epoch {i}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")

    def predict(self, X):
        """Predict class"""
        A2 = self.forward(X)
        return np.argmax(A2, axis=0), A2

    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}, f)

    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.W1 = params['W1']
            self.b1 = params['b1']
            self.W2 = params['W2']
            self.b2 = params['b2']


def preprocess_image(image_path, size=(64, 64)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32)
    # Flatten and normalize
    arr = arr.flatten() / 255.0
    return arr.reshape(-1, 1)


def load_dataset(data_dir):
    """Load dataset from folders"""
    classes = ['empty_cart', 'filled_cart', 'not_a_cart']
    class_map = {cls: i for i, cls in enumerate(classes)}

    X = []
    Y = []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                features = preprocess_image(img_path)
                X.append(features)

                # One-hot encoding
                label = np.zeros((3, 1))
                label[class_map[cls]] = 1
                Y.append(label)

    X = np.hstack(X)
    Y = np.hstack(Y)

    return X, Y, classes


if __name__ == "__main__":
    print("Training CNN model...")

    # Load training data
    X_train, Y_train, classes = load_dataset("dataset/train")
    print(f"Training samples: {X_train.shape[1]}")

    # Create and train model
    model = SimpleCNN(input_size=X_train.shape[0], num_classes=3)
    model.train(X_train, Y_train, epochs=200, learning_rate=0.1)

    # Save model
    model.save("model/cnn_model.pkl")
    print("Model saved to model/cnn_model.pkl")

    # Test on validation set
    print("\nValidation Results:")
    X_val, Y_val, _ = load_dataset("dataset/validation")
    predictions, probs = model.predict(X_val)

    for i, cls in enumerate(classes):
        print(f"  {cls}: {np.sum(predictions == i)} predictions")
