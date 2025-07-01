
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

model = Sequential([
    Conv2D(filters = 6, kernel_size = (5,5), strides = (1,1), padding='valid', activation='relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(filters = 16, kernel_size = (5,5), strides = (1,1), padding = 'valid', activation='relu'),
    MaxPooling2D(pool_size = (2,2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
]
)

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
y_train.shape

X_test.shape
y_test.shape

plt.matshow(X_train[0])

X_train.shape


X_train = X_train/255
X_test = X_test / 255
print(X_train.shape)

X_train_flattened = X_train.reshape(-1, 28, 28, 1)
X_test_flattened = X_test.reshape(-1, 28, 28, 1)

history = model.fit(x=X_train_flattened, y=y_train, batch_size=64, epochs=10, validation_data=(X_test_flattened, y_test))

model.evaluate(X_test_flattened, y_test)

model.summary()

# plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])

plt.figure(figsize=(12, 8))

# Subplot for accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout(pad=3.0)  # Add padding between plots

# Subplot for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout(pad=3.0)  # Ensure padding is applied
  

y_predicted = model.predict(X_test_flattened)

# Convert predicted probabilities to class labels
y_pred = np.argmax(y_predicted, axis=1)

## To get performance metrices
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_predicted = model.predict(X_test_flattened)

np.round(y_predicted[100],3)

# To get confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))
plt.grid(False)
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))    
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')            
plt.show()




