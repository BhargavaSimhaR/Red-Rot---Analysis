# iimport numpy as np

# Plot training & validation accuracy values as bar plots
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
epochs = np.arange(len(history.history['accuracy']))
width = 0.3

plt.bar(epochs - width/2, history.history['accuracy'], width=width, label='Train Accuracy')
plt.bar(epochs + width/2, history.history['val_accuracy'], width=width, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend(loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.bar(epochs - width/2, history.history['loss'], width=width, label='Train Loss')
plt.bar(epochs + width/2, history.history['val_loss'], width=width, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend(loc='upper left')

# Show the plots
plt.tight_layout()
plt.show()
