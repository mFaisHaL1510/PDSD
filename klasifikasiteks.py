import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['Text'] = train_df['Title'] + ' ' + train_df['Description']
test_df['Text'] = test_df['Title'] + ' ' + test_df['Description']

train_df['Text'] = train_df['Text'].fillna('')
test_df['Text'] = test_df['Text'].fillna('')

X_train = train_df['Text'].values
y_train = train_df['Class Index'].values - 1

X_test = test_df['Text'].values
y_test = test_df['Class Index'].values - 1

# Token
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# membagi data latih
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_pad, y_train, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback untuk menghentikan pelatihan
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90:
            print("\nReached 90% accuracy, stopping training!")
            self.model.stop_training = True

callback = MyCallback()

# Latih model
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[callback]
)

# Evaluasi
test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# latihan dan validasi
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Fungsi untuk memprediksi kelas dari teks input
def predict_text_class(text, model, tokenizer, max_len):
    # Preprocess teks input
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len, padding='post')

    # Prediksi kelas
    prediction = model.predict(text_pad, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0] + 1

    return predicted_class, prediction

# Input teks dari pengguna
input_text = input("Masukkan teks untuk diprediksi: ")

# Prediksi kelas
predicted_class, prediction_prob = predict_text_class(input_text, model, tokenizer, max_len)
print(f"Teks: '{input_text}'")
print(f"Prediksi Kelas: {predicted_class}")
print(f"Probabilitas Prediksi: {prediction_prob}")

# Evaluasi akurasi model pada data uji (test.csv)
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nAkurasi Model pada Data Uji (test.csv): {test_acc * 100:.2f}%")

# Jumlah sampel dalam data latih
jumlah_sampel_train = len(train_df)
print(f"Jumlah sampel dalam data latih: {jumlah_sampel_train}")

# Jumlah sampel dalam data uji
jumlah_sampel_test = len(test_df)
print(f"Jumlah sampel dalam data uji: {jumlah_sampel_test}")

# Total sampel
total_sampel = jumlah_sampel_train + jumlah_sampel_test
print(f"Total sampel: {total_sampel}")

# Jumlah kelas dalam data latih
jumlah_kelas_train = train_df['Class Index'].nunique()
print(f"Jumlah kelas dalam data latih: {jumlah_kelas_train}")

# Jumlah kelas dalam data uji
jumlah_kelas_test = test_df['Class Index'].nunique()
print(f"Jumlah kelas dalam data uji: {jumlah_kelas_test}")

# Daftar kelas unik
daftar_kelas = train_df['Class Index'].unique()
print(f"Daftar kelas: {daftar_kelas}")

# Distribusi kelas dalam data latih
distribusi_kelas_train = train_df['Class Index'].value_counts()
print("Distribusi kelas dalam data latih:")
print(distribusi_kelas_train)

# Distribusi kelas dalam data uji
distribusi_kelas_test = test_df['Class Index'].value_counts()
print("\nDistribusi kelas dalam data uji:")
print(distribusi_kelas_test)

# Ringkasan dataset latih
print("Ringkasan data latih:")
print(train_df.describe())

# Ringkasan dataset uji
print("\nRingkasan data uji:")
print(test_df.describe())

