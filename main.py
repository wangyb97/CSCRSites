import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

AAMap = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# read fasta file
def read_fasta_file(fasta_file=''):
    fp = open(fasta_file, 'r')
    seqslst = []
    seq_name = []
    while True:
        s = fp.readline()
        if not s:
            break
        else:
            if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
            else:
                seq_name.append(s.rstrip('\n'))
    return np.array(seqslst)

def read_label(label_file=''):
    label_ls = open(label_file).readlines()
    ls = []
    for item in label_ls:
        ls.append(int(item))
    return np.array(ls)

# One-hot encoding function
def one_hot_encode(sequences):
    mapping = AAMap
    one_hot_encoded = np.zeros((len(sequences), len(sequences[0]), len(mapping)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq):
            one_hot_encoded[i, j, mapping[nucleotide]] = 1
    return one_hot_encoded


def processFastaFile(fastaFileInput):
    seq = fastaFileInput
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 4])
    for i in range(0, seqLength):
        sequence_vector[i, AAMap[seq[i]]] = 1
    return sequence_vector


def dealwithdata(protein):
    dataX = []
    dataY = []
    with open('/home/wangyubo/dataset/37/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                Kmer = processFastaFile(line)
                dataX.append(Kmer.tolist())
                dataY.append([1])
    with open('/home/wangyubo/dataset/37/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                Kmer = processFastaFile(line)
                dataX.append(Kmer.tolist())  
                dataY.append([0])
    dataX = np.array(dataX)
    dataY = to_categorical(dataY)
    return dataX, dataY

protein = "QKI"
sequences,labels = dealwithdata(protein)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_length, 4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_prob = model.predict(X_test).ravel()
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")

# Save the model
model.save('model.h5')
