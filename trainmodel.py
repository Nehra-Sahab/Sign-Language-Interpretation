from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical          # Used for choosing optimizer and loss function
from keras.models import Sequential
from keras.layers import LSTM, Dense            # Used for implementing Neural Network layers
from keras.callbacks import TensorBoard

# Ignoring Warnings
import warnings
warnings.filterwarnings("ignore")

label_map = {label:num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            if res.shape != ():  # Check if the frame is not empty
                window.append(res)
            else:
                print(f"Empty frame found: {os.path.join(DATA_PATH, action, str(sequence), '{}.npy'.format(frame_num))}")
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Inconsistent sequence length found for action {action}, sequence {sequence}")

# Check shapes of keypoints to ensure consistency
for seq in sequences:
    for frame in seq:
        if frame.shape != (63,):  # Assuming each frame should have shape (63,)
            print(f"Inconsistent shape found: {frame.shape}")

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)
assert X.shape[1:] == (sequence_length, 63), f"X shape: {X.shape[1:]}"

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)      # Tensorboard is used for improving model performance by updating hyperparameters
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

# Model is trained in the file named Model.h5
# Can be accessed by app.py only