import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D, Dropout
from transformers import TFBertModel

max_len=70
num_of_classes=4
n_epoch=10
batch_size=32

def create_model(max_len, num_of_classes):
    inp = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    att_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert = TFBertModel.from_pretrained('bert-base-cased')
    embeddings = bert(inp, attention_mask = att_mask)[0]
    x = GlobalMaxPool1D()(embeddings)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    op = Dense(num_of_classes, activation='softmax')(x)
    model = Model(inputs=[inp, att_mask], outputs=op)
    return model

# Load tokenized data
from tokenization import data_loader_agnews
x_train, x_val, x_test, y_train, y_val, y_test = data_loader(max_len)

# Build model
model = create_model(max_len, num_of_classes)
model.layers[2].trainable = True

# Use Adam optimizer and Crossentropy loss
model.compile(optimizer=Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0),
              loss=CategoricalCrossentropy(),
              metrics=CategoricalAccuracy('accuracy'))    

# Train model
model.fit(x = {'input_ids':x_train['input_ids'], 'attention_mask':x_train['attention_mask']},
          y = to_categorical(y_train),
          validation_data = ({'input_ids':x_val['input_ids'], 'attention_mask':x_val['attention_mask']},
          to_categorical(y_val)), epochs=n_epoch, batch_size=batch_size)

# Evaluate model on held-out test set
from sklearn.metrics import classification_report

preds = model.predict({'input_ids': x_test['input_ids'], 'attention_mask': x_test['attention_mask']})
y_pred = np.argmax(preds, axis=1)
print(classification_report(y_test, y_pred))
