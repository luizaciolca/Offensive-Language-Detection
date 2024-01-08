import torch
import tensorflow as tf

from tokenizers import BertWordPieceTokenizer
import re, string
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import BertModel as BertTokenizer
from keras.preprocessing.sequence import pad_sequences


device = torch.device('cuda:0')

data = pd.read_csv(r"C:\Users\Luiza\Desktop\News-RO-Offense.csv")

data.drop(columns=['comment_id', 'reply_to_comment_id', 'comment_nr', 'content_id'], inplace=True)

col_Label = data['LABEL']
transformed_Label = np.where(col_Label >= 1, 1, col_Label)

data.drop('LABEL', axis=1, inplace=True)
data['label'] = transformed_Label 

insult_words = ['INSULT', 'ABUSE', 'PROFANITY']
other_word = ['OTHER']

def generate_new_column(label):
    return "nonoffensive" if label == 0 else "offensive"

data['category'] = data['label'].apply(generate_new_column)
print(data)

def clean_data(data):
    data['comment_text'] = data['comment_text'].str.replace('-', ' ')
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    data['comment_text'] = data['comment_text'].apply(lambda x: x.lower())
    data['comment_text'] = data['comment_text'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    data['comment_text'] = data['comment_text'].apply(lambda x: " ".join(x.split()))
    return data

df = clean_data(data)

result = df.head(10)
print(result)

nonoffensive, offensive = np.bincount(df['label'])
total = nonoffensive + offensive
print('Examples:\n    Total: {}\n    nonoffensive: {} ({:.2f}% of total)\n'.format(
    total, nonoffensive, 100 * nonoffensive / total))
print('Examples:\n    Total: {}\n    offensive: {} ({:.2f}% of total)\n'.format(
    total, offensive, 100 * offensive / total))

weight_for_0 = (1 / nonoffensive)*(total)/3.0
weight_for_1 = (1 / offensive)*(total)/3.0

class_weight = {0: weight_for_0, 1: weight_for_1}

X_train_, X_test, y_train_, y_test = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.10,
    random_state=42,
    stratify=df.label.values,
)

X_train, X_val, y_train, y_val = train_test_split(
    df.loc[X_train_].index.values,
    df.loc[X_train_].label.values,
    test_size=0.10,
    random_state=42,
    stratify=df.loc[X_train_].label.values,
)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
df.loc[X_test, 'data_type'] = 'test'

print(df)

df_train = df.loc[df["data_type"]=="train"]
df_train.head(5)
df_val = df.loc[df["data_type"]=="val"]
df_val.head(5)
df_test = df.loc[df["data_type"]=="test"]
df_test.head(5)

df.groupby(['label', 'comment_text', 'category', 'data_type']).count()

train_ds = tf.data.Dataset.from_tensor_slices((df_train.comment_text.values, df_train.label.values))
val_ds = tf.data.Dataset.from_tensor_slices((df_val.comment_text.values, df_val.label.values))
test_ds = tf.data.Dataset.from_tensor_slices((df_test.comment_text.values, df_test.label.values))

train_ds = train_ds.shuffle(len(df_train)).batch(32, drop_remainder=False)
train_ds
val_ds = val_ds.shuffle(len(df_val)).batch(32, drop_remainder=False)
val_ds
test_ds = test_ds.shuffle(len(df_test)).batch(32, drop_remainder=False)
test_ds

sentences = df.comment_text.values

sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentences:")
print (tokenized_texts[0])
print (tokenized_texts[1])
print (tokenized_texts[2])
print (tokenized_texts[3])
print (tokenized_texts[4])

MAX_LEN = 128

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []

for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.2)
validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, random_state=2018, test_size=0.5)
train_masks, validation_masks, test_masks = [], [], []

for seq in train_inputs:
    seq_mask = [float(i > 0) for i in seq]
    train_masks.append(seq_mask)

for seq in validation_inputs:
    seq_mask = [float(i > 0) for i in seq]
    validation_masks.append(seq_mask)

for seq in test_inputs:
    seq_mask = [float(i > 0) for i in seq]
    test_masks.append(seq_mask)

from keras import layers
from transformers import BertTokenizer, BertConfig
from keras.optimizers import Adam

train_inputs = tf.constant(np.array(train_inputs, dtype=np.int32), dtype=tf.int32)
validation_inputs = tf.constant(np.array(validation_inputs, dtype=np.int32), dtype=tf.int32)
test_inputs = tf.constant(np.array(test_inputs, dtype=np.int32), dtype=tf.int32)

train_labels = tf.constant(np.array(train_labels, dtype=np.int32), dtype=tf.int32)
validation_labels = tf.constant(np.array(validation_labels, dtype=np.int32), dtype=tf.int32)
test_labels = tf.constant(np.array(test_labels, dtype=np.int32), dtype=tf.int32)

train_masks = tf.constant(np.array(train_masks, dtype=np.float32), dtype=tf.float32)
validation_masks = tf.constant(np.array(validation_masks, dtype=np.float32), dtype=tf.float32)
test_masks = tf.constant(np.array(test_masks, dtype=np.float32), dtype=tf.float32)

# Crearea setului de date pentru antrenare
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_masks, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_inputs)).batch(32, drop_remainder=False)

# Crearea setului de date pentru validare
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_masks, validation_labels))
validation_dataset = validation_dataset.batch(32, drop_remainder=False)

# Crearea setului de date pentru testare
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_masks, test_labels))
test_dataset = test_dataset.batch(32, drop_remainder=False)

# Definirea parametrilor BERT
bert_model_name = 'bert-base-uncased'
max_len = 128

from transformers import TFBertModel, BertConfig
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.models import Model

def build_CNN_classifier_model():
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert_model_name = 'bert-base-uncased'
    bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
    bert = TFBertModel.from_pretrained(bert_model_name, config=bert_config)

    embedding_layer = bert(input_ids, attention_mask)[0]
    conv1d_1 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding_layer)
    conv1d_2 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv1d_1)
    global_max_pooling = GlobalMaxPooling1D()(conv1d_2)
    dense = Dense(512, activation='relu')(global_max_pooling)
    dropout = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax', name='output')(dropout)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.summary()

    return model

num_epochs = 10
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * num_epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = Adam(learning_rate=init_lr)
init_lr = init_lr 
num_train_steps = num_train_steps 
num_warmup_steps =num_warmup_steps  

cnn_classifier_model = build_CNN_classifier_model()

bert_layer_name = 'tf_bert_model'  
bert_layer = cnn_classifier_model.get_layer(bert_layer_name)

bert_layer.trainable = True
for sub_layer in bert_layer.layers:
    sub_layer.trainable = True

non_trainable_vars = [var.name for var in cnn_classifier_model.trainable_variables if bert_layer_name not in var.name]
trainable_vars = [var for var in cnn_classifier_model.trainable_variables if var.name not in non_trainable_vars]
optimizer = Adam(learning_rate=init_lr)

cnn_classifier_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))

num_epochs = 10
best_valid_loss = float('inf') 
def train(cnn_classifier_model, train_dataset, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for batch in train_dataset:
        input_ids, attention_mask, labels = batch
        with tf.GradientTape() as tape:
            predictions = cnn_classifier_model([input_ids, attention_mask], training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        epoch_loss += loss.numpy()
        epoch_acc += np.mean(np.argmax(predictions, axis=-1) == labels.numpy())
        num_batches += 1

    return epoch_loss / num_batches, epoch_acc / num_batches

def evaluate(cnn_classifier_model, test_dataset):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for batch in test_dataset:
        input_ids, attention_mask, labels = batch
        predictions = cnn_classifier_model([input_ids, attention_mask], training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)

        epoch_loss += loss.numpy()
        epoch_acc += np.mean(np.argmax(predictions, axis=-1) == labels.numpy())
        num_batches += 1

    return epoch_loss / num_batches, epoch_acc / num_batches

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    for batch in train_dataset:
        input_ids, attention_mask, labels = batch
        with tf.GradientTape() as tape:
            predictions = cnn_classifier_model([input_ids, attention_mask], training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, cnn_classifier_model.trainable_variables, 
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, trainable_vars))   

        epoch_loss += loss.numpy()
        epoch_acc += np.mean(np.argmax(predictions, axis=-1) == labels.numpy())
        num_batches += 1

    train_loss = epoch_loss / num_batches
    train_acc = epoch_acc / num_batches

    valid_loss, valid_acc = evaluate(cnn_classifier_model, validation_dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}:\n'
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%\n'
          f'Validation Loss: {valid_loss:.4f} | Validation Acc: {valid_acc * 100:.2f}%\n')

batch_size = 32
test_loss, test_accuracy = cnn_classifier_model.evaluate(test_dataset, batch_size=batch_size)

# Print results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

test_predictions = cnn_classifier_model.predict(test_dataset)

test_predicted_labels = np.argmax(test_predictions, axis=1)

test_true_labels = np.concatenate([labels.numpy() for _, _, labels in test_dataset])

precision = precision_score(test_true_labels, test_predicted_labels, average='weighted')
recall = recall_score(test_true_labels, test_predicted_labels, average='weighted')
f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

cnn_classifier_model.save("etp3_6ian_3")      
