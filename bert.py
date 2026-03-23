# Import necessary libraries
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Load pre-trained BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define function to tokenize and encode text data
def encode_text(text_list, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in text_list:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)

# Encode training and test data using BERT tokenizer
x_train_input_ids, x_train_attention_masks = encode_text(x_train, tokenizer, max_length=256)
x_test_input_ids, x_test_attention_masks = encode_text(x_test, tokenizer, max_length=256)

# Create a BERT-based classification model
input_ids = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)
attention_masks = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)

output = bert_model([input_ids, attention_masks])
output = output[1]
output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(lr=2e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit([x_train_input_ids, x_train_attention_masks], y_train,
                    epochs=3, batch_size=32, validation_split=0.1)

# Use the trained BERT model to predict test set labels
y_pred_bert = model.predict([x_test_input_ids, x_test_attention_masks])
y_pred_bert = np.round(y_pred_bert)

# Calculate the confusion matrix
cm_bert = confusion_matrix(y_test, y_pred_bert)

# Print the accuracy and specificity of the BERT model
print("BERT Model")
accuracy = (cm_bert[0][0] + cm_bert[1][1]) / sum(sum(cm_bert))
specificity = cm_bert[0][0] / (cm_bert[0][0] + cm_bert[0][1])
print("BERT model accuracy: {:.2f}%".format(accuracy * 100))
print("BERT model specificity: {:.2f}%".format(specificity * 100))
