import numpy as np
from datasets import Dataset
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from keras.preprocessing.sequence import pad_sequences


def create_train_data(sentences, translations):
    input_sentences, input_translations, next_words = [], [], []
    for sentence, rephrase in zip(sentences, translations):
        for i in range(1, len(rephrase)):
            input_sentences.append(sentence)
            input_translations.append(rephrase[:i])
            next_words.append(rephrase[i])
    return input_sentences, input_translations, next_words


def create_model(padding_size, vocabulary_size_en, vocabulary_size_es, embedding_size):
    encoder_inputs = Input(shape=(padding_size,))
    encoder_embedding = Embedding(input_dim=vocabulary_size_en,
                                  output_dim=embedding_size)(encoder_inputs)
    encoder = LSTM(128, return_state=True)
    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(padding_size,))
    decoder_embedding = Embedding(input_dim=vocabulary_size_es, output_dim=embedding_size,
                                  trainable=False)(decoder_inputs)
    decoder = LSTM(128, return_state=True, return_sequences=False)
    decoder_outputs, _, _ = decoder(decoder_embedding,
                                    initial_state=encoder_states)

    decoder_outputs = Dense(vocabulary_size_es, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs],
                  decoder_outputs)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=categorical_crossentropy)

    return model


def decode(model, input_sent, word_to_id, padding_size):
    generated_sent = [word_to_id['<START>']]

    for i in range(padding_size):
        output_sent = pad_sequences([generated_sent], padding_size)
        predictions = model.predict([np.expand_dims(input_sent, axis=0), output_sent])
        next_word = np.argmax(predictions)
        generated_sent.append(next_word)

    return generated_sent


def convert(sentences, id_to_word):
    out_sentences = []

    for sent in sentences:
        out_sentences.append(' '.join([id_to_word[s] for s in sent]))

    return out_sentences


def create_transformers_train_data(sentences, translations, tokenizer):
    inputs_en = tokenizer(sentences, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        outputs_es = tokenizer(translations, max_length=128, truncation=True)

    data = Dataset.from_dict({'input_ids': inputs_en['input_ids'],
                              'attention_mask': inputs_en['attention_mask'],
                              'labels': outputs_es['input_ids']})
    return data


def train_transformer(model, train_loader, optimizer, epochs=5, device='cpu'):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')


def decode_with_transformer(sentence, tokenizer, model, device='cpu'):
    model = model.to(device)
    model.eval()
    tokens = tokenizer([sentence], return_tensors='pt').to(device)
    out = model.generate(**tokens, max_length=128)

    with tokenizer.as_target_tokenizer():
        pred_sentence = tokenizer.decode(out[0], skip_special_tokens=True)

    return pred_sentence
