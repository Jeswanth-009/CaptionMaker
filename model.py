import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Add, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import pickle
import os

class ImageCaptionModel:
    def __init__(self):
        self.max_length = 34
        self.vocab_size = 8485
        self.embedding_dim = 200
        self.features_shape = 2048
        
        # Initialize tokenizer
        self.tokenizer = None
        self.model = None
        self.encoder_model = None
        
    def create_encoder_model(self):
        """Create InceptionV3 based image encoder"""
        # Load pre-trained InceptionV3 model with unique name to avoid conflicts
        inception_model = InceptionV3(weights='imagenet', include_top=True)
        # Remove the last layer (classification layer)
        encoder_model = Model(
            inputs=inception_model.input, 
            outputs=inception_model.layers[-2].output, 
            name="feature_extractor"
        )
        return encoder_model
    
    def create_decoder_model(self):
        """Create LSTM based caption decoder"""
        # Image feature input
        image_input = Input(shape=(self.features_shape,))
        image_dense = Dense(256, activation='relu')(image_input)
        
        # Text input
        text_input = Input(shape=(self.max_length,))
        text_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(text_input)
        text_dropout = Dropout(0.5)(text_embedding)
        text_lstm = LSTM(256)(text_dropout)
        
        # Combine image and text features
        combined = Add()([image_dense, text_lstm])
        combined_dense = Dense(256, activation='relu')(combined)
        combined_dropout = Dropout(0.5)(combined_dense)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(combined_dropout)
        
        # Create model
        model = Model(inputs=[image_input, text_input], outputs=output)
        return model
    
    def extract_image_features(self, image_path):
        """Extract features from image using InceptionV3"""
        if self.encoder_model is None:
            self.encoder_model = self.create_encoder_model()
            
        # Load and preprocess image
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        
        # Extract features
        features = self.encoder_model.predict(image, verbose=0)
        return features
    
    def word_for_id(self, integer, tokenizer):
        """Get word from integer using tokenizer"""
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def generate_caption_beam_search(self, image_features, beam_width=3):
        """Generate caption using beam search for better results"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
            
        # Initialize beam with start token
        start_token = self.tokenizer.word_index.get('startseq', 1)
        sequences = [[[start_token], 0.0]]
        
        for _ in range(self.max_length):
            all_candidates = []
            
            for seq, score in sequences:
                if len(seq) >= self.max_length or seq[-1] == self.tokenizer.word_index.get('endseq', 2):
                    all_candidates.append([seq, score])
                    continue
                
                # Pad sequence
                padded_seq = seq + [0] * (self.max_length - len(seq))
                padded_seq = np.array(padded_seq).reshape(1, self.max_length)
                
                # Predict next word probabilities
                predictions = self.model.predict([image_features, padded_seq], verbose=0)[0]
                
                # Get top beam_width predictions
                top_indices = np.argsort(predictions)[-beam_width:]
                
                for idx in top_indices:
                    new_seq = seq + [idx]
                    new_score = score + np.log(predictions[idx] + 1e-8)
                    all_candidates.append([new_seq, new_score])
            
            # Select top beam_width sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = all_candidates[:beam_width]
            
            # Check if all sequences ended
            if all(seq[0][-1] == self.tokenizer.word_index.get('endseq', 2) for seq in sequences):
                break
        
        # Get best sequence
        best_sequence = sequences[0][0]
        
        # Convert to words
        caption_words = []
        for word_id in best_sequence:
            word = self.word_for_id(word_id, self.tokenizer)
            if word is None or word == 'endseq':
                break
            if word != 'startseq':
                caption_words.append(word)
        
        return ' '.join(caption_words)
    
    def generate_caption_simple(self, image_features):
        """Generate caption using greedy search"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
            
        # Start with start token
        in_text = 'startseq'
        
        for _ in range(self.max_length):
            # Encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.max_length)
            
            # Predict next word
            prediction = self.model.predict([image_features, sequence], verbose=0)
            prediction = np.argmax(prediction)
            
            # Get word from prediction
            word = self.word_for_id(prediction, self.tokenizer)
            
            if word is None or word == 'endseq':
                break
                
            in_text += ' ' + word
        
        # Remove start token
        final_caption = in_text.replace('startseq', '').strip()
        return final_caption
    
    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        """Load pre-trained model and tokenizer"""
        try:
            self.model = load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_sample_model_and_tokenizer(self):
        """Create a sample model and tokenizer for demonstration"""
        # Create a simple tokenizer
        from tensorflow.keras.preprocessing.text import Tokenizer
        
        # Sample vocabulary for demonstration
        sample_captions = [
            'startseq a dog is playing in the park endseq',
            'startseq a cat is sitting on the table endseq',
            'startseq a person is walking on the street endseq',
            'startseq a car is parked on the road endseq',
            'startseq a bird is flying in the sky endseq',
            'startseq a child is playing with toys endseq',
            'startseq a woman is reading a book endseq',
            'startseq a man is riding a bicycle endseq',
            'startseq a group of people are standing together endseq',
            'startseq a beautiful sunset over the mountains endseq'
        ]
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(sample_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Create and compile model
        self.model = self.create_decoder_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        # Save tokenizer
        os.makedirs('models', exist_ok=True)
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        return True