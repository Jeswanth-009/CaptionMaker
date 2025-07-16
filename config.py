"""
Configuration file for the Image Caption Generator
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

# Model configuration
MODEL_CONFIG = {
    'max_length': 34,
    'vocab_size': 8485,
    'embedding_dim': 200,
    'features_shape': 2048,
    'lstm_units': 256,
    'dropout_rate': 0.5
}

# Image processing configuration
IMAGE_CONFIG = {
    'target_size': (299, 299),  # InceptionV3 input size
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_formats': ['jpg', 'jpeg', 'png', 'webp'],
    'quality_threshold': 0.3  # Minimum confidence for good captions
}

# UI configuration
UI_CONFIG = {
    'page_title': "AI Image Caption Generator",
    'page_icon': "🖼️",
    'layout': "wide",
    'theme': {
        'primary_color': "#667eea",
        'secondary_color': "#764ba2",
        'background_color': "#f8f9ff",
        'text_color': "#333333"
    }
}

# Caption generation settings
CAPTION_CONFIG = {
    'beam_width': 3,
    'num_alternatives': 3,
    'min_confidence': 0.1,
    'max_retries': 3
}

# Tone-based caption templates
TONE_TEMPLATES = {
    'creative': {
        'style_words': ['artistic', 'imaginative', 'inspired', 'visionary', 'innovative', 'expressive'],
        'sentence_starters': ['Behold', 'Discover', 'Witness', 'Experience', 'Embrace'],
        'modifiers': ['masterfully', 'creatively', 'artistically', 'imaginatively']
    },
    'professional': {
        'style_words': ['technical excellence', 'precision', 'professional quality', 'expertly crafted', 'polished'],
        'sentence_starters': ['Professional', 'Expert', 'Skillfully captured', 'Technically superior'],
        'modifiers': ['professionally', 'expertly', 'precisely', 'meticulously']
    },
    'casual': {
        'style_words': ['cool', 'awesome', 'nice', 'sweet', 'fun', 'chill'],
        'sentence_starters': ['Check out this', 'Look at this', 'Here\'s a', 'Pretty cool'],
        'modifiers': ['totally', 'really', 'super', 'pretty']
    },
    'poetic': {
        'style_words': ['ethereal', 'sublime', 'transcendent', 'graceful', 'flowing', 'harmonious'],
        'sentence_starters': ['A symphony of', 'Poetry in', 'Like a dream', 'Whispers of'],
        'modifiers': ['gracefully', 'elegantly', 'poetically', 'beautifully']
    },
    'social': {
        'style_words': ['viral-worthy', 'insta-perfect', 'share-ready', 'trending', 'goals'],
        'sentence_starters': ['OMG', 'Obsessed with', 'Can\'t even', 'Living for'],
        'modifiers': ['absolutely', 'totally', 'completely', 'definitely']
    },
    'descriptive': {
        'style_words': ['detailed', 'comprehensive', 'thorough', 'analytical', 'informative'],
        'sentence_starters': ['This image shows', 'Observable here is', 'The scene depicts', 'Present in this frame'],
        'modifiers': ['clearly', 'distinctly', 'precisely', 'accurately']
    }
}

# Scene categories and their keywords
SCENE_CATEGORIES = {
    'people': {
        'keywords': ['person', 'man', 'woman', 'child', 'baby', 'face', 'human', 'people', 'boy', 'girl'],
        'templates': [
            "A captivating portrait showcasing genuine human emotion and natural expression",
            "People sharing a beautiful moment together with authentic connections",
            "A stunning portrait that captures the essence of human personality and character",
            "Individuals expressing themselves naturally in this candid and heartwarming scene"
        ]
    },
    'animal': {
        'keywords': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'lion', 'tiger', 'bear', 'rabbit', 'fish', 'pet'],
        'templates': [
            "A magnificent creature displaying its natural beauty and wild grace",
            "An adorable animal captured in a perfect moment of natural behavior",
            "A stunning wildlife photograph showcasing the animal's unique characteristics and personality",
            "A beautiful creature living freely in its natural environment with perfect timing"
        ]
    },
    'food': {
        'keywords': ['pizza', 'burger', 'sandwich', 'cake', 'bread', 'fruit', 'apple', 'banana', 'food', 'meal', 'dish'],
        'templates': [
            "A mouth-watering culinary masterpiece presented with exquisite attention to detail",
            "Delicious cuisine artfully arranged to showcase its appetizing colors and textures",
            "A gourmet creation that perfectly balances visual appeal with culinary excellence",
            "Fresh ingredients transformed into an irresistible dish that delights all the senses"
        ]
    },
    'vehicle': {
        'keywords': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat', 'ship', 'vehicle'],
        'templates': [
            "A stunning example of automotive design and engineering excellence",
            "Transportation technology showcased from a dynamic and compelling perspective",
            "A beautifully maintained vehicle displaying exceptional craftsmanship and style",
            "Modern mobility captured with attention to design details and aesthetic appeal"
        ]
    },
    'nature': {
        'keywords': ['tree', 'flower', 'mountain', 'beach', 'ocean', 'forest', 'grass', 'sky', 'cloud', 'sunset', 'landscape'],
        'templates': [
            "Nature's breathtaking beauty captured in a moment of perfect harmony and tranquility",
            "A spectacular natural landscape displaying the raw power and elegance of our planet",
            "Mother Nature's artistry revealed through stunning colors, textures, and natural lighting",
            "A serene natural scene that inspires peace and connection with the natural world"
        ]
    },
    'architecture': {
        'keywords': ['building', 'house', 'church', 'tower', 'bridge', 'castle', 'monument', 'structure'],
        'templates': [
            "Magnificent architectural achievement showcasing exceptional design and structural elegance",
            "A building that represents the perfect fusion of form, function, and artistic vision",
            "Architectural brilliance captured from a perspective that highlights its unique features",
            "Structural artistry that demonstrates human creativity and engineering prowess"
        ]
    },
    'indoor': {
        'keywords': ['room', 'kitchen', 'bedroom', 'office', 'restaurant', 'store', 'museum', 'interior'],
        'templates': [
            "A thoughtfully designed interior space with sophisticated lighting and elegant details",
            "An inviting indoor environment that perfectly balances comfort with aesthetic appeal",
            "Interior design excellence showcasing harmonious color schemes and spatial arrangement",
            "A well-appointed indoor setting that creates the perfect atmosphere and ambiance"
        ]
    },
    'outdoor': {
        'keywords': ['park', 'street', 'road', 'garden', 'field', 'landscape', 'outdoor'],
        'templates': [
            "A captivating outdoor scene bathed in natural light with perfect environmental harmony",
            "An outdoor vista that showcases the beauty of open spaces and natural elements",
            "A scenic outdoor landscape featuring dynamic textures and compelling visual depth",
            "An expansive outdoor setting that invites exploration and peaceful contemplation"
        ]
    }
}

# Default captions for fallback
DEFAULT_CAPTIONS = [
    "A visually compelling image that captures attention with its unique composition and artistic merit",
    "An expertly crafted photograph showcasing exceptional attention to detail and visual storytelling",
    "A captivating scene that draws viewers in through masterful use of light, color, and perspective",
    "A beautifully composed image that demonstrates the photographer's artistic vision and technical skill",
    "A striking visual narrative that perfectly balances aesthetic appeal with emotional resonance",
    "An outstanding photograph that captures a moment of genuine beauty and authentic expression",
    "A professionally executed image featuring excellent composition and compelling visual elements"
]

# Error messages
ERROR_MESSAGES = {
    'file_too_large': "File size too large. Please upload an image smaller than 10MB.",
    'invalid_format': "Invalid file format. Please upload JPG, PNG, or WebP images.",
    'model_load_error': "Error loading AI model. Please refresh the page and try again.",
    'processing_error': "Error processing image. Please try a different image.",
    'network_error': "Network error. Please check your internet connection."
}

# Success messages
SUCCESS_MESSAGES = {
    'caption_generated': "🎉 Captions generated successfully!",
    'model_loaded': "✅ AI models loaded successfully!",
    'image_uploaded': "📸 Image uploaded successfully!"
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [MODELS_DIR, TEMP_DIR, SAMPLE_IMAGES_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)

# Initialize directories on import
create_directories()