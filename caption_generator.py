import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os
from PIL import Image
import random
from config import SCENE_CATEGORIES, DEFAULT_CAPTIONS, IMAGE_CONFIG, CAPTION_CONFIG, TONE_TEMPLATES

class SmartCaptionGenerator:
    def __init__(self):
        self.encoder_model = None
        self.inception_full = None
        self.load_encoder()
        
        # Load scene templates from config
        self.scene_templates = {
            category: data['templates'] 
            for category, data in SCENE_CATEGORIES.items()
        }
        
        # Load scene keywords from config
        self.scene_keywords = {
            category: data['keywords'] 
            for category, data in SCENE_CATEGORIES.items()
        }
        
    def load_encoder(self):
        """Load the InceptionV3 encoder model"""
        try:
            # Create a unique name prefix for each model to avoid conflicts
            # Load encoder model first
            encoder_base = InceptionV3(weights='imagenet', include_top=True)
            self.encoder_model = tf.keras.Model(
                inputs=encoder_base.input, 
                outputs=encoder_base.layers[-2].output,
                name="feature_encoder"
            )
            
            # Load the full classifier model separately
            self.inception_full = InceptionV3(weights='imagenet', include_top=True)
            
            print("✅ InceptionV3 encoder loaded successfully")
        except Exception as e:
            print(f"❌ Error loading encoder: {e}")
    
    def extract_features(self, image):
        """Extract features from image using InceptionV3"""
        try:
            # Resize image to 299x299 (InceptionV3 input size)
            if isinstance(image, str):
                img = load_img(image, target_size=(299, 299))
            else:
                img = image.resize((299, 299))
            
            # Convert to array and preprocess
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features directly to avoid retracing issues
            # Use direct predict method instead of calling through tf.function
            features = self.encoder_model.predict(img_array, verbose=0)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def analyze_image_content(self, image):
        """Basic image content analysis to ensure reliable operation"""
        try:
            # Check if model is loaded
            if self.inception_full is None:
                print("Warning: Full InceptionV3 model not loaded. Falling back to feature model.")
                return 'general', 0.5, ['subject']
            
            # Process image
            if isinstance(image, str):
                img = load_img(image, target_size=(299, 299))
            else:
                img = image.resize((299, 299))
            
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict directly with model to avoid retracing issues
            predictions = self.inception_full.predict(img_array, verbose=0)
            decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=10)[0]
            
            # Get top classes and their confidence
            top_classes = [pred[1].lower() for pred in decoded_predictions]
            confidence_scores = [pred[2] for pred in decoded_predictions]
            
            # Basic scene categorization
            scene_type, scene_confidence = self.categorize_scene(top_classes)
            
            return scene_type, scene_confidence, top_classes
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return 'general', 0.5, ['subject']
    
    def categorize_scene(self, predictions):
        """Categorize the scene based on predictions"""
        # Count matches for each category using config keywords
        category_scores = {}
        for category, keywords in self.scene_keywords.items():
            score = sum(1 for pred in predictions if any(keyword in pred for keyword in keywords))
            category_scores[category] = score
        
        # Return category with highest score, or 'general' if no clear match
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category, 0.7
        
        return 'general', 0.5
    
    def generate_smart_caption(self, image, tone="creative"):
        """Generate an intelligent caption based on enhanced image analysis and tone"""
        try:
            # Analyze image content (basic analysis that works)
            scene_type, confidence, top_classes = self.analyze_image_content(image)[:3]
            
            # Get appropriate template based on tone and scene
            if scene_type in self.scene_templates and tone in TONE_TEMPLATES:
                base_templates = self.scene_templates[scene_type]
                tone_data = TONE_TEMPLATES[tone]
                
                # Get main subject
                main_subject = top_classes[0].replace('_', ' ') if top_classes else "subject"
                
                # Generate caption based on tone
                if tone == "creative":
                    creative_words = ["stunning", "breathtaking", "mesmerizing", "captivating"]
                    caption = f"A {random.choice(creative_words)} {main_subject} that creates visual impact with artistic composition"
                
                elif tone == "professional":
                    caption = f"Professional {main_subject} photography showcasing excellent technical execution and attention to detail"
                
                elif tone == "casual":
                    caption = f"Really cool {main_subject}! Love how this was captured"
                
                elif tone == "poetic":
                    poetic_phrases = ["like a painted dream", "poetry in visual form", "a moment frozen in beauty"]
                    caption = f"{main_subject.title()} captured {random.choice(poetic_phrases)}"
                
                elif tone == "social":
                    hashtags = " ".join([f"#{main_subject.replace(' ', '')}", "#photography", "#beautiful"])
                    caption = f"Amazing {main_subject} vibes! ✨ {hashtags}"
                
                elif tone == "descriptive":
                    caption = f"A detailed capture of {main_subject} showcasing clear visual elements and composition"
                    
                else:
                    caption = base_templates[0]
            else:
                # Fallback caption with tone variation
                main_subject = top_classes[0].replace('_', ' ') if top_classes else "composition"
                caption = f"A beautiful {main_subject} captured with artistic vision"
            
            return caption, confidence, scene_type
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            scene_type, confidence = self.categorize_scene(['subject'])
            return f"A beautiful image with unique visual elements", confidence, scene_type
    
    def _generate_sophisticated_caption(self, scene_type, predictions, context_info, visual_elements, tone, confidence):
        """Generate sophisticated captions using comprehensive image analysis"""
        
        # Extract key elements
        primary_subject = context_info.get('primary_objects', [predictions[0].replace('_', ' ') if predictions else 'subject'])[0]
        environment = context_info.get('environment', ['setting'])[0] if context_info.get('environment') else None
        lighting = visual_elements.get('lighting', 'natural lighting')
        colors = visual_elements.get('colors', {})
        composition_elements = visual_elements.get('composition', ['artistic composition'])
        
        # Build caption based on tone
        if tone == "creative":
            return self._build_creative_caption(primary_subject, environment, lighting, colors, composition_elements)
        
        elif tone == "professional":
            return self._build_professional_caption(primary_subject, environment, lighting, composition_elements, confidence)
        
        elif tone == "casual":
            return self._build_casual_caption(primary_subject, context_info, colors)
        
        elif tone == "poetic":
            return self._build_poetic_caption(primary_subject, environment, lighting, colors)
        
        elif tone == "social":
            return self._build_social_caption(primary_subject, context_info, predictions)
        
        elif tone == "descriptive":
            return self._build_descriptive_caption(primary_subject, context_info, visual_elements, environment)
        
        else:
            return f"A beautiful capture of {primary_subject} with {lighting}"
    
    def _build_creative_caption(self, subject, environment, lighting, colors, composition):
        """Build creative, artistic captions"""
        creative_intros = [
            "A mesmerizing capture of", "An artistic portrayal of", "A captivating scene featuring",
            "An imaginative composition showcasing", "A visually stunning display of"
        ]
        
        creative_descriptors = {
            'bright': 'bathed in radiant light',
            'dark': 'shrouded in mysterious shadows',
            'warm': 'glowing with warm, inviting tones',
            'cool': 'rendered in cool, calming hues',
            'balanced': 'harmoniously balanced in color and light'
        }
        
        color_desc = creative_descriptors.get(colors.get('dominant', 'balanced'), 'beautifully illuminated')
        
        if environment:
            return f"{random.choice(creative_intros)} {subject} in a {environment} {color_desc}, creating an enchanting visual narrative"
        else:
            return f"{random.choice(creative_intros)} {subject} {color_desc}, captured with artistic vision and creative flair"
    
    def _build_professional_caption(self, subject, environment, lighting, composition, confidence):
        """Build professional, technical captions"""
        technical_terms = [
            "expertly composed", "professionally captured", "skillfully photographed",
            "technically excellent", "masterfully executed"
        ]
        
        quality_indicators = {
            'high': 'exceptional clarity and detail',
            'medium': 'excellent image quality',
            'low': 'artistic interpretation'
        }
        
        quality_level = 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
        quality_desc = quality_indicators[quality_level]
        
        if environment:
            return f"Professional {subject} photography captured in a {environment} with {lighting}, demonstrating {quality_desc} and {random.choice(technical_terms)} composition"
        else:
            return f"{random.choice(technical_terms).title()} {subject} photography showcasing {quality_desc} with superior {lighting}"
    
    def _build_casual_caption(self, subject, context_info, colors):
        """Build casual, friendly captions"""
        casual_starters = [
            "Check out this awesome", "Love this shot of", "Really cool", "Amazing",
            "Such a great capture of", "Totally loving this"
        ]
        
        casual_endings = [
            "Perfect vibes!", "So cool!", "Amazing shot!", "Love the colors!",
            "This is so good!", "What a moment!"
        ]
        
        color_comments = {
            'bright': "The lighting is incredible!",
            'warm': "Love these warm tones!",
            'cool': "Those cool colors are so nice!",
            'balanced': "Perfect color balance!"
        }
        
        color_comment = color_comments.get(colors.get('dominant', 'balanced'), random.choice(casual_endings))
        
        return f"{random.choice(casual_starters)} {subject}! {color_comment}"
    
    def _build_poetic_caption(self, subject, environment, lighting, colors):
        """Build poetic, lyrical captions"""
        poetic_frameworks = [
            "Like a {metaphor}, {subject} {action} in this {setting}",
            "Where {lighting} meets {subject}, {beauty} unfolds",
            "{subject} captured as if {metaphor}, {emotion} in every detail",
            "In this moment, {subject} becomes {metaphor}, {lighting} like {comparison}"
        ]
        
        metaphors = [
            "poetry in motion", "a painted dream", "nature's own artistry", "a frozen symphony",
            "whispered secrets", "dancing light", "silent music", "living artwork"
        ]
        
        beauty_words = [
            "magic", "wonder", "beauty", "grace", "elegance", "serenity", "harmony"
        ]
        
        actions = [
            "gracefully poses", "silently waits", "gently rests", "peacefully exists",
            "elegantly stands", "softly glows"
        ]
        
        if environment:
            setting = f"{environment} setting"
        else:
            setting = "timeless scene"
        
        framework = random.choice(poetic_frameworks)
        return framework.format(
            subject=subject,
            metaphor=random.choice(metaphors),
            lighting=lighting,
            setting=setting,
            beauty=random.choice(beauty_words),
            action=random.choice(actions),
            emotion="poetry",
            comparison="gentle brushstrokes"
        )
    
    def _build_social_caption(self, subject, context_info, predictions):
        """Build social media optimized captions"""
        social_starters = [
            "Obsessed with this", "Can't get over this", "Living for this", "Absolutely loving this",
            "This is everything!", "Major vibes with this"
        ]
        
        trending_phrases = [
            "aesthetic goals", "pure perfection", "mood forever", "total inspiration",
            "dream vibes", "absolutely stunning"
        ]
        
        hashtags = self._generate_enhanced_hashtags(subject, predictions, context_info)
        
        return f"{random.choice(social_starters)} {subject}! {random.choice(trending_phrases)} ✨ {hashtags}"
    
    def _build_descriptive_caption(self, subject, context_info, visual_elements, environment):
        """Build detailed, descriptive captions"""
        description_parts = [f"This image features {subject}"]
        
        if environment:
            description_parts.append(f"located in a {environment}")
        
        # Add lighting description
        lighting = visual_elements.get('lighting', 'natural lighting')
        description_parts.append(f"captured with {lighting}")
        
        # Add color information
        colors = visual_elements.get('colors', {})
        if colors.get('dominant'):
            description_parts.append(f"displaying {colors['dominant']} tones")
        
        # Add secondary objects if available
        secondary = context_info.get('secondary_objects', [])
        if secondary:
            description_parts.append(f"with {', '.join(secondary[:2])} also visible")
        
        # Add composition details
        composition = visual_elements.get('composition', [])
        if composition:
            description_parts.append(f"showcasing {composition[0]}")
        
        return ". ".join(description_parts) + "."
    
    def _apply_tone_to_caption(self, base_caption, tone, predictions, confidence):
        """Apply tone-specific modifications to the base caption"""
        if tone not in TONE_TEMPLATES:
            return base_caption
            
        tone_data = TONE_TEMPLATES[tone]
        style_words = tone_data['style_words']
        sentence_starters = tone_data['sentence_starters']
        
        # Get main subject from predictions
        main_subject = predictions[0].replace('_', ' ') if predictions else "subject"
        
        # Select style based on confidence
        if confidence > 0.8:
            intensity = "high"
        elif confidence > 0.5:
            intensity = "medium"
        else:
            intensity = "low"
        
        # Generate caption based on tone
        if tone == "creative":
            return f"{random.choice(sentence_starters)} {main_subject} {random.choice(style_words)} in this {random.choice(['captivating', 'mesmerizing', 'enchanting'])} composition"
        
        elif tone == "professional":
            return f"Professional {main_subject} photography showcasing {random.choice(style_words)} with excellent {random.choice(['composition', 'lighting', 'technical execution'])}"
        
        elif tone == "casual":
            return f"{random.choice(sentence_starters)} {main_subject}! {random.choice(style_words)} and totally {random.choice(['awesome', 'cool', 'amazing'])}"
        
        elif tone == "poetic":
            poetic_options = ['poetry in motion', 'a painted dream', "nature's own artistry"]
            return f"{random.choice(sentence_starters)} {main_subject} {random.choice(style_words)}, like {random.choice(poetic_options)}"
        
        elif tone == "social":
            hashtags = self._generate_hashtags(main_subject, predictions)
            return f"{random.choice(sentence_starters)} {main_subject} {random.choice(style_words)}! {hashtags}"
        
        elif tone == "descriptive":
            details = self._extract_visual_details(predictions)
            return f"A detailed view featuring {main_subject} with {details} captured in {random.choice(style_words)} detail"
        
        return base_caption
    
    def _generate_fallback_caption(self, tone, predictions, confidence):
        """Generate fallback captions when scene type is unknown"""
        main_subject = predictions[0].replace('_', ' ') if predictions else "visual elements"
        
        fallback_captions = {
            "creative": f"An imaginative capture featuring {main_subject} with artistic flair",
            "professional": f"A professional photograph showcasing {main_subject} with technical excellence",
            "casual": f"Check out this cool shot of {main_subject}! Pretty awesome!",
            "poetic": f"A visual poem featuring {main_subject}, captured like a moment suspended in time",
            "social": f"Amazing {main_subject} vibes! ✨ {self._generate_hashtags(main_subject, predictions)}",
            "descriptive": f"A comprehensive view of {main_subject} displaying intricate details and composition"
        }
        
        return fallback_captions.get(tone, f"A beautiful image featuring {main_subject}")
    
    def _generate_hashtags(self, main_subject, predictions):
        """Generate relevant hashtags for social media"""
        base_tags = ["#photography", "#beautiful", "#amazing"]
        subject_tags = [f"#{main_subject.replace(' ', '')}", f"#{main_subject.replace(' ', '').lower()}"]
        
        # Add tags based on predictions
        prediction_tags = []
        for pred in predictions[:3]:
            clean_pred = pred.replace('_', '').replace(' ', '').lower()
            if len(clean_pred) > 3:
                prediction_tags.append(f"#{clean_pred}")
        
        all_tags = base_tags + subject_tags + prediction_tags
        return " ".join(all_tags[:6])  # Limit to 6 hashtags
    
    def _extract_visual_details(self, predictions):
        """Extract visual details from predictions for descriptive captions"""
        if not predictions:
            return "various visual elements"
        
        details = []
        for pred in predictions[:3]:
            detail = pred.replace('_', ' ').lower()
            details.append(detail)
        
        if len(details) == 1:
            return details[0]
        elif len(details) == 2:
            return f"{details[0]} and {details[1]}"
        else:
            return f"{', '.join(details[:-1])}, and {details[-1]}"
    
    def _generate_enhanced_hashtags(self, subject, predictions, context_info):
        """Generate enhanced hashtags based on comprehensive analysis"""
        hashtags = set()
        
        # Base photography tags
        hashtags.update(["#photography", "#beautiful", "#amazing", "#photooftheday"])
        
        # Subject-based tags
        clean_subject = subject.replace(' ', '').lower()
        hashtags.add(f"#{clean_subject}")
        
        # Category-specific tags
        if any(keyword in subject.lower() for keyword in ['person', 'man', 'woman', 'people']):
            hashtags.update(["#portrait", "#people", "#human"])
        elif any(keyword in subject.lower() for keyword in ['dog', 'cat', 'animal']):
            hashtags.update(["#animal", "#pet", "#wildlife", "#nature"])
        elif any(keyword in subject.lower() for keyword in ['food', 'meal', 'dish']):
            hashtags.update(["#food", "#foodie", "#delicious", "#yummy"])
        elif any(keyword in subject.lower() for keyword in ['car', 'vehicle']):
            hashtags.update(["#car", "#automotive", "#vehicle"])
        
        # Environment-based tags
        environment = context_info.get('environment', [])
        if environment:
            for env in environment[:2]:
                clean_env = env.replace(' ', '').lower()
                hashtags.add(f"#{clean_env}")
        
        # Mood and activity tags
        activities = context_info.get('activity_indicators', [])
        for activity in activities[:2]:
            clean_activity = activity.replace(' ', '').lower()
            hashtags.add(f"#{clean_activity}")
        
        # Limit to 8-10 most relevant hashtags
        return " ".join(list(hashtags)[:10])
    
    def generate_multiple_captions(self, image, num_captions=3, tone="creative"):
        """Generate multiple caption variations with specified tone"""
        try:
            # Basic image analysis
            scene_type, confidence, top_classes = self.analyze_image_content(image)[:3]
            
            captions = []
            main_subject = top_classes[0].replace('_', ' ') if top_classes else "subject"
            
            # Generate different variations based on tone
            for i in range(num_captions):
                if tone == "creative":
                    creative_words = ["stunning", "breathtaking", "mesmerizing", "captivating", "enchanting"]
                    caption = f"A {random.choice(creative_words)} {main_subject} that {random.choice(['tells a story', 'captures the imagination', 'evokes emotion'])}"
                
                elif tone == "professional":
                    professional_terms = ["composition", "lighting", "perspective", "technical execution"]
                    caption = f"Professional {main_subject} photography with excellent {random.choice(professional_terms)}"
                
                elif tone == "casual":
                    casual_words = ["awesome", "cool", "amazing", "sweet", "nice"]
                    caption = f"Really {random.choice(casual_words)} {main_subject}! Love this shot"
                
                elif tone == "poetic":
                    poetic_phrases = ["like a painted dream", "poetry in visual form", "a moment frozen in beauty"]
                    caption = f"{main_subject.title()} captured {random.choice(poetic_phrases)}"
                
                elif tone == "social":
                    emoji_sets = ["✨🔥", "💫⭐", "🌟💎", "🎨📸"]
                    hashtags = f"#photography #{main_subject.replace(' ', '')}"
                    caption = f"{main_subject.title()} vibes! {random.choice(emoji_sets)} {hashtags}"
                
                elif tone == "descriptive":
                    details = ", ".join(top_classes[1:3]) if len(top_classes) > 1 else "visual elements"
                    caption = f"Detailed capture showing {main_subject} with {details} in clear focus"
                
                else:
                    caption = f"Beautiful {main_subject} captured with artistic vision"
                
                captions.append(caption)
            
            # Ensure we have unique captions
            unique_captions = list(set(captions))
            while len(unique_captions) < num_captions:
                if tone == "creative":
                    new_caption = f"An artistic view of {main_subject} with imaginative composition"
                elif tone == "professional":
                    new_caption = f"Expertly captured {main_subject} with technical excellence"
                else:
                    new_caption = f"A captivating {main_subject} that draws the viewer's attention"
                    
                unique_captions.append(new_caption)
                unique_captions = list(set(unique_captions))
            
            return unique_captions[:num_captions]
            
        except Exception as e:
            print(f"Error generating multiple captions: {e}")
            return [
                f"A beautiful image with interesting composition",
                f"A captivating scene with excellent visual elements",
                f"A well-composed photograph with striking details"
            ]
    
    def _generate_subject_focused_caption(self, subject, context_info, visual_elements, tone):
        """Generate caption focused on the main subject"""
        environment = context_info.get('environment', ['setting'])[0] if context_info.get('environment') else None
        
        if tone == "creative":
            if environment:
                return f"An extraordinary {subject} perfectly positioned in a {environment}, creating visual poetry through masterful composition"
            return f"A magnificent {subject} captured with artistic brilliance and creative vision"
        
        elif tone == "professional":
            return f"Professional {subject} photography demonstrating technical excellence and superior composition skills"
        
        elif tone == "casual":
            return f"Loving this {subject}! Such a perfect shot with amazing details"
        
        elif tone == "poetic":
            return f"Where {subject} meets artistry, magic happens in silent whispers of light and shadow"
        
        elif tone == "social":
            hashtags = self._generate_enhanced_hashtags(subject, [subject], context_info)
            return f"{subject.title()} perfection! Absolutely stunning ✨💎 {hashtags}"
        
        else:  # descriptive
            return f"Detailed capture of {subject} showing exceptional clarity and comprehensive visual information"
    
    def _generate_mood_focused_caption(self, subject, visual_elements, tone):
        """Generate caption focused on mood and lighting"""
        lighting = visual_elements.get('lighting', 'natural lighting')
        colors = visual_elements.get('colors', {})
        
        mood_descriptors = {
            'bright': 'uplifting and energetic',
            'dark': 'mysterious and dramatic',
            'warm': 'cozy and inviting',
            'cool': 'serene and calming',
            'balanced': 'harmonious and peaceful'
        }
        
        mood = mood_descriptors.get(colors.get('dominant', 'balanced'), 'captivating')
        
        if tone == "creative":
            return f"A {mood} capture of {subject} where {lighting} creates an enchanting atmosphere of pure visual magic"
        
        elif tone == "professional":
            return f"Expert use of {lighting} creates {mood} mood in this professionally executed {subject} photograph"
        
        elif tone == "casual":
            return f"The lighting in this {subject} shot is incredible! Such {mood} vibes"
        
        elif tone == "poetic":
            return f"In gentle {lighting}, {subject} whispers stories of {mood} beauty frozen in time"
        
        elif tone == "social":
            return f"{mood.title()} {subject} energy! This lighting is everything! ✨🔥 #mood #perfect"
        
        else:  # descriptive
            return f"{subject.title()} photographed with {lighting}, creating {mood} visual atmosphere with excellent technical execution"
    
    def _generate_artistic_caption(self, subject, visual_elements, tone):
        """Generate caption focused on artistic and compositional elements"""
        composition = visual_elements.get('composition', ['artistic composition'])[0]
        colors = visual_elements.get('colors', {})
        
        if tone == "creative":
            return f"An artistic masterpiece featuring {subject} with {composition}, where every element contributes to visual storytelling excellence"
        
        elif tone == "professional":
            return f"Superior {composition} showcases {subject} with professional-grade technical precision and artistic vision"
        
        elif tone == "casual":
            return f"This {subject} has such amazing composition! Really love the artistic style"
        
        elif tone == "poetic":
            return f"Through {composition}, {subject} becomes poetry, each element dancing in visual harmony"
        
        elif tone == "social":
            return f"Artistic {subject} goals! That composition though! 🎨✨ #art #composition #goals"
        
        else:  # descriptive
            return f"Analytical view of {subject} demonstrating {composition} with precise attention to visual design principles"
    
    def _generate_context_focused_caption(self, subject, context_info, tone):
        """Generate caption focused on context and activities"""
        activities = context_info.get('activity_indicators', [])
        secondary_objects = context_info.get('secondary_objects', [])
        
        context_elements = []
        if activities:
            context_elements.extend(activities[:2])
        if secondary_objects:
            context_elements.extend(secondary_objects[:2])
        
        if context_elements:
            context_desc = f"with {', '.join(context_elements)}"
        else:
            context_desc = "in perfect context"
        
        if tone == "creative":
            return f"A storytelling capture of {subject} {context_desc}, weaving narrative through visual elements"
        
        elif tone == "professional":
            return f"Contextual {subject} photography {context_desc}, demonstrating superior environmental awareness"
        
        elif tone == "casual":
            return f"Great shot of {subject} {context_desc}! Love how everything comes together"
        
        elif tone == "poetic":
            return f"In perfect harmony, {subject} {context_desc} creates a symphony of visual elements"
        
        elif tone == "social":
            return f"{subject.title()} vibes {context_desc}! Perfect scene! 🌟📸 #perfect #scene"
        
        else:  # descriptive
            return f"Comprehensive view of {subject} {context_desc}, providing complete contextual information"
    
    def generate_social_media_caption(self, image, base_caption):
        """Generate a social media optimized caption with hashtags and emojis"""
        try:
            # Basic image analysis
            scene_type, confidence, top_classes = self.analyze_image_content(image)[:3]
            main_subject = top_classes[0].replace('_', ' ') if top_classes else "photo"
            
            # Social media style with emojis and hashtags
            trending_emojis = {
                'people': ['👥', '💫', '✨', '🌟'],
                'animal': ['🐾', '💕', '🦋', '🌸'],
                'food': ['😋', '🤤', '✨', '👌'],
                'vehicle': ['🚗', '⚡', '💨', '🔥'],
                'nature': ['🌿', '🌅', '🍃', '💚'],
                'architecture': ['🏛️', '✨', '📐', '🎨'],
                'general': ['✨', '💫', '🌟', '⭐']
            }
            
            emojis = trending_emojis.get(scene_type, trending_emojis['general'])
            selected_emojis = random.sample(emojis, min(2, len(emojis)))
            
            # Create engaging social caption
            engagement_starters = [
                "Obsessed with this", "Love this", "Can't get enough of this",
                "Major vibes", "Absolutely loving this"
            ]
            
            # Generate hashtags
            hashtags = [
                f"#{main_subject.replace(' ', '')}",
                "#photography",
                "#beautiful",
                "#photooftheday",
                f"#{scene_type}vibes"
            ]
            
            # Build social caption
            social_caption = f"{random.choice(engagement_starters)} {main_subject}! {base_caption} {''.join(selected_emojis)}\n\n"
            social_caption += "Tag someone who would love this! 👇\n\n"
            social_caption += " ".join(hashtags)
            
            return social_caption
            
        except Exception as e:
            print(f"Error generating social media caption: {e}")
            return f"{base_caption} ✨📸 #photography #beautiful #photooftheday"
    
    def advanced_scene_categorization(self, predictions, confidence_scores):
        """Advanced scene categorization with weighted scoring"""
        category_scores = {}
        
        # Enhanced scoring system with confidence weighting
        for category, keywords in self.scene_keywords.items():
            weighted_score = 0
            for i, pred in enumerate(predictions):
                for keyword in keywords:
                    if keyword in pred:
                        # Weight score by position and confidence
                        position_weight = 1.0 - (i * 0.1)  # Reduce weight for lower positions
                        confidence_weight = confidence_scores[i] if i < len(confidence_scores) else 0.1
                        weighted_score += position_weight * confidence_weight * 2
            category_scores[category] = weighted_score
        
        # Find best category with confidence
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            if max_score > 0.3:  # Threshold for category confidence
                scene_confidence = min(max_score, 1.0)
                return best_category, scene_confidence
        
        return 'general', 0.5
    
    def _extract_contextual_info(self, predictions, confidence_scores):
        """Extract contextual information from predictions"""
        context = {
            'primary_objects': [],
            'secondary_objects': [],
            'environment': [],
            'activity_indicators': [],
            'mood_indicators': []
        }
        
        # Categorize predictions into context types
        object_keywords = ['person', 'dog', 'cat', 'car', 'building', 'tree', 'flower', 'food']
        environment_keywords = ['outdoor', 'indoor', 'beach', 'forest', 'city', 'room', 'kitchen']
        activity_keywords = ['playing', 'running', 'sitting', 'walking', 'eating', 'sleeping']
        mood_keywords = ['sunset', 'sunny', 'cloudy', 'bright', 'dark', 'colorful']
        
        for i, pred in enumerate(predictions[:8]):
            confidence = confidence_scores[i] if i < len(confidence_scores) else 0.1
            
            if confidence > 0.1:  # Only consider reasonably confident predictions
                if any(keyword in pred for keyword in object_keywords):
                    if confidence > 0.3:
                        context['primary_objects'].append(pred.replace('_', ' '))
                    else:
                        context['secondary_objects'].append(pred.replace('_', ' '))
                
                if any(keyword in pred for keyword in environment_keywords):
                    context['environment'].append(pred.replace('_', ' '))
                
                if any(keyword in pred for keyword in activity_keywords):
                    context['activity_indicators'].append(pred.replace('_', ' '))
                
                if any(keyword in pred for keyword in mood_keywords):
                    context['mood_indicators'].append(pred.replace('_', ' '))
        
        return context
    
    def _analyze_visual_elements(self, image, predictions):
        """Analyze visual elements of the image"""
        import numpy as np
        
        # Convert image to numpy array for analysis
        if isinstance(image, str):
            img = load_img(image, target_size=(224, 224))
        else:
            img = image.resize((224, 224))
        
        img_array = np.array(img)
        
        # Analyze color distribution
        colors = self._analyze_colors(img_array)
        
        # Analyze brightness and contrast
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Infer lighting and composition
        lighting = self._infer_lighting(brightness, contrast)
        composition = self._infer_composition(predictions, colors)
        
        return {
            'colors': colors,
            'brightness': brightness,
            'contrast': contrast,
            'lighting': lighting,
            'composition': composition
        }
    
    def _analyze_colors(self, img_array):
        """Analyze dominant colors in the image"""
        # Flatten image and find dominant colors
        pixels = img_array.reshape(-1, 3)
        
        # Simple color analysis
        avg_color = np.mean(pixels, axis=0)
        
        # Determine color characteristics
        if avg_color[0] > 150 and avg_color[1] > 150 and avg_color[2] > 150:
            dominant = "bright"
        elif avg_color[0] < 80 and avg_color[1] < 80 and avg_color[2] < 80:
            dominant = "dark"
        elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            dominant = "warm"
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            dominant = "cool"
        else:
            dominant = "balanced"
        
        return {
            'dominant': dominant,
            'avg_rgb': avg_color.tolist(),
            'saturation': 'high' if np.std(pixels) > 50 else 'low'
        }
    
    def _infer_lighting(self, brightness, contrast):
        """Infer lighting conditions from brightness and contrast"""
        if brightness > 180:
            if contrast > 60:
                return "bright with strong contrast"
            else:
                return "bright and even"
        elif brightness > 120:
            if contrast > 50:
                return "well-lit with good contrast"
            else:
                return "naturally lit"
        elif brightness > 80:
            if contrast > 40:
                return "moody with dramatic shadows"
            else:
                return "softly lit"
        else:
            if contrast > 30:
                return "dramatic low-key lighting"
            else:
                return "dimly lit"
    
    def _infer_composition(self, predictions, colors):
        """Infer composition style from predictions and colors"""
        composition_styles = []
        
        # Infer from predictions
        if any('portrait' in pred for pred in predictions):
            composition_styles.append("portrait composition")
        elif any('landscape' in pred for pred in predictions):
            composition_styles.append("landscape composition")
        
        # Infer from colors
        if colors['saturation'] == 'high':
            composition_styles.append("vibrant and colorful")
        
        if colors['dominant'] == 'balanced':
            composition_styles.append("harmoniously balanced")
        
        return composition_styles if composition_styles else ["artistic composition"]