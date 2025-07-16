import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['models', 'sample_images', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_tensorflow():
    """Test TensorFlow installation"""
    print("🧪 Testing TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU support detected!")
        else:
            print("💻 Running on CPU (this is fine for inference)")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("🎨 Creating sample data...")
    
    # Create a simple test to verify the model works
    try:
        from caption_generator import SmartCaptionGenerator
        generator = SmartCaptionGenerator()
        print("✅ Caption generator initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing caption generator: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up AI Image Caption Generator...")
    print("=" * 50)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directories),
        ("Testing TensorFlow", test_tensorflow),
        ("Downloading NLTK data", download_nltk_data),
        ("Creating sample data", create_sample_data)
    ]
    
    success_count = 0
    for step_name, step_function in steps:
        print(f"\n🔄 {step_name}...")
        if step_function():
            success_count += 1
        else:
            print(f"⚠️ {step_name} failed, but continuing...")
    
    print("\n" + "=" * 50)
    print(f"✅ Setup completed! {success_count}/{len(steps)} steps successful.")
    
    if success_count >= len(steps) - 1:  # Allow one failure
        print("\n🎉 Your Image Caption Generator is ready!")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
        print("\n📖 Open your browser and go to the URL shown in the terminal.")
    else:
        print("\n⚠️ Some setup steps failed. Please check the errors above.")
        print("💡 You can try running the app anyway with: streamlit run app.py")

if __name__ == "__main__":
    main()