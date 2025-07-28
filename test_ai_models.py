#!/usr/bin/env python3
"""
Quick test untuk AI Models yang sudah diperbaiki
"""

print("🧪 Testing Enhanced AI Models...")
print("="*50)

try:
    # Test import
    import sys
    import os
    sys.path.append('src')
    
    from src.ai_models import SensorAIModels
    print("✅ AI Models imported successfully")
    
    # Test initialization (tanpa real components)
    print("🤖 Testing model initialization...")
    
    # Create dummy shape
    dummy_shape = (60, 30)  # 60 timesteps, 30 features
    
    try:
        ai_models = SensorAIModels(input_shape=dummy_shape)
        print("✅ AI Models initialized successfully")
        
        # Test model building
        print("🏗️ Testing model building...")
        
        ai_models.build_cnn_model()
        print("✅ CNN model built")
        
        ai_models.build_lstm_model()  
        print("✅ LSTM model built")
        
        ai_models.build_hybrid_cnn_lstm_model()
        print("✅ Hybrid CNN-LSTM model built")
        
        print(f"📊 Total models created: {len(ai_models.models)}")
        print(f"📋 Model types: {list(ai_models.models.keys())}")
        
        # Test health check
        health = ai_models.health_check()
        print("💚 Health check completed")
        print(f"   Models loaded: {health['models_loaded']}")
        print(f"   GPU available: {health['gpu_available']}")
        
        print("\n🎉 ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        print("⚠️ Possible TensorFlow installation issues")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed")

except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("="*50)
print("🏁 Test completed")
