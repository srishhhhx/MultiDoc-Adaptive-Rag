"""
Test Groq Initialization and Availability
Checks if Groq clients are properly initialized
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("GROQ INITIALIZATION TEST")
print("="*80)

# Test 1: Check environment variables
print("\n1. Checking Environment Variables...")
groq_key = os.getenv("GROQ_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if groq_key:
    print(f"   ✅ GROQ_API_KEY: {groq_key[:20]}...")
else:
    print("   ❌ GROQ_API_KEY: NOT SET")

if google_key:
    print(f"   ✅ GOOGLE_API_KEY: {google_key[:20]}...")
else:
    print("   ❌ GOOGLE_API_KEY: NOT SET")

# Test 2: Check Groq SDK
print("\n2. Checking Groq SDK...")
try:
    import groq
    print(f"   ✅ Groq SDK installed: v{groq.__version__ if hasattr(groq, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"   ❌ Groq SDK not installed: {e}")
    sys.exit(1)

# Test 3: Check GroqModelClient
print("\n3. Checking GroqModelClient...")
try:
    from utils.groq_model_client import GroqModelClient, GROQ_AVAILABLE
    print(f"   ✅ GroqModelClient imported")
    print(f"   GROQ_AVAILABLE: {GROQ_AVAILABLE}")
    
    if GROQ_AVAILABLE:
        print("   ✅ Groq is marked as available")
    else:
        print("   ❌ Groq is marked as NOT available")
        print("   This means the Groq SDK import failed in groq_model_client.py")
        
except Exception as e:
    print(f"   ❌ Failed to import GroqModelClient: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Initialize Groq client
print("\n4. Testing Groq Client Initialization...")
try:
    from utils.groq_model_client import GroqModelClient
    
    if not os.getenv("GROQ_API_KEY"):
        print("   ⚠️  GROQ_API_KEY not set, skipping initialization test")
    else:
        client = GroqModelClient(
            model_name="llama-3.1-8b-instant",
            enable_fallback=False
        )
        print("   ✅ Groq client initialized successfully")
        
        # Test inference
        print("\n5. Testing Groq Inference...")
        response, metrics = client.infer(
            prompt="Say 'Hello' in JSON format with a 'message' field.",
            temperature=0.0,
            max_tokens=50,
            json_mode=True
        )
        print(f"   ✅ Groq inference successful!")
        print(f"   Response: {response[:100]}...")
        print(f"   Latency: {metrics.latency_ms:.0f}ms")
        print(f"   Tokens: {metrics.total_tokens}")
        
except Exception as e:
    print(f"   ❌ Groq client initialization/inference failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check evaluation client initialization
print("\n6. Checking Document Evaluation Client...")
try:
    from chains.evaluate_groq import DocumentEvaluationClient
    
    eval_client = DocumentEvaluationClient()
    
    if eval_client.groq_client:
        print("   ✅ Evaluation client has Groq client initialized")
    else:
        print("   ❌ Evaluation client does NOT have Groq client")
        print("   This means Groq initialization failed in DocumentEvaluationClient.__init__")
    
    if eval_client.gemini_client:
        print("   ✅ Evaluation client has Gemini fallback initialized")
    else:
        print("   ❌ Evaluation client does NOT have Gemini fallback")
        
except Exception as e:
    print(f"   ❌ Failed to initialize evaluation client: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check context assessment client initialization
print("\n7. Checking Context Assessment Client...")
try:
    from chains.context_assessment_groq import ContextAssessmentClient
    
    assess_client = ContextAssessmentClient()
    
    if assess_client.groq_client:
        print("   ✅ Assessment client has Groq client initialized")
    else:
        print("   ❌ Assessment client does NOT have Groq client")
    
    if assess_client.gemini_client:
        print("   ✅ Assessment client has Gemini fallback initialized")
    else:
        print("   ❌ Assessment client does NOT have Gemini fallback")
        
except Exception as e:
    print(f"   ❌ Failed to initialize assessment client: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Check document relevance client initialization
print("\n8. Checking Document Relevance Client...")
try:
    from chains.document_relevance_groq import DocumentRelevanceClient
    
    relevance_client = DocumentRelevanceClient()
    
    if relevance_client.groq_client:
        print("   ✅ Relevance client has Groq client initialized")
    else:
        print("   ❌ Relevance client does NOT have Groq client")
    
    if relevance_client.gemini_client:
        print("   ✅ Relevance client has Gemini fallback initialized")
    else:
        print("   ❌ Relevance client does NOT have Gemini fallback")
        
except Exception as e:
    print(f"   ❌ Failed to initialize relevance client: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("GROQ INITIALIZATION TEST COMPLETE")
print("="*80)

print("\n📋 Summary:")
print("If all clients show '✅ has Groq client initialized', then Groq is ready.")
print("If any show '❌ does NOT have Groq client', check the error messages above.")
print("\nCommon issues:")
print("  1. GROQ_API_KEY not set in environment")
print("  2. Groq SDK not installed (pip install groq)")
print("  3. GROQ_AVAILABLE flag is False in groq_model_client.py")
