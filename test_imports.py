"""
Test script to verify all imports work correctly after restructuring
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports after restructuring...")
print("=" * 60)

try:
    print("\n✓ Testing backend.api...")
    from backend import api
    print("  SUCCESS: backend.api imported")
    
    print("\n✓ Testing backend.config...")
    from backend import config
    print("  SUCCESS: backend.config imported")
    
    print("\n✓ Testing backend.document_processor...")
    from backend import document_processor
    print("  SUCCESS: backend.document_processor imported")
    
    print("\n✓ Testing backend.document_loader...")
    from backend import document_loader
    print("  SUCCESS: backend.document_loader imported")
    
    print("\n✓ Testing backend.rag_workflow...")
    from backend import rag_workflow
    print("  SUCCESS: backend.rag_workflow imported")
    
    print("\n✓ Testing backend.session_manager...")
    from backend import session_manager
    print("  SUCCESS: backend.session_manager imported")
    
    print("\n✓ Testing backend.state...")
    from backend import state
    print("  SUCCESS: backend.state imported")
    
    print("\n✓ Testing backend.chains modules...")
    from backend.chains import query_analysis_router
    from backend.chains import multi_tool_executor
    from backend.chains import rerank_documents
    from backend.chains import generate_answer
    print("  SUCCESS: All backend.chains modules imported")
    
    print("\n" + "=" * 60)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)
    print("\nThe restructuring is complete and all modules are accessible.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
