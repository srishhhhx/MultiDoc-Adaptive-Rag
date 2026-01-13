import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import QuestionAnswer from './components/QuestionAnswer';
import StreamingQuestionAnswer from './components/StreamingQuestionAnswer';
import FileIcon from './components/FileIcon';
import TypewriterHeading from './components/TypewriterHeading';
import TypewriterText from './components/TypewriterText';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [showSessionSummary, setShowSessionSummary] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true); // Enable streaming by default

  // Debug logging
  console.log("App render - sessionId:", sessionId);
  console.log("App render - uploadedDocuments:", uploadedDocuments);
  console.log("App render - uploadedDocuments.length:", uploadedDocuments.length);

  const handleUploadSuccess = (result) => {
    console.log("App received upload result:", result);
    
    // Set session ID if this is the first upload
    if (!sessionId && result.session_id) {
      setSessionId(result.session_id);
      console.log("Setting session ID:", result.session_id);
    }
    
    // Add the new document to our list
    const newDocument = {
      document_id: result.metadata?.document_id || `doc_${Date.now()}_${result.metadata?.filename}`,
      filename: result.metadata?.filename || result.filename,
      size: result.metadata?.size || result.size,
      chunks: result.metadata?.chunks || result.chunks,
      uploadedAt: result.metadata?.uploadedAt || new Date().toISOString()
    };
    
    console.log("Adding document:", newDocument);
    setUploadedDocuments(prev => {
      const updated = [...prev, newDocument];
      console.log("Updated documents list:", updated);
      return updated;
    });
  };

  const handleDeleteDocument = async (documentId, filename) => {
    if (!sessionId) return;
    
    try {
      const response = await fetch(`/api/session/${sessionId}/document/${encodeURIComponent(documentId)}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Document deleted:', result);
        
        // Remove document from frontend state
        setUploadedDocuments(prev => 
          prev.filter(doc => doc.document_id !== documentId && doc.filename !== filename)
        );
        
        // If no documents left, reset to upload state
        if (result.remaining_documents === 0) {
          setSessionId(null);
          setUploadedDocuments([]);
        }
      } else {
        const error = await response.json();
        console.error('Delete failed:', error);
        alert(`Failed to delete document: ${error.detail}`);
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert('Failed to delete document. Please try again.');
    }
  };

  const handleNewSession = () => {
    setSessionId(null);
    setUploadedDocuments([]);
    setSessionInfo(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0a0a] via-black to-[#0f0f0f] flex flex-col items-center p-8">
      {/* Header */}
      <div className="w-full max-w-6xl mb-2 text-center mt-16">
        <TypewriterHeading 
          text="Multi-Doc Adaptive RAG"
          className="text-white text-6xl font-extrabold mb-6 tracking-tight"
          style={{fontFamily: 'Product Sans, sans-serif'}}
        />
        <TypewriterText 
          text=""
          className="text-[#888] text-xl font-medium"
          style={{fontFamily: 'Product Sans, sans-serif'}}
          delay={1700} // Start after heading is mostly done
        />
      </div>

      {/* Main Content */}
      <div className="w-full max-w-6xl flex flex-col items-center gap-8">
        {/* File Upload Section */}
        {uploadedDocuments.length === 0 && (
          <FileUploader onUploadSuccess={handleUploadSuccess} sessionId={sessionId} />
        )}

        {/* Session Interface */}
        {uploadedDocuments.length > 0 && (
          <div className="w-full flex gap-6">
            {/* Documents Sidebar - Left */}
            <div className="w-80 flex-shrink-0">
              <div className="bg-gradient-to-br from-[#111] to-[#0a0a0a] border border-[#222] rounded-3xl p-6 sticky top-6 shadow-2xl shadow-black/50">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-white text-lg font-bold tracking-tight uppercase text-[11px] text-[#888]" style={{fontFamily: 'Product Sans, sans-serif'}}>
                    Documents ({uploadedDocuments.length})
                  </h2>
                  <div className="text-[#555] text-[10px] font-mono">
                    Session: {sessionId?.substring(0, 8)}...
                  </div>
                </div>
                
                <div className="space-y-6">
                  {/* Documents List */}
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {uploadedDocuments.map((doc, index) => (
                      <div key={doc.document_id || index} className="bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f] border border-[#333] rounded-xl p-4 group hover:border-[#444] transition-colors">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0">
                            <FileIcon filename={doc.filename} className="w-8 h-8" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-white text-[15px] font-semibold break-words mb-1" style={{fontFamily: 'Product Sans, sans-serif'}}>
                              {doc.filename}
                            </div>
                            <div className="flex gap-3 text-[#666] text-[12px]" style={{fontFamily: 'Product Sans, sans-serif'}}>
                              <span>{doc.size}</span>
                              <span>{doc.chunks} chunks</span>
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteDocument(doc.document_id || doc.filename, doc.filename)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 text-[#666] hover:text-[#ff6b6b] text-lg px-2 py-1 rounded hover:bg-[#2a1a1a]"
                            title={`Delete ${doc.filename}`}
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Session Summary */}
                  <div className="pt-4 border-t border-[#1a1a1a]">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <div className="text-[#666] text-[12px] font-semibold uppercase tracking-wider mb-1.5" style={{fontFamily: 'Product Sans, sans-serif'}}>
                          Total Docs
                        </div>
                        <div className="text-white text-[17px] font-semibold" style={{fontFamily: 'Product Sans, sans-serif'}}>
                          {uploadedDocuments.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-[#666] text-[12px] font-semibold uppercase tracking-wider mb-1.5" style={{fontFamily: 'Product Sans, sans-serif'}}>
                          Total Chunks
                        </div>
                        <div className="text-white text-[17px] font-semibold" style={{fontFamily: 'Product Sans, sans-serif'}}>
                          {uploadedDocuments.reduce((sum, doc) => sum + doc.chunks, 0)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-3">
                    <FileUploader 
                      onUploadSuccess={handleUploadSuccess} 
                      sessionId={sessionId} 
                      isAddMode={true}
                    />
                    <button
                      onClick={handleNewSession}
                      className="w-full bg-gradient-to-r from-[#0f0f0f] to-[#1a1a1a] border border-[#333] rounded-xl px-4 py-3 text-[#aaa] text-[13px] font-semibold hover:from-[#1a1a1a] hover:to-[#222] hover:text-white hover:border-[#22c55e]/50 transition-all duration-300 hover:shadow-lg hover:shadow-[#22c55e]/10"
                    >
                      Start New Session
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Q&A Interface - Right */}
            <div className="flex-1">
              {/* Conditional Rendering: Streaming or Non-Streaming */}
              {useStreaming ? (
                <StreamingQuestionAnswer
                  sessionId={sessionId}
                  useStreaming={useStreaming}
                  setUseStreaming={setUseStreaming}
                />
              ) : (
                <QuestionAnswer sessionId={sessionId} />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="w-full max-w-6xl mt-12 text-center">
        <p className="text-[#555] text-sm">
          Powered by{" "}
          <span className="text-[#888] font-semibold">LangGraph</span> •{" "}
          <span className="text-[#888] font-semibold">FastAPI</span> •{" "}
          <span className="text-[#888] font-semibold">React</span>
        </p>
      </div>
    </div>
  );
}

export default App;
