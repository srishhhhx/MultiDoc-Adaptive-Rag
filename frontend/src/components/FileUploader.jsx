import React, { useState } from "react";
import axios from "axios";

const FileUploader = ({ onUploadSuccess, sessionId, isAddMode = false }) => {
  const [files, setFiles] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadResults, setUploadResults] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0 });
  
  // For backward compatibility in add mode (keep for legacy code)
  const file = files[0] || null;
  const uploadResult = uploadResults[0] || null;

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    // Both modes now support multiple files
    validateFiles(selectedFiles);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const droppedFiles = Array.from(event.dataTransfer.files);
    // Both modes now support multiple files
    validateFiles(droppedFiles);
  };


  const validateFiles = (selectedFiles) => {
    setError(null);
    setUploadResults([]);
    if (!selectedFiles || selectedFiles.length === 0) return;

    const validFiles = [];
    const errors = [];

    for (const file of selectedFiles) {
      if (file.type !== "application/pdf" && 
          !file.name.endsWith('.pdf') &&
          !file.name.endsWith('.docx') &&
          !file.name.endsWith('.txt') &&
          !file.name.endsWith('.csv') &&
          !file.name.endsWith('.xlsx')) {
        errors.push(`${file.name}: Invalid file type`);
        continue;
      }

      if (file.size > 10 * 1024 * 1024) {
        errors.push(`${file.name}: File too large (>10MB)`);
        continue;
      }

      validFiles.push(file);
    }

    if (errors.length > 0) {
      setError(errors.join(', '));
    }

    if (validFiles.length > 0) {
      setFiles(validFiles);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragOver(true);
  };
  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setError(null);
    setUploadResults([]);
    
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      if (sessionId) {
        formData.append('session_id', sessionId);
      }

      const response = await axios.post('/api/upload-multiple', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = response.data;
      
      // Handle multiple upload results
      if (result.successful_uploads) {
        setUploadResults(result.successful_uploads);
        
        // Notify parent for each successful upload with proper metadata format
        if (onUploadSuccess) {
          result.successful_uploads.forEach(upload => {
            const uploadResult = {
              session_id: result.session_id,
              metadata: {
                filename: upload.filename,
                size: upload.size,
                chunks: upload.chunks,
                uploadedAt: result.uploadedAt,
                document_id: upload.filename // Use filename as document ID for now
              }
            };
            onUploadSuccess(uploadResult);
          });
        }
      }
      
      if (result.failed_uploads && result.failed_uploads.length > 0) {
        const errors = result.failed_uploads.map(fail => `${fail.filename}: ${fail.error}`);
        setError(errors.join('; '));
      }
      
      console.log('FileUploader: Multiple upload result:', result);
      
      // Clear files after successful uploads
      setFiles([]);
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Upload failed';
      setError(errorMsg);
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
      setUploadProgress({ current: 0, total: 0 });
    }
  };

  const formatSize = (bytes) => {
    const mb = bytes / (1024 * 1024);
    return mb < 1 ? Math.round(bytes / 1024) + "KB" : mb.toFixed(1) + "MB";
  };

  const handleReset = async () => {
    // Clear the database when resetting
    try {
      await axios.post('/api/clear-database');
      console.log('Database cleared');
    } catch (err) {
      console.error('Error clearing database:', err);
    }
    
    setFiles([]);
    setError(null);
    setUploadResults([]);
    setUploadProgress({ current: 0, total: 0 });
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
    setError(null);
  };

  // Compact mode for adding more documents
  if (isAddMode) {
    return (
      <div className="w-full">
        <div className="bg-[#0f0f0f] border border-[#333] rounded-xl p-4">
          {files.length === 0 && uploadResults.length === 0 && (
            <div
              className={`border-2 border-dashed ${isDragOver ? 'border-[#444]' : 'border-[#2a2a2a]'} rounded-lg p-4 text-center transition-colors duration-200`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <input
                type="file"
                onChange={handleFileChange}
                id="file-input-add"
                className="hidden"
                accept=".pdf,.docx,.txt,.csv,.xlsx"
                multiple
              />
              <label htmlFor="file-input-add" className="cursor-pointer block">
                <div className="text-[#aaa] text-[13px] font-medium">
                  <span className="text-white font-bold">+ Add Another Document</span>
                </div>
              </label>
            </div>
          )}
          
          {files.length > 0 && uploadResults.length === 0 && (
            <div className="space-y-3">
              {/* Show all selected files */}
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {files.map((file, index) => (
                  <div key={index} className="bg-[#1a1a1a] border border-[#333] rounded-lg p-3 flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="text-white text-[13px] font-semibold truncate">
                        {file.name}
                      </div>
                      <div className="text-[#666] text-[11px]">
                        {formatSize(file.size)}
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="text-[#666] hover:text-white text-lg px-2"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="w-full bg-[#22c55e] hover:bg-[#16a34a] text-white text-[13px] font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
              >
                {uploading ? `Uploading ${files.length} file${files.length > 1 ? 's' : ''}...` : `Upload ${files.length} file${files.length > 1 ? 's' : ''}`}
              </button>
            </div>
          )}
          
          {uploadResults.length > 0 && (
            <div className="text-center">
              <div className="text-[#22c55e] text-[13px] font-semibold mb-2">
                ✓ {uploadResults.length} file{uploadResults.length > 1 ? 's' : ''} added successfully!
              </div>
              <div className="text-[#888] text-[11px] mb-3">
                {uploadResults.map(result => result.filename).join(', ')}
              </div>
              <button
                onClick={() => {
                  setUploadResults([]);
                  setFiles([]);
                  setError(null);
                }}
                className="text-[#aaa] text-[12px] hover:text-white"
              >
                Add more documents
              </button>
            </div>
          )}
          
          {error && (
            <div className="text-[#ff6b6b] text-[12px] text-center mt-2">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Full mode for initial upload
  return (
    <div className="w-full max-w-md">
      <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-2xl p-12">
        <div className="text-white text-[48px] font-extrabold text-center mb-10 tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
          Drop
          <div className="text-[#555] text-[15px] font-medium mt-2 tracking-wide uppercase" style={{fontFamily: 'Product Sans, sans-serif'}}>
            PDF • DOCX • TXT • CSV • XLSX • 10MB max
          </div>
        </div>

        {uploadResults.length > 0 ? (
          <div className="mb-8">
            <div className="bg-[#0a140a] border border-[#1a3d1a] rounded-lg p-5 flex items-center gap-3 mb-5">
              <div className="bg-[#22c55e] text-white w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
                ✓
              </div>
              <div className="text-[#22c55e] text-[15px] font-bold tracking-tight">
                {uploadResults.length > 1 ? `${uploadResults.length} Files Uploaded!` : 'Upload Complete!'}
              </div>
            </div>
            <div className="bg-[#0f0f0f] border border-[#222] rounded-lg p-4 mb-5 space-y-3">
              {uploadResults.map((result, index) => (
                <div key={index} className={`${index > 0 ? 'pt-3 border-t border-[#1a1a1a]' : ''}`}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[#666] text-[13px] font-medium">File:</span>
                    <span className="text-white text-[13px] font-semibold text-right truncate max-w-[200px]">
                      {result.metadata.filename}
                    </span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[#666] text-[13px] font-medium">Size:</span>
                    <span className="text-white text-[13px] font-semibold">
                      {result.metadata.size}
                    </span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[#666] text-[13px] font-medium">Chunks:</span>
                    <span className="text-white text-[13px] font-semibold">
                      {result.metadata.chunks}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[#666] text-[13px] font-medium">Uploaded:</span>
                    <span className="text-white text-[13px] font-semibold">
                      {new Date(result.metadata.uploadedAt).toLocaleString()}
                    </span>
                  </div>
                </div>
              ))}
              {uploadResults.length > 1 && (
                <div className="pt-3 border-t border-[#1a1a1a]">
                  <div className="flex justify-between items-center">
                    <span className="text-[#666] text-[13px] font-medium">Total Chunks:</span>
                    <span className="text-white text-[13px] font-semibold">
                      {uploadResults.reduce((sum, result) => sum + result.metadata.chunks, 0)}
                    </span>
                  </div>
                </div>
              )}
            </div>
            <button 
              onClick={handleReset} 
              className="w-full h-12 border border-[#333] rounded-lg bg-transparent text-[#aaa] cursor-pointer text-[15px] font-semibold transition-all duration-200 hover:bg-[#0f0f0f] hover:text-white hover:border-[#444] tracking-tight"
            >
              Upload Another File
            </button>
          </div>
        ) : files.length > 0 ? (
          <div className="mb-8">
            <div className="space-y-3">
              {files.map((file, index) => (
                <div key={index} className="bg-[#0f0f0f] border border-[#222] rounded-lg p-4 flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="text-white text-[14px] font-bold mb-1 truncate tracking-tight">
                      {file.name}
                    </div>
                    <div className="text-[#666] text-[12px] font-medium">
                      {formatSize(file.size)}
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="bg-transparent border-none text-[#666] cursor-pointer text-xl px-2 py-1 rounded transition-colors duration-200 hover:text-white"
                  >
                    ×
                  </button>
                </div>
              ))}
              {files.length > 1 && (
                <div className="text-center text-[#666] text-[12px] font-medium">
                  {files.length} files selected
                </div>
              )}
            </div>
          </div>
        ) : (
          <div
            className={`border-2 border-dashed ${isDragOver ? 'border-[#444]' : 'border-[#2a2a2a]'} rounded-xl p-10 text-center mb-8 transition-colors duration-200`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              type="file"
              onChange={handleFileChange}
              id="file-input"
              className="hidden"
              accept=".pdf,.docx,.txt,.csv,.xlsx"
              multiple
            />
            <label htmlFor="file-input" className="cursor-pointer block">
              <div className="relative w-20 h-20 mx-auto mb-5 group">
                <div className="relative z-10 bg-[#1a1a1a] border border-[#333] w-20 h-20 rounded-lg flex items-center justify-center text-[#888] transition-all duration-300 group-hover:translate-x-5 group-hover:-translate-y-5 group-hover:border-[#444]">
                  <svg
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    strokeWidth={2}
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 24 24"
                    width={20}
                    height={20}
                  >
                    <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2" />
                    <path d="M7 9l5 -5l5 5" />
                    <path d="M12 4l0 12" />
                  </svg>
                </div>
                <div className="absolute inset-0 border-2 border-dashed border-[#444] rounded-lg opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
              </div>
              <div className="text-[#aaa] text-[15px] font-medium">
                Drop files or <span className="text-white font-bold">browse</span>
                <div className="text-[#666] text-[12px] mt-1">
                  Select multiple files to upload at once
                </div>
              </div>
            </label>
          </div>
        )}

        {error && (
          <div className="bg-[#0f0808] border border-[#2a1a1a] rounded-lg p-4 mb-6 text-[#ff6b6b] text-sm font-semibold text-center tracking-tight">
            {error}
          </div>
        )}

        {uploadResults.length === 0 && (
          <button
            className={`w-[140px] h-12 mx-auto block border-none rounded-lg cursor-pointer relative overflow-hidden transition-all duration-200 ${
              files.length === 0 || uploading ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            style={{
              background: files.length === 0 || uploading 
                ? 'linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%)'
                : 'linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%)'
            }}
          >
            <div 
              className="absolute inset-[1px] rounded-[7px]"
              style={{
                background: 'linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 100%)'
              }}
            />
            <div className="relative z-10 text-white text-[15px] font-bold h-full flex items-center justify-center gap-2 tracking-tight">
              {uploading ? (
                <>
                  <div className="w-3 h-3 border-2 border-[#333] border-t-white rounded-full animate-spin" />
                  {uploadProgress.total > 1 ? `${uploadProgress.current}/${uploadProgress.total}` : 'Uploading'}
                </>
              ) : (
                <>
                  {files.length > 1 ? `Upload ${files.length}` : 'Upload'}
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 53 58"
                    height={12}
                    width={12}
                  >
                    <path
                      strokeWidth={9}
                      stroke="currentColor"
                      d="M44.25 36.3612L17.25 51.9497C11.5833 55.2213 4.5 51.1318 4.50001 44.5885L4.50001 13.4115C4.50001 6.86824 11.5833 2.77868 17.25 6.05033L44.25 21.6388C49.9167 24.9104 49.9167 33.0896 44.25 36.3612Z"
                    />
                  </svg>
                </>
              )}
            </div>
          </button>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
