import React from "react";

const FileIcon = ({ filename, className = "" }) => {
  // Extract file extension
  const getFileExtension = (filename) => {
    const parts = filename.split('.');
    return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
  };

  const extension = getFileExtension(filename);

  // PDF Icon
  if (extension === 'pdf') {
    return (
      <div className={`relative ${className}`}>
        <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
          {/* Document Shape */}
          <path
            d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
            fill="#E53E3E"
            opacity="0.9"
          />
          <path
            d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
            fill="#C53030"
            opacity="0.8"
          />
          {/* PDF Text */}
          <text
            x="32"
            y="38"
            textAnchor="middle"
            fill="white"
            fontSize="12"
            fontWeight="bold"
            fontFamily="Arial, sans-serif"
          >
            PDF
          </text>
        </svg>
      </div>
    );
  }

  // Word/DOCX Icon
  if (extension === 'doc' || extension === 'docx') {
    return (
      <div className={`relative ${className}`}>
        <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
          {/* Document Shape */}
          <path
            d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
            fill="#2B6CB0"
            opacity="0.9"
          />
          <path
            d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
            fill="#2C5282"
            opacity="0.8"
          />
          {/* Word Icon */}
          <text
            x="32"
            y="38"
            textAnchor="middle"
            fill="white"
            fontSize="12"
            fontWeight="bold"
            fontFamily="Arial, sans-serif"
          >
            DOC
          </text>
        </svg>
      </div>
    );
  }

  // Text File Icon
  if (extension === 'txt') {
    return (
      <div className={`relative ${className}`}>
        <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
          {/* Document Shape */}
          <path
            d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
            fill="#718096"
            opacity="0.9"
          />
          <path
            d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
            fill="#4A5568"
            opacity="0.8"
          />
          {/* Lines */}
          <line x1="18" y1="22" x2="46" y2="22" stroke="white" strokeWidth="2" opacity="0.7" />
          <line x1="18" y1="30" x2="46" y2="30" stroke="white" strokeWidth="2" opacity="0.7" />
          <line x1="18" y1="38" x2="38" y2="38" stroke="white" strokeWidth="2" opacity="0.7" />
          {/* TXT Text */}
          <text
            x="32"
            y="52"
            textAnchor="middle"
            fill="white"
            fontSize="10"
            fontWeight="bold"
            fontFamily="Arial, sans-serif"
          >
            TXT
          </text>
        </svg>
      </div>
    );
  }

  // Markdown Icon
  if (extension === 'md' || extension === 'markdown') {
    return (
      <div className={`relative ${className}`}>
        <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
          {/* Document Shape */}
          <path
            d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
            fill="#805AD5"
            opacity="0.9"
          />
          <path
            d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
            fill="#6B46C1"
            opacity="0.8"
          />
          {/* MD Text */}
          <text
            x="32"
            y="38"
            textAnchor="middle"
            fill="white"
            fontSize="12"
            fontWeight="bold"
            fontFamily="Arial, sans-serif"
          >
            MD
          </text>
        </svg>
      </div>
    );
  }

  // CSV/Excel Icon
  if (extension === 'csv' || extension === 'xlsx' || extension === 'xls') {
    return (
      <div className={`relative ${className}`}>
        <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
          {/* Document Shape */}
          <path
            d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
            fill="#38A169"
            opacity="0.9"
          />
          <path
            d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
            fill="#2F855A"
            opacity="0.8"
          />
          {/* Grid */}
          <rect x="18" y="24" width="28" height="20" stroke="white" strokeWidth="2" fill="none" opacity="0.7" />
          <line x1="18" y1="32" x2="46" y2="32" stroke="white" strokeWidth="2" opacity="0.7" />
          <line x1="32" y1="24" x2="32" y2="44" stroke="white" strokeWidth="2" opacity="0.7" />
        </svg>
      </div>
    );
  }

  // Default Generic Document Icon
  return (
    <div className={`relative ${className}`}>
      <svg viewBox="0 0 64 64" className="w-full h-full" fill="none">
        {/* Document Shape */}
        <path
          d="M14 4h28l8 8v44a4 4 0 0 1-4 4H14a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4z"
          fill="#4A5568"
          opacity="0.9"
        />
        <path
          d="M42 4l8 8h-6a2 2 0 0 1-2-2V4z"
          fill="#2D3748"
          opacity="0.8"
        />
        {/* File Icon */}
        <rect x="20" y="24" width="24" height="3" rx="1" fill="white" opacity="0.7" />
        <rect x="20" y="32" width="20" height="3" rx="1" fill="white" opacity="0.7" />
        <rect x="20" y="40" width="22" height="3" rx="1" fill="white" opacity="0.7" />
      </svg>
    </div>
  );
};

export default FileIcon;
