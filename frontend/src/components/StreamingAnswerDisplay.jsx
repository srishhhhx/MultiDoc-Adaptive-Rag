import React, { useEffect, useRef, useState } from 'react';
import { EVENT_TYPES, STAGES } from '../types/streamEvents';

/**
 * StreamingAnswerDisplay Component
 *
 * Displays streaming answer with smooth transitions for:
 * - Provisional answer (streaming tokens)
 * - Rewrite transitions (fade out/in)
 * - Final validated answer
 *
 * Features:
 * - Smooth CSS animations
 * - Memory-efficient token accumulation
 * - No unnecessary re-renders
 * - Automatic scroll to bottom
 */
const StreamingAnswerDisplay = ({
  provisionalAnswer,
  finalAnswer,
  isRewriting,
  currentStage,
  attemptInfo,
  isStreaming
}) => {
  const answerRef = useRef(null);
  const [shouldAnimate, setShouldAnimate] = useState(false);

  // Auto-scroll to bottom as answer streams in
  useEffect(() => {
    if (answerRef.current && (provisionalAnswer || finalAnswer)) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [provisionalAnswer, finalAnswer]);

  // Trigger rewrite animation
  useEffect(() => {
    if (isRewriting) {
      setShouldAnimate(true);
      // Reset animation after it completes
      const timer = setTimeout(() => setShouldAnimate(false), 600);
      return () => clearTimeout(timer);
    }
  }, [isRewriting]);

  // Determine what to display
  const displayText = finalAnswer || provisionalAnswer;
  const isProvisional = !finalAnswer && provisionalAnswer;
  const showAttemptBadge = attemptInfo && attemptInfo.current > 1;

  if (!displayText && !isRewriting && !isStreaming) {
    return null;
  }

  return (
    <div className="w-full max-w-4xl">
      {/* Answer Container */}
      <div className="relative">
        {/* Gradient Corner Accent */}
        <div className="absolute -top-3 -left-3 w-24 h-24 bg-gradient-to-br from-[#22c55e]/20 to-transparent blur-2xl rounded-full"></div>

        <div
          className={`relative bg-gradient-to-br from-[#0f0f0f] via-[#0a0a0a] to-black border border-[#1a1a1a] rounded-3xl p-8 transition-all duration-500 ${
            shouldAnimate ? 'animate-rewrite' : ''
          }`}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-1 h-8 bg-gradient-to-b from-[#22c55e] to-transparent rounded-full"></div>
              <h2
                className="text-white text-3xl font-bold tracking-tight"
                style={{ fontFamily: 'Product Sans, sans-serif' }}
              >
                {finalAnswer ? 'Response' : 'Generating...'}
              </h2>
            </div>

            {/* Status Badges */}
            <div className="flex items-center gap-2">
              {showAttemptBadge && (
                <div className="flex items-center gap-2 text-[#fbbf24] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#fbbf24]/5 border border-[#fbbf24]/30">
                  <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>ATTEMPT {attemptInfo.current}/{attemptInfo.max}</span>
                </div>
              )}

              {isProvisional && !finalAnswer && (
                <div className="flex items-center gap-2 text-[#6b9fff] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#6b9fff]/5 border border-[#6b9fff]/30">
                  <div className="w-2 h-2 bg-[#6b9fff] rounded-full animate-pulse"></div>
                  <span>STREAMING</span>
                </div>
              )}

              {finalAnswer && (
                <div className="flex items-center gap-2 text-[#22c55e] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#22c55e]/5 border border-[#22c55e]/30">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span>VALIDATED</span>
                </div>
              )}
            </div>
          </div>

          {/* Answer Text */}
          {isRewriting ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <div className="w-8 h-8 border-3 border-[#fbbf24]/30 border-t-[#fbbf24] rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-[#888] text-sm" style={{ fontFamily: 'Product Sans, sans-serif' }}>
                  {currentStage?.message || 'Refining answer...'}
                </p>
              </div>
            </div>
          ) : displayText ? (
            <div
              ref={answerRef}
              className={`text-[#d0d0d0] text-[17px] leading-relaxed max-h-[600px] overflow-y-auto custom-scrollbar ${
                isProvisional ? 'opacity-90' : 'opacity-100'
              }`}
              style={{ fontFamily: 'Product Sans, sans-serif', whiteSpace: 'pre-wrap' }}
            >
              {displayText.split('\n').map((paragraph, idx) => {
                // Skip empty lines
                if (!paragraph.trim()) {
                  return <div key={idx} className="h-4"></div>;
                }

                // Check if it's a numbered list item or bullet point
                const isNumberedList = /^\d+\.\s+/.test(paragraph);
                const isBulletPoint = /^[•\-\*]\s+/.test(paragraph);
                const isBoldHeader = /^\*\*.*\*\*/.test(paragraph);

                if (isBoldHeader) {
                  // Bold header (e.g., **Header:**)
                  const text = paragraph.replace(/\*\*/g, '');
                  return (
                    <div key={idx} className="font-bold text-[#fff] mt-4 mb-2">
                      {text}
                    </div>
                  );
                } else if (isNumberedList || isBulletPoint) {
                  // List item with proper indentation
                  return (
                    <div key={idx} className="mb-3 pl-4">
                      {paragraph}
                    </div>
                  );
                } else {
                  // Regular paragraph
                  return (
                    <div key={idx} className="mb-4">
                      {paragraph}
                    </div>
                  );
                }
              })}
              {isProvisional && (
                <span className="inline-block w-2 h-5 bg-[#22c55e] ml-1 animate-blink"></span>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StreamingAnswerDisplay;
