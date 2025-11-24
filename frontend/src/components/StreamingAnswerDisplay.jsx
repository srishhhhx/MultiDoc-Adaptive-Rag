import React, { useEffect, useRef, useState } from 'react';
import { EVENT_TYPES, STAGES } from '../types/streamEvents';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
  progressInfo,
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
                {finalAnswer ? 'Response' : 'Generating answer'}
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

          {/* Stage Progress Indicator */}
          {currentStage && isStreaming && !finalAnswer && (
            <div className="mb-6 bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-4">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-2 h-2 bg-[#6b9fff] rounded-full animate-pulse"></div>
                <span className="text-[#6b9fff] text-[13px] font-semibold">
                  {currentStage.message || 'Processing...'}
                </span>
              </div>

              {/* Progress Details */}
              {progressInfo && progressInfo.message && (
                <div className="mt-3 ml-5 space-y-2">
                  <div className="text-[#888] text-[12px]">
                    {progressInfo.message}
                  </div>

                  {/* Show routing decision */}
                  {progressInfo.routing && (
                    <div className="flex items-center gap-2">
                      <span className="text-[#aaa] text-[11px] font-semibold uppercase tracking-wide">Strategy:</span>
                      <span className={`px-2 py-1 rounded text-[10px] font-bold ${
                        progressInfo.routing === 'hybrid' ? 'bg-[#6b9fff]/10 text-[#6b9fff]' :
                        progressInfo.routing === 'web' ? 'bg-[#fbbf24]/10 text-[#fbbf24]' :
                        'bg-[#22c55e]/10 text-[#22c55e]'
                      }`}>
                        {progressInfo.routing === 'hybrid' ? 'HYBRID (Docs + Web)' :
                         progressInfo.routing === 'web' ? 'WEB SEARCH' :
                         'DOCUMENT SEARCH'}
                      </span>
                    </div>
                  )}

                  {/* Show document counts */}
                  {(progressInfo.doc_count !== undefined || progressInfo.web_count !== undefined) && (
                    <div className="flex items-center gap-4 text-[11px]">
                      {progressInfo.doc_count !== undefined && progressInfo.doc_count > 0 && (
                        <div className="flex items-center gap-1.5">
                          <svg className="w-3 h-3 text-[#22c55e]" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                            <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                          </svg>
                          <span className="text-[#aaa]">{progressInfo.doc_count} document chunks</span>
                        </div>
                      )}
                      {progressInfo.web_count !== undefined && progressInfo.web_count > 0 && (
                        <div className="flex items-center gap-1.5">
                          <svg className="w-3 h-3 text-[#6b9fff]" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.083 9h1.946c.089-1.546.383-2.97.837-4.118A6.004 6.004 0 004.083 9zM10 2a8 8 0 100 16 8 8 0 000-16zm0 2c-.076 0-.232.032-.465.262-.238.234-.497.623-.737 1.182-.389.907-.673 2.142-.766 3.556h3.936c-.093-1.414-.377-2.649-.766-3.556-.24-.56-.5-.948-.737-1.182C10.232 4.032 10.076 4 10 4zm3.971 5c-.089-1.546-.383-2.97-.837-4.118A6.004 6.004 0 0115.917 9h-1.946zm-2.003 2H8.032c.093 1.414.377 2.649.766 3.556.24.56.5.948.737 1.182.233.23.389.262.465.262.076 0 .232-.032.465-.262.238-.234.498-.623.737-1.182.389-.907.673-2.142.766-3.556zm1.166 4.118c.454-1.147.748-2.572.837-4.118h1.946a6.004 6.004 0 01-2.783 4.118zm-6.268 0C6.412 13.97 6.118 12.546 6.03 11H4.083a6.004 6.004 0 002.783 4.118z" clipRule="evenodd" />
                          </svg>
                          <span className="text-[#aaa]">{progressInfo.web_count} web sources</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Show relevance statistics */}
                  {progressInfo.relevant_count !== undefined && progressInfo.total_count !== undefined && progressInfo.total_count > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-[#aaa] text-[11px] font-semibold uppercase tracking-wide">Quality:</span>
                      <span className="text-[#22c55e] text-[11px] font-bold">
                        {progressInfo.relevant_count}/{progressInfo.total_count} highly relevant
                      </span>
                      <div className="flex-1 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-[#22c55e] to-[#16a34a] rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(0, (progressInfo.relevant_count / progressInfo.total_count) * 100))}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Answer Text */}
          {isRewriting ? (
            <div className="py-8">
              <div className="text-center mb-6">
                <p className="text-[#888] text-sm" style={{ fontFamily: 'Product Sans, sans-serif' }}>
                  {currentStage?.message || 'Refining answer...'}
                </p>
              </div>
              {/* Green loading bar - dynamically animated */}
              <div className="w-full bg-[#1a1a1a] rounded-full h-2 overflow-hidden relative">
                <div className="absolute inset-0 h-full bg-gradient-to-r from-[#22c55e] to-[#16a34a] rounded-full animate-loading-bar"></div>
                <div className="absolute inset-0 h-full bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer-slow"></div>
              </div>
            </div>
          ) : displayText ? (
            <div
              ref={answerRef}
              className={`text-[#d0d0d0] text-[17px] leading-relaxed max-h-[600px] overflow-y-auto custom-scrollbar ${
                isProvisional ? 'opacity-90' : 'opacity-100'
              }`}
              style={{ fontFamily: 'Product Sans, sans-serif' }}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  // Paragraphs
                  p: ({ children }) => (
                    <div className="mb-4">{children}</div>
                  ),
                  // Bold text
                  strong: ({ children }) => (
                    <span className="font-bold text-white">{children}</span>
                  ),
                  // Italic text
                  em: ({ children }) => (
                    <span className="italic text-[#e0e0e0]">{children}</span>
                  ),
                  // Unordered lists
                  ul: ({ children }) => (
                    <ul className="list-disc pl-6 mb-4 space-y-2">{children}</ul>
                  ),
                  // Ordered lists
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-6 mb-4 space-y-2">{children}</ol>
                  ),
                  // List items
                  li: ({ children }) => (
                    <li className="mb-2">{children}</li>
                  ),
                  // Headings
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-bold text-white mt-6 mb-4">{children}</h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-xl font-bold text-white mt-5 mb-3">{children}</h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-lg font-bold text-white mt-4 mb-2">{children}</h3>
                  ),
                  // Code blocks
                  code: ({ inline, children }) =>
                    inline ? (
                      <code className="bg-[#1a1a1a] text-[#22c55e] px-2 py-1 rounded text-sm font-mono">
                        {children}
                      </code>
                    ) : (
                      <code className="block bg-[#1a1a1a] text-[#22c55e] p-4 rounded-lg mb-4 font-mono text-sm overflow-x-auto">
                        {children}
                      </code>
                    ),
                  // Blockquotes
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-[#6b9fff] pl-4 mb-4 italic text-[#aaa]">
                      {children}
                    </blockquote>
                  ),
                  // Links
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#6b9fff] hover:text-[#8bb3ff] underline"
                    >
                      {children}
                    </a>
                  ),
                }}
              >
                {displayText}
              </ReactMarkdown>
              {isProvisional && (
                <span className="inline-block w-2 h-5 bg-[#22c55e] ml-1 animate-blink"></span>
              )}
            </div>
          ) : (
            <div className="py-8">
              {/* Green loading bar - dynamically animated */}
              <div className="w-full bg-[#1a1a1a] rounded-full h-2 overflow-hidden relative">
                <div className="absolute inset-0 h-full bg-gradient-to-r from-[#22c55e] to-[#16a34a] rounded-full animate-loading-bar"></div>
                <div className="absolute inset-0 h-full bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer-slow"></div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StreamingAnswerDisplay;
