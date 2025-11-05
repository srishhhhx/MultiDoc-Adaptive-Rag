import React from 'react';
import { STAGES } from '../types/streamEvents';

/**
 * StatusIndicator Component
 *
 * Shows the current stage of the RAG pipeline with visual indicators.
 * Displays:
 * - Current pipeline stage (analyzing, retrieving, generating, validating)
 * - Progress animation
 * - Stage-specific icons and colors
 */
const StatusIndicator = ({ currentStage, isVisible }) => {
  if (!isVisible || !currentStage) return null;

  // Stage configurations
  const stageConfig = {
    [STAGES.ANALYZING]: {
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      color: '#6b9fff',
      bgColor: 'bg-[#6b9fff]/10',
      borderColor: 'border-[#6b9fff]/30'
    },
    [STAGES.RETRIEVING]: {
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      ),
      color: '#8b5cf6',
      bgColor: 'bg-[#8b5cf6]/10',
      borderColor: 'border-[#8b5cf6]/30'
    },
    [STAGES.GENERATING]: {
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
      ),
      color: '#22c55e',
      bgColor: 'bg-[#22c55e]/10',
      borderColor: 'border-[#22c55e]/30'
    },
    [STAGES.VALIDATING]: {
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: '#fbbf24',
      bgColor: 'bg-[#fbbf24]/10',
      borderColor: 'border-[#fbbf24]/30'
    },
    'rewriting': {
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      ),
      color: '#f97316',
      bgColor: 'bg-[#f97316]/10',
      borderColor: 'border-[#f97316]/30'
    },
    'success': {
      icon: (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
      ),
      color: '#22c55e',
      bgColor: 'bg-[#22c55e]/10',
      borderColor: 'border-[#22c55e]/30'
    }
  };

  const config = stageConfig[currentStage.stage] || stageConfig[STAGES.GENERATING];

  return (
    <div className="w-full max-w-4xl mb-6">
      <div
        className={`${config.bgColor} ${config.borderColor} border rounded-2xl p-4 backdrop-blur-sm transition-all duration-300 animate-slideDown`}
      >
        <div className="flex items-center gap-3">
          {/* Animated Icon */}
          <div
            className="flex-shrink-0 animate-pulse"
            style={{ color: config.color }}
          >
            {config.icon}
          </div>

          {/* Progress Bar */}
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-sm font-semibold"
                style={{ color: config.color, fontFamily: 'Product Sans, sans-serif' }}
              >
                {currentStage.message}
              </span>
            </div>
            <div className="h-1 bg-[#1a1a1a] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full animate-progressBar"
                style={{ backgroundColor: config.color }}
              ></div>
            </div>
          </div>

          {/* Spinner for active stages */}
          {currentStage.stage !== 'success' && (
            <div className="flex-shrink-0">
              <div
                className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin"
                style={{ borderColor: `${config.color}40`, borderTopColor: 'transparent' }}
              ></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StatusIndicator;
