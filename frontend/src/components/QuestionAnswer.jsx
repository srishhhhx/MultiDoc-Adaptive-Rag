import { useState } from "react";
import axios from "axios";
import TypewriterFormattedAnswer from "./TypewriterFormattedAnswer";
import CircularGauge from "./CircularGauge";
import ProgressBar from "./ProgressBar";

const QuestionAnswer = ({ sessionId }) => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [error, setError] = useState(null);
  const [isComplete, setIsComplete] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setIsComplete(false);

    try {
      const response = await axios.post('http://localhost:8000/api/ask', {
        question: question,
        session_id: sessionId
      });

      setAnswer(response.data);
      setIsComplete(true);

      // Small delay to show 100% before hiding progress bar
      setTimeout(() => {
        setLoading(false);
      }, 500);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to get answer');
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl">
      {/* Question Input Section */}
      <div className="bg-gradient-to-br from-[#111] to-[#0a0a0a] border border-[#222] rounded-3xl p-8 mb-6 shadow-2xl shadow-black/50">
        <h2 className="text-white text-2xl font-bold mb-6 tracking-tight uppercase text-[13px] text-[#888]" style={{fontFamily: 'Product Sans, sans-serif'}}>
          Ask a Question
        </h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask anything about your document..."
              className="w-full bg-[#0f0f0f] border border-[#222] rounded-2xl p-5 text-white text-[17px] font-normal placeholder-[#444] focus:outline-none focus:border-[#22c55e]/50 focus:shadow-lg focus:shadow-[#22c55e]/5 transition-all resize-none"
              style={{fontFamily: 'Product Sans, sans-serif'}}
              rows={3}
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className={`w-full h-14 border border-[#333] rounded-2xl cursor-pointer relative overflow-hidden transition-all duration-300 group ${
              loading || !question.trim() ? 'opacity-40 cursor-not-allowed' : 'hover:border-[#444] hover:shadow-lg hover:shadow-black/50'
            }`}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f]"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative z-10 text-white text-[16px] font-semibold h-full flex items-center justify-center gap-2 tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
              {loading ? (
                <span>Processing...</span>
              ) : (
                <>
                  <span>Ask Question</span>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2.5}
                    stroke="currentColor"
                    className="w-4 h-4 group-hover:translate-x-1 transition-transform"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                  </svg>
                </>
              )}
            </div>
          </button>
        </form>
      </div>

      {/* Progress Bar */}
      <ProgressBar
        isVisible={loading}
        isComplete={isComplete}
        onComplete={() => {
          // Progress bar completion is handled by the actual API response
        }}
      />

      {/* Error Message */}
      {error && (
        <div className="bg-gradient-to-r from-[#2a0a0a] to-[#1a0505] border border-[#ff6b6b]/20 rounded-2xl p-4 mb-6 backdrop-blur-sm">
          <div className="text-[#ff6b6b] text-sm font-semibold flex items-center gap-2">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Answer Section */}
      {answer && (
        <div className="space-y-10">
          {/* Answer Block */}
          <div className="relative">
            {/* Gradient Corner Accent */}
            <div className="absolute -top-3 -left-3 w-24 h-24 bg-gradient-to-br from-[#22c55e]/20 to-transparent blur-2xl rounded-full"></div>
            
            <div className="relative bg-gradient-to-br from-[#0f0f0f] via-[#0a0a0a] to-black border border-[#1a1a1a] rounded-3xl p-8">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-1 h-8 bg-gradient-to-b from-[#22c55e] to-transparent rounded-full"></div>
                  <h2 className="text-white text-3xl font-bold tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
                    Response
                  </h2>
                </div>
                {answer.online_search ? (
                  <div className="flex items-center gap-2 text-[#6b9fff] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#6b9fff]/5 border border-[#6b9fff]/30">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                      <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                    </svg>
                    ONLINE
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-[#22c55e] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#22c55e]/5 border border-[#22c55e]/30">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                    </svg>
                    DOCUMENT
                  </div>
                )}
              </div>
              
              {/* Answer Text */}
              <div className="text-[#d0d0d0] text-[20px] leading-relaxed" style={{fontFamily: 'Product Sans, sans-serif'}}>
                <TypewriterFormattedAnswer text={answer.answer} />
              </div>
            </div>
          </div>

          {/* Evaluation Scores */}
          {(answer.question_relevance_score || answer.document_relevance_score) && (
            <div className="relative">
              {/* Gradient Corner Accent */}
              <div className="absolute -top-3 -right-3 w-24 h-24 bg-gradient-to-bl from-[#6b9fff]/20 to-transparent blur-2xl rounded-full"></div>
              
              <div className="relative bg-gradient-to-br from-[#0f0f0f] via-[#0a0a0a] to-black border border-[#1a1a1a] rounded-3xl p-8">
                {/* Header */}
                <div className="flex items-center gap-3 mb-8">
                  <div className="w-1 h-8 bg-gradient-to-b from-[#6b9fff] to-transparent rounded-full"></div>
                  <h3 className="text-white text-3xl font-bold tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
                    Quality Metrics
                  </h3>
                </div>
                
                {/* Metrics Display with Circular Gauges */}
                <div className="grid grid-cols-2 gap-8">
                  {/* Question-Answer Relevance */}
                  {answer.question_relevance_score && (
                    <div className="flex flex-col items-center">
                      <CircularGauge
                        value={answer.question_relevance_score.relevance_score !== null ? (answer.question_relevance_score.relevance_score * 100) : (answer.question_relevance_score.binary_score ? 85 : 15)}
                        max={100}
                        size="lg"
                        color={answer.question_relevance_score.binary_score ? 'green' : 'red'}
                        label="Question Relevance"
                      />
                      <div className="mt-4 text-center">
                        <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${
                          answer.question_relevance_score.binary_score 
                            ? 'bg-[#22c55e]/10 border border-[#22c55e]/30' 
                            : 'bg-[#ff6b6b]/10 border border-[#ff6b6b]/30'
                        }`}>
                          <div className={`w-2 h-2 rounded-full ${
                            answer.question_relevance_score.binary_score ? 'bg-[#22c55e]' : 'bg-[#ff6b6b]'
                          }`}></div>
                          <span className={`text-[11px] font-bold ${
                            answer.question_relevance_score.binary_score ? 'text-[#22c55e]' : 'text-[#ff6b6b]'
                          }`}>
                            {answer.question_relevance_score.binary_score ? 'MATCHED' : 'POOR'}
                          </span>
                        </div>
                      </div>
                      {answer.question_relevance_score.completeness && (
                        <div className="mt-3">
                          <span className={`text-[11px] font-semibold uppercase px-3 py-1 rounded-full ${
                            answer.question_relevance_score.completeness === 'complete'
                              ? 'text-[#22c55e] bg-[#22c55e]/10'
                              : answer.question_relevance_score.completeness === 'partial'
                              ? 'text-[#fbbf24] bg-[#fbbf24]/10'
                              : 'text-[#ff6b6b] bg-[#ff6b6b]/10'
                          }`}>
                            {answer.question_relevance_score.completeness}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Document Grounding */}
                  {answer.document_relevance_score && (
                    <div className="flex flex-col items-center">
                      <CircularGauge
                        value={answer.document_relevance_score.confidence !== null ? (answer.document_relevance_score.confidence * 100) : (answer.document_relevance_score.binary_score ? 90 : 10)}
                        max={100}
                        size="lg"
                        color={answer.document_relevance_score.binary_score ? 'blue' : 'red'}
                        label="Answer Grounding"
                      />
                      <div className="mt-4">
                        <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${
                          answer.document_relevance_score.binary_score 
                            ? 'bg-[#22c55e]/10 border border-[#22c55e]/30' 
                            : 'bg-[#ff6b6b]/10 border border-[#ff6b6b]/30'
                        }`}>
                          <div className={`w-2 h-2 rounded-full ${
                            answer.document_relevance_score.binary_score ? 'bg-[#22c55e]' : 'bg-[#ff6b6b]'
                          }`}></div>
                          <span className={`text-[11px] font-bold ${
                            answer.document_relevance_score.binary_score ? 'text-[#22c55e]' : 'text-[#ff6b6b]'
                          }`}>
                            {answer.document_relevance_score.binary_score ? 'GROUNDED' : 'NOT GROUNDED'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Additional Details */}
                {answer.question_relevance_score?.missing_aspects && (
                  <div className="mt-8 pt-6 border-t border-[#1a1a1a]/50">
                    <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-2xl p-4">
                      <div className="text-[#888] text-[11px] font-semibold uppercase tracking-wide mb-2">Missing Aspects</div>
                      <div className="text-[#aaa] text-[13px] leading-relaxed">{answer.question_relevance_score.missing_aspects}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Document Evaluations (Metric #1) */}
          {answer.document_evaluations && answer.document_evaluations.length > 0 && (
            <div className="relative">
              {/* Gradient Corner Accent */}
              <div className="absolute -top-3 -right-3 w-24 h-24 bg-gradient-to-bl from-[#fbbf24]/20 to-transparent blur-2xl rounded-full"></div>
              
              <div className="relative bg-gradient-to-br from-[#0f0f0f] via-[#0a0a0a] to-black border border-[#1a1a1a] rounded-3xl p-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-gradient-to-b from-[#fbbf24] to-transparent rounded-full"></div>
                    <h3 className="text-white text-3xl font-bold tracking-tight" style={{fontFamily: 'Product Sans, sans-serif'}}>
                      Document Analysis
                    </h3>
                  </div>
                  <div className="flex items-center gap-2 text-[#fbbf24] text-[10px] font-bold px-3 py-1.5 rounded-full bg-[#fbbf24]/5 border border-[#fbbf24]/30">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                      <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                    </svg>
                    {(() => {
                      const validScores = answer.document_evaluations
                        .filter(e => e.relevance_score !== null && e.relevance_score !== undefined)
                        .map(e => e.relevance_score);
                      if (validScores.length === 0) return `${answer.document_evaluations.length} DOCS`;
                      const avgScore = Math.round((validScores.reduce((a, b) => a + b, 0) / validScores.length) * 100);
                      return `${avgScore}%`;
                    })()}
                  </div>
                </div>
                
                <div className="space-y-6">
                  {answer.document_evaluations.map((evaluation, idx) => (
                    <div key={idx} className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-2xl p-6 space-y-4 hover:border-[#222] transition-colors">
                      {/* Document Header */}
                      <div className="flex justify-between items-center pb-3 border-b border-[#1a1a1a]/50">
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm ${
                            evaluation.score.toLowerCase() === 'yes' 
                              ? 'bg-[#22c55e]/10 text-[#22c55e]' 
                              : 'bg-[#ff6b6b]/10 text-[#ff6b6b]'
                          }`}>
                            {idx + 1}
                          </div>
                          <span className="text-white text-[17px] font-semibold" style={{fontFamily: 'Product Sans, sans-serif'}}>Retrieved Chunk {idx + 1}</span>
                        </div>
                        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                          evaluation.score.toLowerCase() === 'yes' 
                            ? 'bg-[#22c55e]/10 border border-[#22c55e]/30' 
                            : 'bg-[#ff6b6b]/10 border border-[#ff6b6b]/30'
                        }`}>
                          <div className={`w-2 h-2 rounded-full ${
                            evaluation.score.toLowerCase() === 'yes' ? 'bg-[#22c55e]' : 'bg-[#ff6b6b]'
                          }`}></div>
                          <span className={`text-[11px] font-bold uppercase ${
                            evaluation.score.toLowerCase() === 'yes' 
                              ? 'text-[#22c55e]' 
                              : 'text-[#ff6b6b]'
                          }`}>
                            {evaluation.score}
                          </span>
                        </div>
                      </div>

                      {/* Relevance Score with Modern Radial Display */}
                      {evaluation.relevance_score !== null && evaluation.relevance_score !== undefined && (
                        <div className="relative">
                          <div className="flex items-center justify-between">
                            <span className="text-[#888] text-[11px] font-semibold uppercase tracking-wider">Relevance</span>
                            
                            {/* Circular Score Indicator */}
                            <div className="flex items-center gap-4">
                              <div className="relative w-16 h-16">
                                {/* Background circle */}
                                <svg className="w-16 h-16 transform -rotate-90">
                                  <circle
                                    cx="32"
                                    cy="32"
                                    r="28"
                                    stroke="rgba(255,255,255,0.05)"
                                    strokeWidth="6"
                                    fill="none"
                                  />
                                  <circle
                                    cx="32"
                                    cy="32"
                                    r="28"
                                    stroke={
                                      (evaluation.relevance_score * 100) >= 70 
                                        ? '#22c55e'
                                        : (evaluation.relevance_score * 100) >= 40
                                        ? '#fbbf24'
                                        : '#ff6b6b'
                                    }
                                    strokeWidth="6"
                                    fill="none"
                                    strokeDasharray={`${2 * Math.PI * 28}`}
                                    strokeDashoffset={`${2 * Math.PI * 28 * (1 - evaluation.relevance_score)}`}
                                    strokeLinecap="round"
                                    className="transition-all duration-1000 ease-out"
                                    style={{
                                      filter: `drop-shadow(0 0 6px ${
                                        (evaluation.relevance_score * 100) >= 70 
                                          ? 'rgba(34, 197, 94, 0.6)'
                                          : (evaluation.relevance_score * 100) >= 40
                                          ? 'rgba(251, 191, 36, 0.6)'
                                          : 'rgba(255, 107, 107, 0.6)'
                                      })`
                                    }}
                                  />
                                </svg>
                                {/* Center percentage */}
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <span className={`text-sm font-black ${
                                    (evaluation.relevance_score * 100) >= 70 
                                      ? 'text-[#22c55e]'
                                      : (evaluation.relevance_score * 100) >= 40
                                      ? 'text-[#fbbf24]'
                                      : 'text-[#ff6b6b]'
                                  }`}>
                                    {Math.round(evaluation.relevance_score * 100)}
                                  </span>
                                </div>
                              </div>
                              
                              {/* Status indicator */}
                              <div className={`px-3 py-1.5 rounded-lg ${
                                (evaluation.relevance_score * 100) >= 70 
                                  ? 'bg-[#22c55e]/10 text-[#22c55e]'
                                  : (evaluation.relevance_score * 100) >= 40
                                  ? 'bg-[#fbbf24]/10 text-[#fbbf24]'
                                  : 'bg-[#ff6b6b]/10 text-[#ff6b6b]'
                              }`}>
                                <div className="text-[10px] font-bold uppercase tracking-wider">
                                  {(evaluation.relevance_score * 100) >= 70 ? 'High' : (evaluation.relevance_score * 100) >= 40 ? 'Medium' : 'Low'}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Coverage Assessment */}
                      {evaluation.coverage_assessment && evaluation.coverage_assessment.trim() !== '' && (
                        <div className="bg-[#0f0f0f] border border-[#1a1a1a] rounded-xl p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <svg className="w-3.5 h-3.5 text-[#6b9fff]" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                            <div className="text-[#aaa] text-[11px] font-semibold uppercase tracking-wide">Coverage</div>
                          </div>
                          <div className="text-[#ccc] text-[13px] leading-relaxed">
                            {evaluation.coverage_assessment}
                          </div>
                        </div>
                      )}

                      {/* Missing Information */}
                      {evaluation.missing_information && evaluation.missing_information.trim() !== '' && (
                        <div className="bg-[#0f0f0f] border border-[#1a1a1a] rounded-xl p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <svg className="w-3.5 h-3.5 text-[#888]" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                            <div className="text-[#aaa] text-[11px] font-semibold uppercase tracking-wide">Missing Info</div>
                          </div>
                          <div className="text-[#ccc] text-[13px] leading-relaxed">
                            {evaluation.missing_information}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Reasoning Details */}
          {(answer.question_relevance_score?.reasoning || answer.document_relevance_score?.reasoning) && (
            <details className="pt-6 border-t border-[#1a1a1a] group">
              <summary className="cursor-pointer text-white text-[15px] font-semibold hover:text-[#22c55e] transition-colors list-none flex items-center gap-2">
                <span className="text-[#666] group-open:rotate-90 transition-transform">▸</span>
                Evaluation Reasoning
              </summary>
              <div className="mt-4 space-y-6 pl-6">
                {answer.question_relevance_score?.reasoning && (
                  <div className="pl-4 border-l-2 border-[#333]">
                    <h4 className="text-[#ddd] text-[13px] font-semibold mb-2">Question Relevance</h4>
                    <p className="text-[#aaa] text-[13px] leading-relaxed">
                      {answer.question_relevance_score.reasoning}
                    </p>
                  </div>
                )}
                {answer.document_relevance_score?.reasoning && (
                  <div className="pl-4 border-l-2 border-[#333]">
                    <h4 className="text-[#ddd] text-[13px] font-semibold mb-2">Document Relevance</h4>
                    <p className="text-[#aaa] text-[13px] leading-relaxed">
                      {answer.document_relevance_score.reasoning}
                    </p>
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
};

export default QuestionAnswer;
