import { useState, useEffect, useRef, useCallback } from 'react';
import { EVENT_TYPES, CONNECTION_STATES, isValidEvent, safeJSONParse } from '../types/streamEvents';

/**
 * useStreamingRAG Hook
 *
 * A robust, crash-resistant hook for managing EventSource streaming connections.
 *
 * Features:
 * - Automatic connection cleanup on unmount
 * - Prevents multiple concurrent connections
 * - Safe JSON parsing with error handling
 * - Proper state management to avoid race conditions
 * - Memory leak prevention
 * - Automatic error recovery
 *
 * @param {string} sessionId - Session ID for the RAG workflow
 * @returns {object} - Hook state and methods
 */
export function useStreamingRAG(sessionId) {
  // State management
  const [connectionState, setConnectionState] = useState(CONNECTION_STATES.IDLE);
  const [currentStage, setCurrentStage] = useState(null);
  const [progressInfo, setProgressInfo] = useState(null);
  const [provisionalAnswer, setProvisionalAnswer] = useState('');
  const [finalAnswer, setFinalAnswer] = useState(null);
  const [error, setError] = useState(null);
  const [isRewriting, setIsRewriting] = useState(false);
  const [attemptInfo, setAttemptInfo] = useState({ current: 0, max: 2 });
  const [qualityMetrics, setQualityMetrics] = useState(null);

  // Refs for connection management (survives re-renders)
  const eventSourceRef = useRef(null);
  const isMountedRef = useRef(true);
  const isConnectingRef = useRef(false);

  // Typewriter effect refs
  const typewriterBufferRef = useRef('');
  const typewriterIntervalRef = useRef(null);
  const drainIntervalRef = useRef(null); // Track drain interval to prevent premature cleanup
  const TYPEWRITER_SPEED = 20; // milliseconds per character (50 chars/second)
  const provisionalAnswerRef = useRef(''); // Track current provisional answer value

  /**
   * Cleanup function - CRITICAL for preventing memory leaks
   * Called on unmount and before new connections
   */
  const cleanup = useCallback(() => {
    console.log('[useStreamingRAG] Cleaning up connection');

    // CRITICAL: Don't cleanup typewriter if we're still draining buffer for final answer
    if (drainIntervalRef.current) {
      console.log('[useStreamingRAG] Skipping typewriter cleanup - buffer still draining');
      // Only cleanup the connection, not the typewriter
      if (eventSourceRef.current) {
        try {
          if (typeof eventSourceRef.current.close === 'function') {
            eventSourceRef.current.close();
          } else if (typeof eventSourceRef.current.abort === 'function') {
            eventSourceRef.current.abort();
          }
          console.log('[useStreamingRAG] Connection closed successfully');
        } catch (err) {
          console.error('[useStreamingRAG] Error closing connection:', err);
        }
        eventSourceRef.current = null;
      }
      isConnectingRef.current = false;
      return; // Don't clear typewriter or buffer
    }

    if (eventSourceRef.current) {
      try {
        // Handle both EventSource and fetch Reader cleanup
        if (typeof eventSourceRef.current.close === 'function') {
          // Old EventSource approach
          eventSourceRef.current.close();
        } else if (typeof eventSourceRef.current.abort === 'function') {
          // New fetch Reader approach
          eventSourceRef.current.abort();
        }
        console.log('[useStreamingRAG] Connection closed successfully');
      } catch (err) {
        console.error('[useStreamingRAG] Error closing connection:', err);
      }
      eventSourceRef.current = null;
    }

    isConnectingRef.current = false;

    // Clear typewriter interval
    if (typewriterIntervalRef.current) {
      clearInterval(typewriterIntervalRef.current);
      typewriterIntervalRef.current = null;
    }
    typewriterBufferRef.current = '';

    // Only update state if component is still mounted
    if (isMountedRef.current) {
      setConnectionState(CONNECTION_STATES.CLOSED);
    }
  }, []);

  /**
   * Safe state update wrapper - only updates if component is mounted
   */
  const safeSetState = useCallback((setter, value) => {
    if (isMountedRef.current) {
      setter(value);
    }
  }, []);

  /**
   * Start typewriter effect - slowly releases characters from buffer to display
   */
  const startTypewriter = useCallback(() => {
    // Clear existing interval if any
    if (typewriterIntervalRef.current) {
      clearInterval(typewriterIntervalRef.current);
    }

    console.log('[useStreamingRAG] Typewriter started');

    typewriterIntervalRef.current = setInterval(() => {
      if (!isMountedRef.current) {
        clearInterval(typewriterIntervalRef.current);
        return;
      }

      if (typewriterBufferRef.current.length > 0) {
        // Release 1-3 characters at a time for smoother effect
        const charsToRelease = Math.min(2, typewriterBufferRef.current.length);
        const chunk = typewriterBufferRef.current.substring(0, charsToRelease);
        typewriterBufferRef.current = typewriterBufferRef.current.substring(charsToRelease);

        console.log('[useStreamingRAG] Typewriter releasing', charsToRelease, 'chars, buffer remaining:', typewriterBufferRef.current.length);
        safeSetState(setProvisionalAnswer, prev => {
          const newValue = prev + chunk;
          provisionalAnswerRef.current = newValue; // Keep ref in sync
          return newValue;
        });
      }
    }, TYPEWRITER_SPEED);
  }, [safeSetState]);

  /**
   * Stop typewriter and flush remaining buffer
   */
  const stopTypewriter = useCallback(() => {
    if (typewriterIntervalRef.current) {
      clearInterval(typewriterIntervalRef.current);
      typewriterIntervalRef.current = null;
    }

    // Flush any remaining buffer
    if (typewriterBufferRef.current.length > 0 && isMountedRef.current) {
      safeSetState(setProvisionalAnswer, prev => prev + typewriterBufferRef.current);
      typewriterBufferRef.current = '';
    }
  }, [safeSetState]);

  /**
   * Reset all state to initial values
   */
  const resetState = useCallback(() => {
    if (!isMountedRef.current) return;

    setConnectionState(CONNECTION_STATES.IDLE);
    setCurrentStage(null);
    setProgressInfo(null);
    setProvisionalAnswer('');
    setFinalAnswer(null);
    setError(null);
    setIsRewriting(false);
    setAttemptInfo({ current: 0, max: 2 });
    setQualityMetrics(null);
    provisionalAnswerRef.current = ''; // Reset ref too
  }, []);

  /**
   * Handle incoming events from EventSource
   */
  const handleEvent = useCallback((event) => {
    // Safety check: only process if component is mounted
    if (!isMountedRef.current) {
      console.log('[useStreamingRAG] Ignoring event - component unmounted');
      return;
    }

    // Parse event data safely
    const parsedData = safeJSONParse(event.data);
    if (!parsedData) {
      console.error('[useStreamingRAG] Invalid event data');
      return;
    }

    // Validate event structure
    if (!isValidEvent(parsedData)) {
      console.error('[useStreamingRAG] Invalid event structure:', parsedData);
      return;
    }

    console.log('[useStreamingRAG] Received event:', parsedData.type);

    // Handle different event types
    switch (parsedData.type) {
      case EVENT_TYPES.STAGE:
        safeSetState(setCurrentStage, {
          stage: parsedData.stage,
          message: parsedData.message
        });
        safeSetState(setConnectionState, CONNECTION_STATES.CONNECTED);
        break;

      case EVENT_TYPES.PROGRESS:
        // Store progress information for display
        safeSetState(setProgressInfo, {
          stage: parsedData.stage,
          message: parsedData.message,
          routing: parsedData.routing,
          doc_count: parsedData.doc_count,
          web_count: parsedData.web_count,
          relevant_count: parsedData.relevant_count,
          total_count: parsedData.total_count
        });
        break;

      case EVENT_TYPES.PROVISIONAL_TOKEN:
        // Add token to typewriter buffer for slow display
        typewriterBufferRef.current += parsedData.content;
        console.log('[useStreamingRAG] Token received, buffer now:', typewriterBufferRef.current.length, 'chars');

        // Start typewriter if not already running
        if (!typewriterIntervalRef.current) {
          console.log('[useStreamingRAG] Starting typewriter');
          startTypewriter();
        }

        safeSetState(setAttemptInfo, prev => ({
          ...prev,
          current: parsedData.attempt
        }));
        safeSetState(setIsRewriting, false);
        break;

      case EVENT_TYPES.VALIDATION_SUCCESS:
        safeSetState(setCurrentStage, {
          stage: 'success',
          message: parsedData.message
        });
        break;

      case EVENT_TYPES.REWRITE:
        console.log('[useStreamingRAG] Rewrite triggered:', parsedData.reason);

        // Stop typewriter and clear buffer
        stopTypewriter();

        // Clear provisional answer for rewrite
        safeSetState(setProvisionalAnswer, '');
        provisionalAnswerRef.current = ''; // Clear ref too
        safeSetState(setIsRewriting, true);
        safeSetState(setAttemptInfo, {
          current: parsedData.attempt,
          max: parsedData.max_attempts
        });
        safeSetState(setCurrentStage, {
          stage: 'rewriting',
          message: parsedData.reason
        });
        break;

      case EVENT_TYPES.FINAL_ANSWER:
        console.log('[useStreamingRAG] Final answer received');
        console.log('[useStreamingRAG] Final answer event data:', parsedData);
        console.log('[useStreamingRAG] document_relevance:', parsedData.document_relevance);
        console.log('[useStreamingRAG] question_relevance:', parsedData.question_relevance);
        console.log('[useStreamingRAG] Typewriter interval running?', !!typewriterIntervalRef.current);
        console.log('[useStreamingRAG] Current buffer length:', typewriterBufferRef.current.length);

        // CRITICAL FIX: Don't set finalAnswer immediately!
        // Setting finalAnswer will override provisionalAnswer display and show full text at once
        // Instead: Wait for typewriter buffer to drain completely, THEN set final state

        if (typewriterIntervalRef.current) {
          // Typewriter is running - wait for buffer to drain naturally
          console.log('[useStreamingRAG] Waiting for typewriter buffer to drain...');

          drainIntervalRef.current = setInterval(() => {
            if (typewriterBufferRef.current.length === 0) {
              // Buffer is empty - typewriter finished displaying everything
              console.log('[useStreamingRAG] Buffer drained, setting final state');
              clearInterval(drainIntervalRef.current);
              drainIntervalRef.current = null; // Clear ref so cleanup can proceed
              clearInterval(typewriterIntervalRef.current);
              typewriterIntervalRef.current = null;

              // Now safe to set final answer
              // CRITICAL: Wait one more render cycle to ensure provisionalAnswer state has updated
              // This prevents flash by ensuring displayed text matches finalAnswer
              setTimeout(() => {
                if (isMountedRef.current) {
                  console.log('[useStreamingRAG] Setting final answer after buffer drain');
                  console.log('[useStreamingRAG] Final answer length:', parsedData.content.length);
                  console.log('[useStreamingRAG] Provisional ref length:', provisionalAnswerRef.current.length);

                  // Use parsedData.content as the authoritative source
                  safeSetState(setFinalAnswer, parsedData.content);
                  safeSetState(setCurrentStage, null); // Clear validation stage

                  const metrics = {
                    document_relevance: parsedData.document_relevance,
                    question_relevance: parsedData.question_relevance,
                    total_attempts: parsedData.total_attempts,
                    document_evaluations: parsedData.document_evaluations || []
                  };
                  console.log('[useStreamingRAG] Setting quality metrics:', metrics);
                  safeSetState(setQualityMetrics, metrics);
                  safeSetState(setIsRewriting, false);

                  // Now cleanup can proceed safely
                  cleanup();
                }
              }, 100); // Small delay to ensure state updates have propagated
            }
          }, 50); // Check every 50ms if buffer is empty
        } else {
          // No typewriter running (shouldn't happen but handle it)
          console.log('[useStreamingRAG] No typewriter, setting final state immediately');
          safeSetState(setFinalAnswer, parsedData.content);
          safeSetState(setCurrentStage, null);
          safeSetState(setQualityMetrics, {
            document_relevance: parsedData.document_relevance,
            question_relevance: parsedData.question_relevance,
            total_attempts: parsedData.total_attempts,
            document_evaluations: parsedData.document_evaluations || []
          });
          safeSetState(setIsRewriting, false);
        }
        break;

      case EVENT_TYPES.ERROR:
        console.error('[useStreamingRAG] Error event:', parsedData.message);
        safeSetState(setError, parsedData.message);
        safeSetState(setConnectionState, CONNECTION_STATES.ERROR);
        cleanup();
        break;

      case EVENT_TYPES.END:
        console.log('[useStreamingRAG] Stream ended:', parsedData.success ? 'success' : 'failure');
        safeSetState(setConnectionState, CONNECTION_STATES.CLOSED);
        cleanup();
        break;

      default:
        console.warn('[useStreamingRAG] Unknown event type:', parsedData.type);
    }
  }, [cleanup, safeSetState]);


  /**
   * Start streaming for a question
   * Uses fetch with ReadableStream for robust POST-based streaming
   * @param {string} question - Question to ask
   */
  const startStreaming = useCallback(async (question) => {
    // Prevent multiple concurrent connections
    if (isConnectingRef.current || eventSourceRef.current) {
      console.warn('[useStreamingRAG] Connection already active, cleaning up first');
      cleanup();
    }

    if (!question || !question.trim()) {
      safeSetState(setError, 'Question cannot be empty');
      return;
    }

    if (!sessionId) {
      safeSetState(setError, 'Session ID is required');
      return;
    }

    console.log('[useStreamingRAG] Starting streaming for question:', question);

    // Reset state for new question
    resetState();
    isConnectingRef.current = true;
    safeSetState(setConnectionState, CONNECTION_STATES.CONNECTING);

    try {
      // Use fetch with streaming response
      const response = await fetch('/api/ask-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          question: question,
          session_id: sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      safeSetState(setConnectionState, CONNECTION_STATES.CONNECTED);
      isConnectingRef.current = false;

      // Get the reader from the response body
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Store a reference for cleanup (we'll store the reader abort function)
      eventSourceRef.current = { reader, abort: () => reader.cancel() };

      // Buffer for incomplete SSE messages
      let buffer = '';

      // Read the stream
      let chunkCount = 0;
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          console.log('[useStreamingRAG] Stream complete');
          break;
        }

        chunkCount++;
        // Decode the chunk
        const decodedChunk = decoder.decode(value, { stream: true });
        console.log(`[useStreamingRAG] 📦 Chunk #${chunkCount} received (${decodedChunk.length} bytes)`);
        console.log(`[useStreamingRAG]    First 200 chars: ${decodedChunk.substring(0, 200)}`);

        buffer += decodedChunk;

        // Split by double newline (SSE message boundary) - handle both \n\n and \r\n\r\n
        const messages = buffer.split(/\r?\n\r?\n/);
        console.log(`[useStreamingRAG] 📨 Found ${messages.length - 1} complete messages in buffer`);

        // Keep the last incomplete message in the buffer
        buffer = messages.pop() || '';

        // Process complete messages
        for (const message of messages) {
          if (!message.trim()) continue;

          console.log(`[useStreamingRAG] 🔍 Processing message: ${message.substring(0, 100)}...`);

          // Parse SSE format: "event: message\ndata: {...}" or just "data: {...}"
          const lines = message.split(/\r?\n/);
          let eventType = 'message';
          let data = null;

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('event:')) {
              eventType = trimmedLine.substring(6).trim();
            } else if (trimmedLine.startsWith('data:')) {
              const dataStr = trimmedLine.substring(5).trim();
              console.log(`[useStreamingRAG] 📄 Parsing data: ${dataStr.substring(0, 100)}...`);
              data = safeJSONParse(dataStr);
            }
          }

          if (data) {
            console.log(`[useStreamingRAG] ✅ Successfully parsed event type: ${data.type}`);
            // Simulate EventSource message event format
            handleEvent({ data: JSON.stringify(data) });
          } else {
            console.warn(`[useStreamingRAG] ⚠️ Failed to parse data from message`);
          }
        }
      }

      // Stream ended normally
      safeSetState(setConnectionState, CONNECTION_STATES.CLOSED);
      cleanup();

    } catch (err) {
      console.error('[useStreamingRAG] Streaming error:', err);

      if (isMountedRef.current) {
        safeSetState(setError, err.message || 'Failed to establish connection');
        safeSetState(setConnectionState, CONNECTION_STATES.ERROR);
      }

      isConnectingRef.current = false;
      cleanup();
    }
  }, [sessionId, cleanup, resetState, handleEvent, safeSetState]);

  /**
   * Cancel ongoing streaming
   */
  const cancelStreaming = useCallback(() => {
    console.log('[useStreamingRAG] Cancelling streaming');
    cleanup();
    resetState();
  }, [cleanup, resetState]);

  /**
   * Cleanup on unmount - CRITICAL
   */
  useEffect(() => {
    isMountedRef.current = true;

    return () => {
      console.log('[useStreamingRAG] Component unmounting, cleaning up');
      isMountedRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  return {
    // State
    connectionState,
    currentStage,
    progressInfo,
    provisionalAnswer,
    finalAnswer,
    error,
    isRewriting,
    attemptInfo,
    qualityMetrics,

    // Methods
    startStreaming,
    cancelStreaming,
    resetState,

    // Computed
    isStreaming: connectionState === CONNECTION_STATES.CONNECTING || connectionState === CONNECTION_STATES.CONNECTED,
    isConnected: connectionState === CONNECTION_STATES.CONNECTED,
    hasError: connectionState === CONNECTION_STATES.ERROR || error !== null,
  };
}
