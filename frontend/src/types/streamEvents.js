/**
 * Event Type Definitions for Streaming RAG
 *
 * These types match the backend event schema and provide
 * type safety for event handling in the frontend.
 */

// Event type constants for type checking
export const EVENT_TYPES = {
  PROVISIONAL_TOKEN: 'provisional_token',
  STAGE: 'stage',
  VALIDATION_SUCCESS: 'validation_success',
  REWRITE: 'rewrite',
  FINAL_ANSWER: 'final_answer',
  ERROR: 'error',
  END: 'end'
};

// Stage constants
export const STAGES = {
  ANALYZING: 'analyzing',
  RETRIEVING: 'retrieving',
  GENERATING: 'generating',
  VALIDATING: 'validating'
};

// Connection states
export const CONNECTION_STATES = {
  IDLE: 'idle',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error',
  CLOSED: 'closed'
};

/**
 * Validate event structure
 * @param {object} event - Event object to validate
 * @returns {boolean} - True if valid
 */
export function isValidEvent(event) {
  if (!event || typeof event !== 'object') return false;
  if (!event.type || typeof event.type !== 'string') return false;
  return Object.values(EVENT_TYPES).includes(event.type);
}

/**
 * Safe JSON parse with error handling
 * @param {string} data - JSON string to parse
 * @returns {object|null} - Parsed object or null on error
 */
export function safeJSONParse(data) {
  try {
    const parsed = JSON.parse(data);
    return parsed;
  } catch (error) {
    console.error('Failed to parse event data:', error);
    return null;
  }
}
