/**
 * API Module - Handles communication with the Marketing Knowledge Base API
 * Provides methods for sending chat messages and processing responses
 */

// API endpoint constants
const API_ENDPOINTS = {
    CHAT: '/api/chat/message',
    STREAM: '/api/chat/stream'
};

/**
 * API client object
 */
const api = {
    /**
     * Send a message to the chat API
     * 
     * @param {Array} messages - Array of message objects with role and content
     * @param {Object} options - Additional options like includeContext
     * @returns {Promise} - Promise resolving to the API response
     */
    sendMessage: async function(messages, options = {}) {
        try {
            const requestBody = {
                messages: messages,
                include_context: options.includeContext || false,
                include_evaluation: options.includeEvaluation || false,
                stream: false
            };

            const response = await fetch(API_ENDPOINTS.CHAT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    },

    /**
     * Stream a chat response from the API
     * 
     * @param {Array} messages - Array of message objects with role and content
     * @param {Object} options - Additional options like includeContext
     * @param {Function} onChunk - Callback function for each chunk of streamed response
     * @returns {Promise} - Promise that resolves when streaming is complete
     */
    streamMessage: async function(messages, options = {}, onChunk) {
        try {
            const requestBody = {
                messages: messages,
                include_context: options.includeContext || false,
                include_evaluation: options.includeEvaluation || false,
                stream: true
            };

            const response = await fetch(API_ENDPOINTS.STREAM, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} ${response.statusText}`);
            }

            // Process the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                    break;
                }
                
                const chunk = decoder.decode(value, { stream: true });
                
                // Process server-sent events (SSE)
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (onChunk && data) {
                            onChunk(data);
                            accumulatedText += data;
                        }
                    }
                }
            }

            return accumulatedText;
        } catch (error) {
            console.error('Error streaming message:', error);
            throw error;
        }
    }
};