/**
 * UI Module - Handles DOM manipulation and UI updates
 * Provides methods to update the chat interface and handle interactions
 */

const ui = {
    // DOM element references
    elements: {
        chatMessages: document.getElementById('chat-messages'),
        chatInput: document.getElementById('chat-input'),
        sendButton: document.getElementById('send-button'),
        loadingIndicator: document.getElementById('loading-indicator'),
        optionsPanel: document.getElementById('options-panel'),
        toggleOptionsButton: document.getElementById('toggle-options'),
        includeContextCheckbox: document.getElementById('include-context'),
        includeEvaluationCheckbox: document.getElementById('include-evaluation'),
        streamResponseCheckbox: document.getElementById('stream-response')
    },

    /**
     * Initialize UI elements and event listeners
     */
    init: function() {
        // Set up options panel toggle
        this.elements.toggleOptionsButton.addEventListener('click', () => {
            this.elements.optionsPanel.classList.toggle('open');
        });

        // Set up auto-resize for the textarea
        this.elements.chatInput.addEventListener('input', this.autoResizeTextarea.bind(this));

        // Set up the Enter key to send messages (Shift+Enter for new line)
        this.elements.chatInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.elements.sendButton.click();
            }
        });
    },

    /**
     * Auto-resize the textarea based on content
     */
    autoResizeTextarea: function() {
        const textarea = this.elements.chatInput;
        
        // Reset height to calculate correct scrollHeight
        textarea.style.height = 'auto';
        
        // Set new height (limited by CSS max-height)
        textarea.style.height = (textarea.scrollHeight) + 'px';
    },

    /**
     * Add a message to the chat display
     * 
     * @param {string} content - Message content
     * @param {string} role - Message role ('user' or 'assistant')
     * @param {boolean} isComplete - Whether the message is complete (for streaming)
     * @returns {HTMLElement} - The message element
     */
    addMessage: function(content, role, isComplete = true) {
        // Check if there's an incomplete message we should update
        if (!isComplete && role === 'assistant') {
            const lastMessage = this.elements.chatMessages.lastElementChild;
            if (lastMessage && lastMessage.classList.contains('assistant-message')) {
                const contentElement = lastMessage.querySelector('.message-content');
                if (contentElement) {
                    contentElement.textContent = content;
                    return lastMessage;
                }
            }
        }

        // Create new message element
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        this.elements.chatMessages.appendChild(messageDiv);
        
        // Scroll to the bottom
        this.scrollToBottom();
        
        return messageDiv;
    },

    /**
     * Format message content with simple markdown-like syntax
     * 
     * @param {string} content - Raw message content
     * @returns {string} - Formatted HTML content
     */
    formatMessageContent: function(content) {
        // This is a simple implementation. For production, consider using a markdown library.
        // Replace code blocks
        content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Replace inline code
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Replace paragraphs
        content = content.split('\n\n').map(p => `<p>${p}</p>`).join('');
        
        return content;
    },

    /**
     * Update the content of an existing message (for streaming)
     * 
     * @param {HTMLElement} messageElement - The message element to update
     * @param {string} content - New content
     */
    updateMessage: function(messageElement, content) {
        const contentElement = messageElement.querySelector('.message-content');
        if (contentElement) {
            contentElement.innerHTML = this.formatMessageContent(content);
        }
        this.scrollToBottom();
    },

    /**
     * Scroll the chat messages container to the bottom
     */
    scrollToBottom: function() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    },

    /**
     * Show the loading indicator
     */
    showLoading: function() {
        this.elements.loadingIndicator.style.display = 'flex';
        this.elements.sendButton.disabled = true;
        this.elements.chatInput.disabled = true;
    },

    /**
     * Hide the loading indicator
     */
    hideLoading: function() {
        this.elements.loadingIndicator.style.display = 'none';
        this.elements.sendButton.disabled = false;
        this.elements.chatInput.disabled = false;
        this.elements.chatInput.focus();
    },

    /**
     * Clear the input field
     */
    clearInput: function() {
        this.elements.chatInput.value = '';
        this.elements.chatInput.style.height = 'auto';
    },

    /**
     * Get the current options from the UI
     * 
     * @returns {Object} - Object containing the current options
     */
    getOptions: function() {
        return {
            includeContext: this.elements.includeContextCheckbox.checked,
            includeEvaluation: this.elements.includeEvaluationCheckbox.checked,
            streamResponse: this.elements.streamResponseCheckbox.checked
        };
    }
};