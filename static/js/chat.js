/**
 * Chat Module - Main application logic for the chat interface
 * Connects the UI module with the API module
 */

// Chat state management
const chatState = {
    messages: [],
    isProcessing: false
};

/**
 * Initialize the chat interface
 */
function initChat() {
    // Initialize UI elements and event listeners
    ui.init();

    // Add event listener for the send button
    ui.elements.sendButton.addEventListener('click', handleMessageSubmit);

    // Add welcome message
    addInitialMessages();
}

/**
 * Add initial welcome messages to the chat
 */
function addInitialMessages() {
    const welcomeMessage = "👋 Welcome to the Marketing Knowledge Base! I can help you find information about company metrics, marketing strategies, and other company knowledge. How can I assist you today?";
    
    ui.addMessage(welcomeMessage, 'assistant');
    
    // Add to message history
    chatState.messages.push({
        role: 'assistant',
        content: welcomeMessage
    });
}

/**
 * Handle message submission
 */
async function handleMessageSubmit() {
    // Get user input
    const userInput = ui.elements.chatInput.value.trim();
    
    // Validate input
    if (!userInput || chatState.isProcessing) {
        return;
    }
    
    try {
        // Update state
        chatState.isProcessing = true;
        
        // Add user message to UI
        ui.addMessage(userInput, 'user');
        
        // Clear input
        ui.clearInput();
        
        // Add user message to history
        chatState.messages.push({
            role: 'user',
            content: userInput
        });
        
        // Get current options
        const options = ui.getOptions();
        
        // Show loading indicator
        ui.showLoading();
        
        // Process the message
        if (options.streamResponse) {
            await handleStreamingResponse(options);
        } else {
            await handleRegularResponse(options);
        }
    } catch (error) {
        console.error('Error processing message:', error);
        ui.addMessage(`Sorry, there was an error processing your request: ${error.message}`, 'assistant');
    } finally {
        // Update state
        chatState.isProcessing = false;
        
        // Hide loading indicator
        ui.hideLoading();
    }
}

/**
 * Handle regular (non-streaming) response
 * 
 * @param {Object} options - Chat options
 */
async function handleRegularResponse(options) {
    const response = await api.sendMessage(chatState.messages, {
        includeContext: options.includeContext,
        includeEvaluation: options.includeEvaluation
    });
    
    // Add assistant response to UI
    if (response && response.message) {
        ui.addMessage(response.message.content, 'assistant');
        
        // Add to message history
        chatState.messages.push({
            role: 'assistant',
            content: response.message.content
        });
        
        // Handle context display if requested and available
        if (options.includeContext && response.context) {
            displayContext(response.context);
        }
        
        // Handle evaluation display if requested and available
        if (options.includeEvaluation && response.evaluation) {
            displayEvaluation(response.evaluation);
        }
    }
}

/**
 * Handle streaming response
 * 
 * @param {Object} options - Chat options
 */
async function handleStreamingResponse(options) {
    let accumulatedResponse = '';
    let messageElement = ui.addMessage('', 'assistant', false);
    
    await api.streamMessage(
        chatState.messages,
        {
            includeContext: options.includeContext,
            includeEvaluation: options.includeEvaluation
        },
        (chunk) => {
            // Update accumulated response
            accumulatedResponse += chunk;
            
            // Update UI with the current accumulated response
            ui.updateMessage(messageElement, accumulatedResponse);
        }
    );
    
    // Add to message history
    chatState.messages.push({
        role: 'assistant',
        content: accumulatedResponse
    });
}

/**
 * Display context information in the chat
 * 
 * @param {Object} context - Context object from the API
 */
function displayContext(context) {
    // This is a simple implementation
    // In a more advanced UI, this might be displayed in a sidebar or expandable panel
    console.log('Context from API:', context);
}

/**
 * Display evaluation information in the chat
 * 
 * @param {Object} evaluation - Evaluation object from the API
 */
function displayEvaluation(evaluation) {
    // This is a simple implementation
    // In a more advanced UI, this might be displayed as a score or detailed breakdown
    console.log('Evaluation from API:', evaluation);
}

// Initialize chat when the DOM is loaded
document.addEventListener('DOMContentLoaded', initChat);