// chat.js - Main chat logic with neural network learning

// Initialize the AI when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Show loading message
    addMessage("🧠 Initializing neural networks...", 'bot');
    setTimeout(() => {
        addMessage("Ready to learn! Let's have a conversation.", 'bot');
    }, 1000);
});

function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (message) {
        addMessage(message, 'user');
        input.value = '';
        
        showTypingIndicator();
        
        // Process with neural network
        setTimeout(() => {
            removeTypingIndicator();
            
            // Get response from neural network
            const response = conversationAI.processInput(message);
            addMessage(response, 'bot');
            
            // Train on this interaction
            conversationAI.learn(message, response);
        }, 500);
    }
}

function addMessage(text, sender) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
    const messagesDiv = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message bot';
    typingDiv.textContent = '🧠 Processing...';
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) {
        typingDiv.remove();
    }
}

document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
        e.preventDefault();
    }
});
