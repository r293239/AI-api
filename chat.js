// chat.js - Fixed integration with all systems

// Initialize the AI
let smartAI;

document.addEventListener('DOMContentLoaded', () => {
    // Create the smart AI instance
    smartAI = new SmartAIBrain();
    
    // Try to load previous state
    smartAI.loadState();
    
    // Welcome message
    addMessage("🧠 Hello! I'm your smart AI with memory systems. I can remember our conversations and learn about you over time. What would you like to talk about?", 'bot');
    
    console.log("AI Systems initialized:", smartAI.getStats());
});

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Process with AI (small delay for natural feel)
        setTimeout(async () => {
            // Get AI response
            const result = await smartAI.chat(message);
            
            // Remove typing indicator and add response
            removeTypingIndicator();
            addMessage(result.response, 'bot');
            
            // Auto-save state periodically
            if (Math.random() > 0.7) {
                smartAI.saveState();
                console.log("AI state saved");
            }
        }, 500);
        
    } catch (error) {
        console.error("AI Error:", error);
        removeTypingIndicator();
        addMessage("I encountered an error processing that. Let's try again!", 'bot');
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
    if (document.getElementById('typingIndicator')) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message bot';
    typingDiv.textContent = '🧠 Thinking...';
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) {
        typingDiv.remove();
    }
}

// Handle Enter key
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
        e.preventDefault();
    }
});

// Add a stats command (hidden feature)
window.showStats = function() {
    if (smartAI) {
        const stats = smartAI.getStats();
        const statsMessage = `
📊 **AI Stats:**
• Conversations: ${stats.conversationLength}
• Topics learned: ${stats.topicsLearned}
• Facts about you: ${stats.factsLearned}
• Emotional states tracked: ${stats.emotionalStates}
• Long-term memories: ${stats.memories}
        `;
        addMessage(statsMessage, 'bot');
    }
};
