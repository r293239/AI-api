// conversation_ai.js - Main conversation AI with neural learning

class ConversationAI {
    constructor() {
        this.vocabulary = new Set();
        this.wordToIndex = {};
        this.indexToWord = {};
        this.conversationMemory = [];
        this.contextWindow = [];
        this.maxMemory = 100;
        
        // Initialize with basic vocabulary
        this.initVocabulary();
        
        // Create neural network
        // Input: bag of words (vocabulary size)
        // Hidden layers: 128, 64 neurons
        // Output: response probability distribution
        this.vocabSize = this.vocabulary.size;
        this.network = new NeuralNetwork(
            this.vocabSize,  // input size
            [128, 64],       // hidden layers
            this.vocabSize,  // output size
            0.001            // learning rate
        );
        
        // Response templates
        this.responseTemplates = this.loadTemplates();
    }
    
    initVocabulary() {
        // Add common words to vocabulary
        const commonWords = [
            'hello', 'hi', 'hey', 'how', 'are', 'you', 'what', 'is', 'your', 'name',
            'i', 'am', 'good', 'bad', 'great', 'awesome', 'thanks', 'thank', 'welcome',
            'bye', 'goodbye', 'see', 'later', 'help', 'can', 'do', 'tell', 'joke',
            'math', 'calculate', 'plus', 'minus', 'times', 'divide', 'equals',
            'weather', 'time', 'today', 'tomorrow', 'yes', 'no', 'maybe', 'please',
            'sorry', 'ok', 'okay', 'sure', 'cool', 'nice', 'love', 'hate', 'like'
        ];
        
        commonWords.forEach(word => this.vocabulary.add(word));
        this.updateIndices();
    }
    
    updateIndices() {
        this.wordToIndex = {};
        this.indexToWord = {};
        let i = 0;
        this.vocabulary.forEach(word => {
            this.wordToIndex[word] = i;
            this.indexToWord[i] = word;
            i++;
        });
    }
    
    loadTemplates() {
        return {
            greeting: [
                "Hello! How can I help you today?",
                "Hi there! Nice to talk to you!",
                "Hey! What's on your mind?"
            ],
            farewell: [
                "Goodbye! Come back anytime!",
                "See you later! Take care!",
                "Bye! It was nice chatting!"
            ],
            thanks: [
                "You're welcome!",
                "Happy to help!",
                "Anytime!"
            ],
            unknown: [
                "That's interesting! Tell me more.",
                "I'm still learning about that.",
                "Could you explain that differently?",
                "I'm not sure I understand."
            ],
            joke: [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fake noodle? An impasta!",
                "Why did the math book look so sad? Because it had too many problems!"
            ]
        };
    }
    
    textToVector(text) {
        const words = text.toLowerCase().split(/\W+/);
        const vector = new Array(this.vocabSize).fill(0);
        
        words.forEach(word => {
            if (this.vocabulary.has(word)) {
                vector[this.wordToIndex[word]] += 1;
            } else {
                // Add new word to vocabulary
                this.vocabulary.add(word);
                this.updateIndices();
                // Resize vector (simplified - in practice you'd reinitialize)
            }
        });
        
        return vector;
    }
    
    vectorToText(vector) {
        // Find highest probability words
        const indices = [];
        for (let i = 0; i < vector.length; i++) {
            indices.push({ index: i, value: vector[i] });
        }
        
        indices.sort((a, b) => b.value - a.value);
        
        // Take top 3 words and try to form response
        const topWords = indices.slice(0, 3).map(item => this.indexToWord[item.index]);
        
        return this.generateResponse(topWords);
    }
    
    generateResponse(keywords) {
        // Check for specific patterns
        if (keywords.includes('hello') || keywords.includes('hi') || keywords.includes('hey')) {
            return this.getRandomResponse('greeting');
        }
        
        if (keywords.includes('bye') || keywords.includes('goodbye')) {
            return this.getRandomResponse('farewell');
        }
        
        if (keywords.includes('thanks') || keywords.includes('thank')) {
            return this.getRandomResponse('thanks');
        }
        
        if (keywords.includes('joke')) {
            return this.getRandomResponse('joke');
        }
        
        if (keywords.includes('math') || keywords.includes('calculate')) {
            return "I can do basic math! Try asking 'what is 2+2'?";
        }
        
        // Check context for better responses
        if (this.contextWindow.length > 0) {
            const lastTopic = this.contextWindow[this.contextWindow.length - 1];
            if (lastTopic === 'greeting') {
                return "How are you doing today?";
            }
        }
        
        // Default response
        return this.getRandomResponse('unknown');
    }
    
    getRandomResponse(category) {
        const templates = this.responseTemplates[category];
        return templates[Math.floor(Math.random() * templates.length)];
    }
    
    processInput(input) {
        // Convert input to vector
        const inputVector = this.textToVector(input);
        
        // Get network prediction
        const outputVector = this.network.predict(inputVector);
        
        // Convert prediction to response
        let response = this.vectorToText(outputVector);
        
        // Check for math expressions
        const mathResult = this.solveMath(input);
        if (mathResult) {
            response = mathResult;
        }
        
        // Update context
        this.updateContext(input, response);
        
        return response;
    }
    
    solveMath(input) {
        // Simple math solver
        const mathPatterns = [
            { pattern: /(\d+)\s*\+\s*(\d+)/, operation: (a, b) => parseInt(a) + parseInt(b) },
            { pattern: /(\d+)\s*\-\s*(\d+)/, operation: (a, b) => parseInt(a) - parseInt(b) },
            { pattern: /(\d+)\s*\*\s*(\d+)/, operation: (a, b) => parseInt(a) * parseInt(b) },
            { pattern: /(\d+)\s*\/\s*(\d+)/, operation: (a, b) => parseInt(a) / parseInt(b) }
        ];
        
        for (let { pattern, operation } of mathPatterns) {
            const match = input.match(pattern);
            if (match) {
                const result = operation(match[1], match[2]);
                return `${match[1]} ${input.includes('+') ? '+' : 
                                      input.includes('-') ? '-' :
                                      input.includes('*') ? '×' : '÷'} ${match[2]} = ${result}`;
            }
        }
        
        return null;
    }
    
    updateContext(input, response) {
        // Determine topic
        let topic = 'general';
        if (input.match(/hello|hi|hey/i)) topic = 'greeting';
        else if (input.match(/bye|goodbye/i)) topic = 'farewell';
        else if (input.match(/joke/i)) topic = 'humor';
        else if (input.match(/math|\d+/)) topic = 'math';
        
        this.contextWindow.push(topic);
        if (this.contextWindow.length > 5) {
            this.contextWindow.shift();
        }
    }
    
    learn(input, response) {
        // Store in memory
        this.conversationMemory.push({ input, response, timestamp: Date.now() });
        if (this.conversationMemory.length > this.maxMemory) {
            this.conversationMemory.shift();
        }
        
        // Train on this interaction (simplified - would need proper training pairs)
        const inputVector = this.textToVector(input);
        const targetVector = this.textToVector(response);
        
        // Do a training step
        this.network.backward(inputVector, targetVector);
        
        console.log("Learned from interaction:", { input, response });
    }
    
    getStats() {
        return {
            vocabularySize: this.vocabulary.size,
            memorySize: this.conversationMemory.length,
            contextLength: this.contextWindow.length,
            networkDepth: this.network.layers.length
        };
    }
}

// Initialize the AI
const conversationAI = new ConversationAI();

// Add stats command to chat (hidden feature)
window.showStats = function() {
    const stats = conversationAI.getStats();
    addMessage(`📊 AI Stats:
• Vocabulary: ${stats.vocabularySize} words
• Memory: ${stats.memorySize} conversations
• Context window: ${stats.contextLength}
• Neural network depth: ${stats.networkDepth}`, 'bot');
};
