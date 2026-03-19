// data.js - Training data for the AI

const trainingData = {
    // Greetings
    greetings: [
        { input: "hello", output: "Hi there! How can I help you today?" },
        { input: "hi", output: "Hello! Nice to meet you!" },
        { input: "hey", output: "Hey! What's up?" },
        { input: "good morning", output: "Good morning! How are you today?" },
        { input: "good afternoon", output: "Good afternoon! How can I assist?" },
        { input: "good evening", output: "Good evening! Hope you're having a nice day!" }
    ],
    
    // Basic math
    math: [
        { input: "what is 2+2", output: "2 + 2 = 4" },
        { input: "what is 5+3", output: "5 + 3 = 8" },
        { input: "what is 10-4", output: "10 - 4 = 6" },
        { input: "what is 3*3", output: "3 × 3 = 9" },
        { input: "what is 12/4", output: "12 ÷ 4 = 3" },
        { input: "calculate 7+8", output: "7 + 8 = 15" }
    ],
    
    // Simple patterns
    patterns: [
        { input: "how are you", output: "I'm doing great, thanks for asking!" },
        { input: "what's your name", output: "I'm your AI assistant, still learning every day!" },
        { input: "who made you", output: "I was created from scratch to learn and help!" },
        { input: "what can you do", output: "I can learn patterns, do basic math, and have conversations!" },
        { input: "tell me a joke", output: "Why don't scientists trust atoms? Because they make up everything!" }
    ],
    
    // Farewells
    farewells: [
        { input: "bye", output: "Goodbye! Come back anytime!" },
        { input: "goodbye", output: "See you later! Take care!" },
        { input: "see you", output: "See you soon! Have a great day!" }
    ],
    
    // Help responses
    help: [
        { input: "help", output: "I can help with: greetings, basic math, jokes, and simple conversations!" }
    ]
};

// Default response when no pattern matches
const defaultResponse = "I'm still learning! Can you rephrase that or ask me something else?";
