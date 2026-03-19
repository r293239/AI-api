// training.js - AI training and pattern recognition

class AIBot {
    constructor() {
        this.trainedData = {};
        this.patterns = {};
        this.train();
    }
    
    // Train the AI with our data
    train() {
        console.log("Training AI with data...");
        
        // Flatten all training data into a single object
        for (let category in trainingData) {
            if (Array.isArray(trainingData[category])) {
                trainingData[category].forEach(item => {
                    // Store by input pattern
                    this.trainedData[item.input.toLowerCase()] = item.output;
                    
                    // Also store patterns for partial matching
                    this.storePattern(item.input.toLowerCase(), item.output);
                });
            }
        }
        
        console.log("Training complete!");
    }
    
    // Store patterns for partial matching
    storePattern(input, output) {
        // Split into words and store key phrases
        const words = input.split(' ');
        words.forEach(word => {
            if (word.length > 2) { // Only store words longer than 2 chars
                if (!this.patterns[word]) {
                    this.patterns[word] = [];
                }
                this.patterns[word].push(output);
            }
        });
    }
    
    // Find best response for user input
    getResponse(input) {
        input = input.toLowerCase().trim();
        
        // 1. Check for exact match first
        if (this.trainedData[input]) {
            return this.trainedData[input];
        }
        
        // 2. Check for math expressions
        if (this.isMathExpression(input)) {
            return this.solveMath(input);
        }
        
        // 3. Check for partial matches based on keywords
        const words = input.split(' ');
        let matches = [];
        
        words.forEach(word => {
            if (this.patterns[word]) {
                matches = matches.concat(this.patterns[word]);
            }
        });
        
        if (matches.length > 0) {
            // Return the most common match
            return this.getMostCommon(matches);
        }
        
        // 4. Check for greetings, farewells by keyword
        if (this.containsAny(input, ['hello', 'hi', 'hey'])) {
            return this.trainedData['hello'] || "Hello!";
        }
        
        if (this.containsAny(input, ['bye', 'goodbye', 'see you'])) {
            return this.trainedData['bye'] || "Goodbye!";
        }
        
        // 5. Return default if no match
        return defaultResponse;
    }
    
    // Helper: Check if input contains any keywords
    containsAny(input, keywords) {
        return keywords.some(keyword => input.includes(keyword));
    }
    
    // Helper: Get most common item in array
    getMostCommon(arr) {
        return arr.sort((a,b) =>
            arr.filter(v => v === a).length - arr.filter(v => v === b).length
        ).pop();
    }
    
    // Check if input is a math expression
    isMathExpression(input) {
        const mathPatterns = [
            /what is (\d+)\s*\+\s*(\d+)/i,
            /what is (\d+)\s*\-\s*(\d+)/i,
            /what is (\d+)\s*\*\s*(\d+)/i,
            /what is (\d+)\s*\/\s*(\d+)/i,
            /calculate (\d+)\s*\+\s*(\d+)/i,
            /add (\d+) and (\d+)/i,
            /subtract (\d+) from (\d+)/i,
            /multiply (\d+) and (\d+)/i,
            /divide (\d+) by (\d+)/i
        ];
        
        return mathPatterns.some(pattern => pattern.test(input));
    }
    
    // Solve math expressions
    solveMath(input) {
        input = input.toLowerCase();
        
        // Pattern: what is X + Y
        let match = input.match(/what is (\d+)\s*\+\s*(\d+)/i);
        if (match) {
            let result = parseInt(match[1]) + parseInt(match[2]);
            return `${match[1]} + ${match[2]} = ${result}`;
        }
        
        match = input.match(/what is (\d+)\s*\-\s*(\d+)/i);
        if (match) {
            let result = parseInt(match[1]) - parseInt(match[2]);
            return `${match[1]} - ${match[2]} = ${result}`;
        }
        
        match = input.match(/what is (\d+)\s*\*\s*(\d+)/i);
        if (match) {
            let result = parseInt(match[1]) * parseInt(match[2]);
            return `${match[1]} × ${match[2]} = ${result}`;
        }
        
        match = input.match(/what is (\d+)\s*\/\s*(\d+)/i);
        if (match) {
            let result = parseInt(match[1]) / parseInt(match[2]);
            return `${match[1]} ÷ ${match[2]} = ${result}`;
        }
        
        // Pattern: add X and Y
        match = input.match(/add (\d+) and (\d+)/i);
        if (match) {
            let result = parseInt(match[1]) + parseInt(match[2]);
            return `${match[1]} + ${match[2]} = ${result}`;
        }
        
        return "I'm not sure how to solve that math problem yet.";
    }
}

// Create and export the AI instance
const aiBot = new AIBot();
