// enhanced_conversation_ai.js - AI with knowledge base and web search

class EnhancedConversationAI {
    constructor() {
        this.knowledgeBase = KnowledgeBase;
        this.webSearcher = new WebSearcher();
        this.vocabulary = new Set();
        this.conversationMemory = [];
        this.maxMemory = 200;
        
        // Initialize with knowledge base terms
        this.initVocabulary();
        
        // Response templates
        this.templates = this.loadTemplates();
        
        // Search mode flag
        this.webSearchEnabled = true;
    }
    
    initVocabulary() {
        // Add knowledge base terms to vocabulary
        const terms = [
            'country', 'capital', 'book', 'author', 'history', 'science',
            'technology', 'invention', 'population', 'language', 'year',
            'search', 'wikipedia', 'fact', 'random', 'tell me about'
        ];
        terms.forEach(term => this.vocabulary.add(term));
    }
    
    loadTemplates() {
        return {
            greeting: [
                "Hello! I have access to world knowledge and can search the web!",
                "Hi there! Ask me about countries, books, history, or anything!",
                "Hey! I'm enhanced with knowledge and web search capabilities!"
            ],
            farewell: [
                "Goodbye! Come learn more anytime!",
                "See you later! Keep exploring!"
            ],
            knowledge: [
                "Here's what I know:",
                "According to my knowledge base:",
                "I found this information:"
            ],
            search: [
                "Let me search the web for that...",
                "Searching online for information...",
                "I'll look that up for you:"
            ],
            no_result: [
                "I couldn't find information on that.",
                "That's not in my knowledge base yet.",
                "I'm still learning about that topic."
            ]
        };
    }
    
    async processInput(input) {
        const lowerInput = input.toLowerCase();
        
        // Check for web search command
        if (lowerInput.startsWith('/search') || lowerInput.includes('search online')) {
            const query = lowerInput.replace('/search', '').replace('search online', '').trim();
            if (query) {
                return await this.performWebSearch(query);
            }
        }
        
        // Check for random fact
        if (lowerInput.includes('random fact') || lowerInput.includes('tell me something')) {
            return this.knowledgeBase.randomFact();
        }
        
        // Check knowledge base first
        const knowledge = this.knowledgeBase.search(lowerInput);
        if (knowledge) {
            const response = `${this.getRandomResponse('knowledge')}\n${knowledge.join('\n')}`;
            this.learn(input, response);
            return response;
        }
        
        // Check for specific queries
        if (lowerInput.includes('capital of')) {
            return this.getCapitalInfo(lowerInput);
        }
        
        if (lowerInput.includes('book by') || lowerInput.includes('author')) {
            return this.getBookInfo(lowerInput);
        }
        
        if (lowerInput.includes('when did') || lowerInput.includes('history of')) {
            return this.getHistoryInfo(lowerInput);
        }
        
        // If no knowledge and web search is enabled, search the web
        if (this.webSearchEnabled && this.shouldSearchWeb(lowerInput)) {
            return await this.performWebSearch(lowerInput);
        }
        
        // Default to neural network response
        return this.getNeuralResponse(input);
    }
    
    getCapitalInfo(input) {
        const match = input.match(/capital of (\w+)/i);
        if (match) {
            const country = match[1].toLowerCase();
            const found = this.knowledgeBase.countries.find(c => 
                c.name.toLowerCase().includes(country)
            );
            if (found) {
                return `The capital of ${found.name} is ${found.capital}. Population: ${found.population}`;
            }
        }
        return null;
    }
    
    getBookInfo(input) {
        if (input.includes('book by')) {
            const author = input.replace('book by', '').trim();
            const books = this.knowledgeBase.books.filter(b => 
                b.author.toLowerCase().includes(author)
            );
            if (books.length > 0) {
                return `Books by ${author}: ${books.map(b => b.title).join(', ')}`;
            }
        }
        return null;
    }
    
    getHistoryInfo(input) {
        const match = input.match(/when did (.+?) happen/i) || 
                     input.match(/history of (.+)/i);
        if (match) {
            const event = match[1].toLowerCase();
            const found = this.knowledgeBase.history.find(h => 
                h.event.toLowerCase().includes(event)
            );
            if (found) {
                return `${found.event} happened in ${found.date}. ${found.description}`;
            }
        }
        return null;
    }
    
    shouldSearchWeb(input) {
        // Determine if we should search the web
        const searchTriggers = [
            'what is', 'who is', 'when was', 'where is', 'why is',
            'latest', 'news', 'current', 'today', 'update',
            'tell me about', 'information on', 'search for'
        ];
        
        return searchTriggers.some(trigger => input.includes(trigger));
    }
    
    async performWebSearch(query) {
        // Show typing indicator
        if (window.showTypingIndicator) showTypingIndicator();
        
        const response = this.getRandomResponse('search');
        addMessage(response, 'bot');
        
        // Small delay for natural feel
        await this.sleep(500);
        
        // Perform actual search
        const results = await this.webSearcher.search(query);
        
        if (window.removeTypingIndicator) removeTypingIndicator();
        
        return results.join('\n');
    }
    
    getNeuralResponse(input) {
        // Fallback to pattern matching or neural response
        if (input.match(/hello|hi|hey/i)) return this.getRandomResponse('greeting');
        if (input.match(/bye|goodbye/i)) return this.getRandomResponse('farewell');
        
        return this.getRandomResponse('no_result') + 
               " You can try asking about countries, books, history, or use '/search' for web search!";
    }
    
    getRandomResponse(category) {
        const templates = this.templates[category];
        return templates[Math.floor(Math.random() * templates.length)];
    }
    
    learn(input, response) {
        this.conversationMemory.push({ input, response, timestamp: Date.now() });
        if (this.conversationMemory.length > this.maxMemory) {
            this.conversationMemory.shift();
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
