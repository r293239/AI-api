// memory_systems.js - Advanced memory and learning systems adapted from Swift

// MARK: - Emotion Analysis System
class EmotionAnalyzer {
    constructor() {
        this.emotionKeywords = {
            happy: ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'love', 'perfect', 'delighted'],
            sad: ['sad', 'depressed', 'down', 'upset', 'disappointed', 'hurt', 'crying', 'terrible', 'awful', 'miserable', 'gloomy'],
            angry: ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'hate', 'disgusted', 'outraged', 'pissed'],
            anxious: ['worried', 'nervous', 'anxious', 'scared', 'afraid', 'stressed', 'panic', 'overwhelmed', 'tense', 'fear'],
            surprised: ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible', 'unbelievable', 'omg'],
            confused: ['confused', 'puzzled', 'lost', 'understand', 'unclear', 'bewildered', 'perplexed', 'unsure'],
            grateful: ['thank', 'grateful', 'appreciate', 'thankful', 'blessed', 'gratitude'],
            excited: ['excited', 'thrilled', 'pumped', 'wait', 'looking forward', 'can\'t wait', 'hyped'],
            curious: ['curious', 'wonder', 'interesting', 'fascinating', 'intriguing', 'tell me more'],
            thoughtful: ['think', 'consider', 'reflect', 'ponder', 'contemplate', 'philosophical']
        };
        
        this.emotionIntensity = {
            high: ['extremely', 'very', 'so', 'really', 'absolutely', 'completely'],
            medium: ['quite', 'somewhat', 'fairly', 'pretty'],
            low: ['slightly', 'a bit', 'a little', 'kinda']
        };
        
        this.emotionalMemory = new Map(); // Track emotional patterns over time
    }

    analyzeEmotion(text) {
        const lowercased = text.toLowerCase();
        const words = lowercased.split(/\s+/);
        
        let emotionScores = new Map();
        let intensity = this.detectIntensity(lowercased);
        
        // Analyze each word for emotional content
        words.forEach(word => {
            for (let [emotion, keywords] of Object.entries(this.emotionKeywords)) {
                if (keywords.some(keyword => word.includes(keyword))) {
                    let score = emotionScores.get(emotion) || 0;
                    emotionScores.set(emotion, score + 1);
                }
            }
        });
        
        // Check for negations (changes emotion)
        if (lowercased.includes('not ') || lowercased.includes('don\'t ') || lowercased.includes('no ')) {
            emotionScores = this.invertEmotions(emotionScores);
        }
        
        // Find dominant emotion
        let dominantEmotion = 'neutral';
        let maxScore = 0;
        
        for (let [emotion, score] of emotionScores) {
            if (score > maxScore) {
                maxScore = score;
                dominantEmotion = emotion;
            }
        }
        
        // Store in emotional memory
        this.updateEmotionalMemory(dominantEmotion, intensity);
        
        return {
            primary: dominantEmotion,
            intensity: intensity,
            scores: Object.fromEntries(emotionScores),
            timestamp: Date.now()
        };
    }

    detectIntensity(text) {
        if (this.emotionIntensity.high.some(word => text.includes(word))) return 'high';
        if (this.emotionIntensity.medium.some(word => text.includes(word))) return 'medium';
        if (this.emotionIntensity.low.some(word => text.includes(word))) return 'low';
        return 'moderate';
    }

    invertEmotions(scores) {
        const inverted = new Map();
        const oppositeMap = {
            happy: 'sad',
            sad: 'happy',
            angry: 'calm',
            anxious: 'calm',
            excited: 'calm'
        };
        
        for (let [emotion, score] of scores) {
            const opposite = oppositeMap[emotion] || emotion;
            inverted.set(opposite, (inverted.get(opposite) || 0) + score);
        }
        
        return inverted;
    }

    updateEmotionalMemory(emotion, intensity) {
        const now = Date.now();
        const recent = this.emotionalMemory.get('recent') || [];
        recent.push({ emotion, intensity, time: now });
        
        // Keep only last 50 emotional states
        if (recent.length > 50) recent.shift();
        this.emotionalMemory.set('recent', recent);
    }

    getEmotionalTrend() {
        const recent = this.emotionalMemory.get('recent') || [];
        if (recent.length < 5) return null;
        
        const emotions = recent.map(e => e.emotion);
        const mostCommon = this.mode(emotions);
        const stability = this.calculateStability(emotions);
        
        return {
            dominantMood: mostCommon,
            stability: stability,
            volatility: 1 - stability
        };
    }

    mode(array) {
        return array.sort((a,b) =>
            array.filter(v => v === a).length - array.filter(v => v === b).length
        ).pop();
    }

    calculateStability(emotions) {
        const changes = emotions.reduce((count, emotion, i) => {
            if (i > 0 && emotion !== emotions[i-1]) count++;
            return count;
        }, 0);
        
        return 1 - (changes / emotions.length);
    }
}

// MARK: - Intent Classification System
class IntentClassifier {
    constructor() {
        this.intentPatterns = {
            question: {
                patterns: ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which'],
                weight: 1.0
            },
            greeting: {
                patterns: ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                weight: 1.0
            },
            farewell: {
                patterns: ['bye', 'goodbye', 'see you', 'later', 'take care'],
                weight: 1.0
            },
            help_request: {
                patterns: ['help', 'assist', 'support', 'can you', 'could you'],
                weight: 1.2
            },
            explanation_request: {
                patterns: ['explain', 'describe', 'tell me about', 'what is', 'define', 'meaning of'],
                weight: 1.3
            },
            emotional_expression: {
                patterns: ['feel', 'feeling', 'emotion', 'mood', 'i am', 'i\'m'],
                weight: 1.4
            },
            personal_info: {
                patterns: ['my name', 'i am', 'i work', 'i study', 'i live', 'my job'],
                weight: 1.5
            },
            opinion: {
                patterns: ['think', 'believe', 'opinion', 'view on', 'thoughts about'],
                weight: 1.2
            },
            preference: {
                patterns: ['like', 'love', 'prefer', 'favorite', 'enjoy', 'dislike', 'hate'],
                weight: 1.3
            },
            knowledge_query: {
                patterns: ['know about', 'information on', 'facts about', 'tell me'],
                weight: 1.4
            },
            comparison: {
                patterns: ['compare', 'difference between', 'vs', 'versus', 'better than'],
                weight: 1.3
            },
            request: {
                patterns: ['please', 'would you', 'can you', 'will you'],
                weight: 1.2
            }
        };
        
        this.contextualIntents = new Map(); // Track intent patterns in context
    }

    classifyIntent(message, context = {}) {
        const lower = message.toLowerCase();
        let intentScores = new Map();
        
        // Score each intent based on pattern matches
        for (let [intent, data] of Object.entries(this.intentPatterns)) {
            let score = 0;
            for (let pattern of data.patterns) {
                if (lower.includes(pattern)) {
                    score += data.weight;
                }
            }
            
            // Boost score for patterns at start of message
            for (let pattern of data.patterns) {
                if (lower.startsWith(pattern)) {
                    score += data.weight * 0.5;
                }
            }
            
            if (score > 0) {
                intentScores.set(intent, score);
            }
        }
        
        // Consider conversation context
        if (context.lastIntent) {
            const contextBoost = intentScores.get(context.lastIntent) || 0;
            intentScores.set(context.lastIntent, contextBoost + 0.5);
        }
        
        // Get top intent
        let topIntent = 'general_conversation';
        let topScore = 0;
        
        for (let [intent, score] of intentScores) {
            if (score > topScore) {
                topScore = score;
                topIntent = intent;
            }
        }
        
        return {
            primary: topIntent,
            confidence: topScore / 5, // Normalize to 0-1
            alternatives: Array.from(intentScores.entries())
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([intent, score]) => ({ intent, confidence: score / 5 }))
        };
    }

    extractTopics(text) {
        const topicKeywords = {
            technology: ['computer', 'software', 'ai', 'programming', 'code', 'internet', 'tech', 'digital', 'app', 'website', 'hardware'],
            science: ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment', 'theory', 'discovery', 'scientific', 'lab'],
            arts: ['art', 'music', 'painting', 'drawing', 'creative', 'design', 'culture', 'literature', 'book', 'movie', 'film'],
            sports: ['sport', 'football', 'basketball', 'soccer', 'tennis', 'exercise', 'fitness', 'game', 'competition', 'team'],
            food: ['food', 'cooking', 'recipe', 'restaurant', 'eat', 'meal', 'dish', 'cuisine', 'taste', 'baking'],
            travel: ['travel', 'trip', 'vacation', 'country', 'city', 'place', 'visit', 'explore', 'journey', 'tourist'],
            education: ['school', 'university', 'study', 'learn', 'education', 'teacher', 'student', 'class', 'knowledge', 'course'],
            work: ['work', 'job', 'career', 'business', 'office', 'professional', 'company', 'employment', 'industry'],
            health: ['health', 'medical', 'doctor', 'medicine', 'fitness', 'wellness', 'hospital', 'treatment', 'exercise'],
            family: ['family', 'parent', 'child', 'mother', 'father', 'sibling', 'relative', 'relationship', 'home'],
            philosophy: ['philosophy', 'meaning', 'purpose', 'existence', 'consciousness', 'reality', 'truth'],
            politics: ['politics', 'government', 'policy', 'law', 'rights', 'democracy', 'election'],
            environment: ['environment', 'climate', 'nature', 'planet', 'earth', 'green', 'sustainable'],
            history: ['history', 'past', 'ancient', 'era', 'century', 'historical', 'timeline']
        };
        
        const lowercased = text.toLowerCase();
        const words = lowercased.split(/\s+/);
        const detectedTopics = new Set();
        
        // Check each word for topic matches
        words.forEach(word => {
            for (let [topic, keywords] of Object.entries(topicKeywords)) {
                if (keywords.some(keyword => word.includes(keyword) || keyword.includes(word))) {
                    detectedTopics.add(topic);
                }
            }
        });
        
        // Also check multi-word phrases
        for (let [topic, keywords] of Object.entries(topicKeywords)) {
            if (keywords.some(keyword => lowercased.includes(keyword))) {
                detectedTopics.add(topic);
            }
        }
        
        return Array.from(detectedTopics);
    }

    extractEntities(text) {
        const entities = {
            dates: this.extractDates(text),
            numbers: this.extractNumbers(text),
            names: this.extractNames(text),
            locations: this.extractLocations(text),
            urls: this.extractUrls(text),
            emails: this.extractEmails(text)
        };
        
        return entities;
    }

    extractDates(text) {
        const datePatterns = [
            /\d{1,2}\/\d{1,2}\/\d{2,4}/g, // MM/DD/YYYY
            /\d{4}-\d{1,2}-\d{1,2}/g, // YYYY-MM-DD
            /(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}/gi,
            /tomorrow|yesterday|today|next week|last week/gi
        ];
        
        let found = [];
        datePatterns.forEach(pattern => {
            const matches = text.match(pattern);
            if (matches) found.push(...matches);
        });
        
        return found;
    }

    extractNumbers(text) {
        return text.match(/\d+(\.\d+)?/g) || [];
    }

    extractNames(text) {
        // Simple name extraction (capitalized words after "my name is" or "call me")
        const patterns = [
            /my name is (\w+)/i,
            /i'm (\w+)/i,
            /call me (\w+)/i
        ];
        
        for (let pattern of patterns) {
            const match = text.match(pattern);
            if (match) return [match[1]];
        }
        
        return [];
    }

    extractLocations(text) {
        const locationPatterns = [
            /in (\w+)/gi,
            /from (\w+)/gi,
            /at (\w+)/gi
        ];
        
        let locations = [];
        locationPatterns.forEach(pattern => {
            const matches = text.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    const location = match.split(' ')[1];
                    if (location && location[0] === location[0].toUpperCase()) {
                        locations.push(location);
                    }
                });
            }
        });
        
        return locations;
    }

    extractUrls(text) {
        const urlPattern = /https?:\/\/[^\s]+/g;
        return text.match(urlPattern) || [];
    }

    extractEmails(text) {
        const emailPattern = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
        return text.match(emailPattern) || [];
    }
}

// MARK: - Context Management System
class ContextManager {
    constructor(maxContextSize = 50) {
        this.maxContextSize = maxContextSize;
        this.conversationContext = {
            messages: [],
            topics: [],
            entities: new Map(),
            flow: [],
            userState: {
                currentMood: 'neutral',
                attentionLevel: 1.0,
                engagement: 1.0,
                lastResponseTime: null
            }
        };
        
        this.topicGraph = new Map(); // Track topic relationships
    }

    addMessage(message, isUser, metadata = {}) {
        const messageObj = {
            content: message,
            isUser,
            timestamp: Date.now(),
            topics: metadata.topics || [],
            entities: metadata.entities || {},
            intent: metadata.intent || 'unknown',
            emotion: metadata.emotion || 'neutral'
        };
        
        this.conversationContext.messages.push(messageObj);
        
        // Maintain context size
        if (this.conversationContext.messages.length > this.maxContextSize) {
            this.conversationContext.messages.shift();
        }
        
        // Update topics
        this.updateTopics(messageObj);
        
        // Update user state if it's a user message
        if (isUser) {
            this.updateUserState(messageObj);
        }
        
        // Update flow
        this.conversationContext.flow.push({
            type: isUser ? 'user' : 'bot',
            intent: metadata.intent,
            time: Date.now()
        });
        
        return messageObj;
    }

    updateTopics(message) {
        message.topics.forEach(topic => {
            if (!this.conversationContext.topics.includes(topic)) {
                this.conversationContext.topics.push(topic);
                
                // Keep only most recent 20 topics
                if (this.conversationContext.topics.length > 20) {
                    this.conversationContext.topics.shift();
                }
            }
            
            // Update topic graph
            message.topics.forEach(otherTopic => {
                if (topic !== otherTopic) {
                    this.updateTopicGraph(topic, otherTopic);
                }
            });
        });
    }

    updateTopicGraph(topic1, topic2) {
        if (!this.topicGraph.has(topic1)) {
            this.topicGraph.set(topic1, new Map());
        }
        
        const connections = this.topicGraph.get(topic1);
        connections.set(topic2, (connections.get(topic2) || 0) + 1);
    }

    updateUserState(message) {
        // Update mood based on emotion
        if (message.emotion) {
            this.conversationContext.userState.currentMood = message.emotion;
        }
        
        // Update engagement based on message length and response time
        const messageLength = message.content.length;
        if (messageLength > 50) {
            this.conversationContext.userState.engagement = 
                Math.min(1.0, this.conversationContext.userState.engagement + 0.1);
        }
        
        this.conversationContext.userState.lastResponseTime = Date.now();
    }

    getContextSummary() {
        const recentMessages = this.conversationContext.messages.slice(-10);
        const userMessages = recentMessages.filter(m => m.isUser);
        const botMessages = recentMessages.filter(m => !m.isUser);
        
        return {
            recentTopics: this.getActiveTopics(5),
            userMood: this.conversationContext.userState.currentMood,
            messageCount: this.conversationContext.messages.length,
            userEngagement: this.conversationContext.userState.engagement,
            conversationFlow: this.analyzeFlow(),
            topicRelationships: this.getTopicRelationships(),
            entities: this.extractRecentEntities()
        };
    }

    getActiveTopics(limit = 5) {
        // Get most recent topics based on frequency
        const topicCount = new Map();
        
        this.conversationContext.messages.slice(-20).forEach(msg => {
            msg.topics.forEach(topic => {
                topicCount.set(topic, (topicCount.get(topic) || 0) + 1);
            });
        });
        
        return Array.from(topicCount.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([topic]) => topic);
    }

    analyzeFlow() {
        const flow = this.conversationContext.flow.slice(-10);
        const patterns = {
            questionAnswer: 0,
            emotional: 0,
            informational: 0
        };
        
        for (let i = 0; i < flow.length - 1; i++) {
            if (!flow[i].isUser && flow[i+1]?.isUser) {
                if (flow[i].intent === 'question') patterns.questionAnswer++;
                if (flow[i].intent === 'emotional_expression') patterns.emotional++;
            }
        }
        
        return patterns;
    }

    getTopicRelationships() {
        const relationships = [];
        
        for (let [topic, connections] of this.topicGraph) {
            const topConnections = Array.from(connections.entries())
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([connected]) => connected);
            
            if (topConnections.length > 0) {
                relationships.push({
                    topic,
                    related: topConnections
                });
            }
        }
        
        return relationships;
    }

    extractRecentEntities() {
        const entities = new Map();
        
        this.conversationContext.messages.slice(-20).forEach(msg => {
            Object.entries(msg.entities || {}).forEach(([type, values]) => {
                if (!entities.has(type)) entities.set(type, new Set());
                values.forEach(value => entities.get(type).add(value));
            });
        });
        
        return Object.fromEntries(
            Array.from(entities.entries()).map(([type, set]) => [type, Array.from(set)])
        );
    }

    getRelevantContext(query) {
        const queryTopics = new IntentClassifier().extractTopics(query);
        const relevantMessages = [];
        
        // Find messages with matching topics
        this.conversationContext.messages.forEach(msg => {
            if (msg.topics.some(topic => queryTopics.includes(topic))) {
                relevantMessages.push(msg);
            }
        });
        
        return relevantMessages.slice(-5); // Return most recent relevant messages
    }
}

// MARK: - User Profile System
class UserProfileSystem {
    constructor() {
        this.profile = {
            name: '',
            age: null,
            location: '',
            occupation: '',
            interests: new Map(), // interest -> confidence (0-1)
            facts: new Map(), // fact type -> value
            preferences: new Map(), // preference -> value
            relationships: new Set(),
            goals: [],
            personality: {
                traits: new Map(), // trait -> score
                communicationStyle: 'balanced',
                humorLevel: 0.5,
                formalityLevel: 0.5,
                empathy: 0.5,
                curiosity: 0.5
            },
            emotionalHistory: [],
            learningHistory: [],
            memory: {
                shortTerm: [], // Recent interactions
                longTerm: new Map(), // Important learned info
                episodic: [] // Significant events
            }
        };
        
        this.learningRate = 0.1;
        this.forgettingRate = 0.01;
    }

    learnFromInteraction(message, metadata) {
        // Update short-term memory
        this.profile.memory.shortTerm.push({
            message,
            ...metadata,
            timestamp: Date.now()
        });
        
        if (this.profile.memory.shortTerm.length > 100) {
            this.consolidateMemories();
        }
        
        // Extract and update interests
        this.updateInterests(metadata.topics || []);
        
        // Update personality based on interaction
        this.updatePersonality(metadata);
        
        // Extract personal facts
        this.extractPersonalFacts(message, metadata);
        
        // Track emotional patterns
        if (metadata.emotion) {
            this.trackEmotion(metadata.emotion, metadata.intensity);
        }
        
        // Identify significant events
        this.identifySignificantEvents(message, metadata);
    }

    updateInterests(topics) {
        topics.forEach(topic => {
            const currentScore = this.profile.interests.get(topic) || 0.5;
            const newScore = Math.min(1.0, currentScore + this.learningRate);
            this.profile.interests.set(topic, newScore);
        });
        
        // Decay unused interests
        this.profile.interests.forEach((score, topic) => {
            if (!topics.includes(topic)) {
                const newScore = Math.max(0, score - this.forgettingRate);
                if (newScore <= 0) {
                    this.profile.interests.delete(topic);
                } else {
                    this.profile.interests.set(topic, newScore);
                }
            }
        });
    }

    updatePersonality(metadata) {
        const traits = this.profile.personality.traits;
        
        // Update based on message patterns
        if (metadata.messageLength > 100) {
            traits.set('thoughtful', (traits.get('thoughtful') || 0.5) + 0.05);
        }
        
        if (metadata.intent === 'question') {
            traits.set('curious', (traits.get('curious') || 0.5) + 0.1);
        }
        
        if (metadata.emotion && metadata.emotion !== 'neutral') {
            traits.set('emotional', (traits.get('emotional') || 0.5) + 0.05);
        }
        
        // Normalize traits
        traits.forEach((value, trait) => {
            traits.set(trait, Math.min(1.0, Math.max(0, value)));
        });
        
        // Update communication style based on user's patterns
        this.updateCommunicationStyle(metadata);
    }

    updateCommunicationStyle(metadata) {
        const style = this.profile.personality;
        
        // Detect formality
        const formalWords = ['please', 'thank you', 'would you', 'could you', 'appreciate'];
        const informalWords = ['hey', 'yeah', 'gonna', 'wanna', 'cool'];
        
        const formalCount = formalWords.filter(w => metadata.message?.includes(w)).length;
        const informalCount = informalWords.filter(w => metadata.message?.includes(w)).length;
        
        if (formalCount > informalCount) {
            style.formalityLevel = Math.min(1.0, style.formalityLevel + 0.05);
        } else if (informalCount > formalCount) {
            style.formalityLevel = Math.max(0, style.formalityLevel - 0.05);
        }
        
        // Detect humor
        if (metadata.message?.includes('😂') || metadata.message?.includes('lol') || 
            metadata.message?.includes('haha')) {
            style.humorLevel = Math.min(1.0, style.humorLevel + 0.1);
        }
    }

    extractPersonalFacts(message, metadata) {
        const lower = message.toLowerCase();
        const facts = this.profile.facts;
        
        // Name extraction
        const nameMatch = message.match(/(?:my name is|i'm|call me) (\w+)/i);
        if (nameMatch && !this.profile.name) {
            this.profile.name = nameMatch[1];
            facts.set('name', nameMatch[1]);
        }
        
        // Age extraction
        const ageMatch = message.match(/(\d+)\s*(?:years old|yrs?)/i);
        if (ageMatch) {
            this.profile.age = parseInt(ageMatch[1]);
            facts.set('age', this.profile.age);
        }
        
        // Location extraction
        const locationMatch = message.match(/(?:from|in|at) (\w+(?:\s+\w+)?)/i);
        if (locationMatch && !this.profile.location) {
            this.profile.location = locationMatch[1];
            facts.set('location', this.profile.location);
        }
        
        // Occupation extraction
        const occupationMatch = message.match(/(?:work as|job is|i'm a) (\w+(?:\s+\w+)?)/i);
        if (occupationMatch) {
            this.profile.occupation = occupationMatch[1];
            facts.set('occupation', this.profile.occupation);
        }
        
        // Goals extraction
        const goalPatterns = [
            /(?:want to|hope to|planning to|goal is to) (.*?)(?:\.|$)/i,
            /(?:dream of|aspire to) (.*?)(?:\.|$)/i
        ];
        
        goalPatterns.forEach(pattern => {
            const match = message.match(pattern);
            if (match && !this.profile.goals.includes(match[1])) {
                this.profile.goals.push(match[1]);
            }
        });
    }

    trackEmotion(emotion, intensity) {
        this.profile.emotionalHistory.push({
            emotion,
            intensity,
            timestamp: Date.now()
        });
        
        // Keep only last 100 emotional states
        if (this.profile.emotionalHistory.length > 100) {
            this.profile.emotionalHistory.shift();
        }
    }

    identifySignificantEvents(message, metadata) {
        // Identify if this is a significant event (strong emotion, important info, etc.)
        const significance = this.calculateSignificance(metadata);
        
        if (significance > 0.7) {
            this.profile.memory.episodic.push({
                message,
                metadata,
                significance,
                timestamp: Date.now()
            });
        }
    }

    calculateSignificance(metadata) {
        let score = 0.5;
        
        // Strong emotions increase significance
        if (metadata.intensity === 'high') score += 0.2;
        if (metadata.intensity === 'medium') score += 0.1;
        
        // Personal information is significant
        if (metadata.intent === 'personal_info') score += 0.2;
        
        // Long messages might indicate importance
        if (metadata.messageLength > 200) score += 0.1;
        
        return Math.min(1.0, score);
    }

    consolidateMemories() {
        // Move important short-term memories to long-term
        const shortTerm = this.profile.memory.shortTerm;
        
        shortTerm.forEach(memory => {
            const importance = this.calculateImportance(memory);
            if (importance > 0.6) {
                const key = `${memory.topic}_${Date.now()}`;
                this.profile.memory.longTerm.set(key, {
                    ...memory,
                    importance,
                    consolidatedAt: Date.now()
                });
            }
        });
        
        // Clear short-term memory but keep recent
        this.profile.memory.shortTerm = this.profile.memory.shortTerm.slice(-20);
    }

    calculateImportance(memory) {
        let importance = 0.5;
        
        if (memory.topics?.length > 0) importance += 0.1;
        if (memory.emotion && memory.emotion !== 'neutral') importance += 0.1;
        if (memory.intent === 'personal_info') importance += 0.2;
        
        return Math.min(1.0, importance);
    }

    getPersonalizedContext() {
        return {
            name: this.profile.name || 'friend',
            interests: Array.from(this.profile.interests.entries())
                .filter(([_, score]) => score > 0.5)
                .map(([interest]) => interest),
            mood: this.getCurrentMood(),
            personality: {
                humor: this.profile.personality.humorLevel,
                formality: this.profile.personality.formalityLevel
            },
            facts: Object.fromEntries(this.profile.facts),
            recentTopics: this.getRecentTopics(5),
            goals: this.profile.goals.slice(0, 3)
        };
    }

    getCurrentMood() {
        if (this.profile.emotionalHistory.length === 0) return 'neutral';
        
        const recent = this.profile.emotionalHistory.slice(-10);
        const moods = recent.map(e => e.emotion);
        
        // Find most common recent mood
        return this.mode(moods);
    }

    getRecentTopics(limit) {
        const topics = new Map();
        
        this.profile.memory.shortTerm.forEach(memory => {
            memory.topics?.forEach(topic => {
                topics.set(topic, (topics.get(topic) || 0) + 1);
            });
        });
        
        return Array.from(topics.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([topic]) => topic);
    }

    mode(array) {
        return array.sort((a,b) =>
            array.filter(v => v === a).length - array.filter(v => v === b).length
        ).pop();
    }

    toJSON() {
        return {
            name: this.profile.name,
            age: this.profile.age,
            location: this.profile.location,
            occupation: this.profile.occupation,
            interests: Object.fromEntries(this.profile.interests),
            facts: Object.fromEntries(this.profile.facts),
            preferences: Object.fromEntries(this.profile.preferences),
            relationships: Array.from(this.profile.relationships),
            goals: this.profile.goals,
            personality: {
                ...this.profile.personality,
                traits: Object.fromEntries(this.profile.personality.traits)
            },
            memory: {
                longTerm: Object.fromEntries(this.profile.memory.longTerm),
                episodic: this.profile.memory.episodic
            }
        };
    }

    fromJSON(data) {
        this.profile.name = data.name || '';
        this.profile.age = data.age || null;
        this.profile.location = data.location || '';
        this.profile.occupation = data.occupation || '';
        this.profile.interests = new Map(Object.entries(data.interests || {}));
        this.profile.facts = new Map(Object.entries(data.facts || {}));
        this.profile.preferences = new Map(Object.entries(data.preferences || {}));
        this.profile.relationships = new Set(data.relationships || []);
        this.profile.goals = data.goals || [];
        this.profile.personality = data.personality || this.profile.personality;
        this.profile.personality.traits = new Map(Object.entries(data.personality?.traits || {}));
        this.profile.memory.longTerm = new Map(Object.entries(data.memory?.longTerm || {}));
        this.profile.memory.episodic = data.memory?.episodic || [];
    }
}

// MARK: - Response Generation System
class ResponseGenerator {
    constructor() {
        this.templates = {
            greeting: [
                "Hello {name}! How are you feeling today?",
                "Hi {name}! It's great to chat with you again.",
                "Hey {name}! I've been thinking about our last conversation.",
                "Hello there! How's your day going so far?"
            ],
            farewell: [
                "Goodbye {name}! Can't wait to chat again!",
                "See you later! Take care of yourself.",
                "Bye for now! I'll remember what we talked about.",
                "Until next time! Stay awesome!"
            ],
            emotional: {
                happy: [
                    "I'm so glad to hear you're feeling happy, {name}! What's bringing you joy?",
                    "Your happiness is contagious! Tell me more about what's making you smile.",
                    "That's wonderful to hear! I love when you share your positive moments."
                ],
                sad: [
                    "I'm sorry you're feeling down, {name}. I'm here to listen if you want to talk.",
                    "It's okay to feel sad sometimes. Would sharing help?",
                    "I hear the sadness in your words. Remember that difficult feelings pass."
                ],
                anxious: [
                    "I sense you're feeling anxious. Let's take a deep breath together.",
                    "It sounds like you have a lot on your mind. Want to talk through it?",
                    "Anxiety can be overwhelming. What's the main thing worrying you?"
                ],
                angry: [
                    "I can hear the frustration in your voice. What happened?",
                    "It's okay to feel angry sometimes. Want to talk about what's bothering you?",
                    "I'm here to listen without judgment. What's making you feel this way?"
                ],
                curious: [
                    "I love your curiosity, {name}! What would you like to explore?",
                    "That's a fascinating question! Let's think about it together.",
                    "Your curiosity is one of the things I appreciate about our conversations."
                ]
            },
            question: {
                personal: [
                    "From what I remember, you've mentioned {fact}. Is that still accurate?",
                    "Based on our conversations, I think {answer}. But I'd love to hear your thoughts.",
                    "That's an interesting question! Given your interest in {interest}, what do you think?"
                ],
                factual: [
                    "Let me think about that... {answer}",
                    "Based on what I know, {answer}. Does that help?",
                    "That's a great question! Here's what I understand: {answer}"
                ],
                reflective: [
                    "That's a deep question. It makes me think about {reflection}. What's your perspective?",
                    "I don't have a simple answer, but it reminds me of {connection}. Does that resonate?",
                    "Questions like these are why I enjoy talking with you. Here's my take: {thought}"
                ]
            },
            personal_info: [
                "Thank you for sharing that with me, {name}! I'll remember that.",
                "That's really interesting! It helps me understand you better.",
                "I appreciate you telling me that. Your experiences are valuable.",
                "Learning about you is what makes our conversations special."
            ],
            general: [
                "That's really interesting! Tell me more about {topic}.",
                "I appreciate you sharing that perspective. It gives me new insight.",
                "You always have such unique thoughts. What else is on your mind?",
                "I'm fascinated by our conversations. You make me think in new ways."
            ],
            help: [
                "I'd be happy to help! What specific aspect would you like assistance with?",
                "Of course! I can help with questions, brainstorming, or just listening.",
                "I'm here to help however I can. What do you need?"
            ],
            knowledge: [
                "Based on what I know, {answer}",
                "Here's some information: {answer}",
                "I can share that {answer}"
            ]
        };
        
        this.personalizedResponses = new Map();
    }

    generateResponse(message, intent, context, userProfile) {
        const name = userProfile.name || 'friend';
        const mood = context.userMood || 'neutral';
        const topics = context.recentTopics || [];
        const interests = userProfile.interests || [];
        
        // Personalize based on user profile
        let response = '';
        
        switch (intent.primary) {
            case 'greeting':
                response = this.personalizeTemplate(
                    this.templates.greeting,
                    { name }
                );
                break;
                
            case 'farewell':
                response = this.personalizeTemplate(
                    this.templates.farewell,
                    { name }
                );
                break;
                
            case 'emotional_expression':
                response = this.generateEmotionalResponse(
                    message,
                    mood,
                    name,
                    userProfile
                );
                break;
                
            case 'question':
                response = this.generateQuestionResponse(
                    message,
                    context,
                    userProfile
                );
                break;
                
            case 'personal_info':
                response = this.personalizeTemplate(
                    this.templates.personal_info,
                    { name }
                );
                break;
                
            case 'help_request':
                response = this.personalizeTemplate(
                    this.templates.help,
                    {}
                );
                break;
                
            case 'knowledge_query':
                response = this.generateKnowledgeResponse(
                    message,
                    context
                );
                break;
                
            default:
                response = this.generateGeneralResponse(
                    message,
                    context,
                    userProfile
                );
        }
        
        // Add personal touches
        response = this.addPersonalTouches(response, userProfile, context);
        
        // Store in personalized responses
        this.storeResponse(message, response, userProfile);
        
        return response;
    }

    generateEmotionalResponse(message, mood, name, userProfile) {
        const emotionalResponses = this.templates.emotional[mood] || 
                                   this.templates.emotional.neutral ||
                                   ["I hear you. Tell me more about how you're feeling."];
        
        return this.personalizeTemplate(emotionalResponses, { name });
    }

    generateQuestionResponse(message, context, userProfile) {
        // Determine question type
        const lower = message.toLowerCase();
        let type = 'factual';
        
        if (lower.includes('you think') || lower.includes('your opinion')) {
            type = 'reflective';
        } else if (lower.includes('remember') || lower.includes('know about me')) {
            type = 'personal';
        }
        
        // Get relevant context
        const relevantMemory = this.findRelevantMemory(message, userProfile);
        const interest = userProfile.interests[0] || 'this topic';
        
        let answer = '';
        if (type === 'personal' && relevantMemory) {
            answer = relevantMemory;
        } else {
            answer = this.generateAnswer(message, context);
        }
        
        const templates = this.templates.question[type];
        return this.personalizeTemplate(templates, {
            fact: relevantMemory || 'something about you',
            answer: answer,
            interest: interest,
            reflection: this.generateReflection(message),
            connection: this.findConnection(message, userProfile),
            thought: this.generateThought(message)
        });
    }

    generateGeneralResponse(message, context, userProfile) {
        const topic = context.recentTopics[0] || 'that';
        const templates = this.templates.general;
        
        return this.personalizeTemplate(templates, { topic });
    }

    generateKnowledgeResponse(message, context) {
        const answer = this.generateAnswer(message, context);
        const templates = this.templates.knowledge;
        
        return this.personalizeTemplate(templates, { answer });
    }

    generateAnswer(message, context) {
        // Simple answer generation
        if (message.toLowerCase().includes('time')) {
            return `the current time is ${new Date().toLocaleTimeString()}`;
        }
        
        if (message.toLowerCase().includes('date')) {
            return `today's date is ${new Date().toLocaleDateString()}`;
        }
        
        // Try to find relevant info from knowledge base
        const knowledge = window.knowledgeBase?.search(message);
        if (knowledge && knowledge.length > 0) {
            return knowledge[0];
        }
        
        return "I'm still learning about that. Could you tell me more?";
    }

    generateReflection(message) {
        const reflections = [
            "the complexity of human experience",
            "how our perspectives shape our reality",
            "the connections between different ideas",
            "how much we learn from each conversation"
        ];
        
        return reflections[Math.floor(Math.random() * reflections.length)];
    }

    generateThought(message) {
        const thoughts = [
            "we're all constantly learning and growing",
            "every question opens up new possibilities",
            "curiosity is what drives understanding",
            "conversations like these make us wiser"
        ];
        
        return thoughts[Math.floor(Math.random() * thoughts.length)];
    }

    findRelevantMemory(message, userProfile) {
        // Look for relevant memories in user profile
        const memories = Array.from(userProfile.memory?.longTerm?.values() || []);
        const words = message.toLowerCase().split(/\s+/);
        
        for (let memory of memories) {
            for (let word of words) {
                if (memory.message?.toLowerCase().includes(word)) {
                    return memory.message;
                }
            }
        }
        
        return null;
    }

    findConnection(message, userProfile) {
        const interests = Array.from(userProfile.interests?.keys() || []);
        if (interests.length > 0) {
            return `your interest in ${interests[0]}`;
        }
        return "things we've discussed before";
    }

    personalizeTemplate(templates, variables) {
        let template = templates[Math.floor(Math.random() * templates.length)];
        
        // Replace variables in template
        for (let [key, value] of Object.entries(variables)) {
            template = template.replace(`{${key}}`, value);
        }
        
        return template;
    }

    addPersonalTouches(response, userProfile, context) {
        // Add name if missing but we know it
        if (!response.includes(userProfile.name) && userProfile.name) {
            if (Math.random() > 0.5) {
                response = response.replace(/^(Hello|Hi|Hey)/, `$1 ${userProfile.name}`);
            }
        }
        
        // Reference past conversations
        if (context.recentTopics?.length > 0 && Math.random() > 0.7) {
            const topic = context.recentTopics[0];
            response += ` Speaking of ${topic}, I remember our last conversation about it.`;
        }
        
        // Add personality-based flourishes
        if (userProfile.personality?.humorLevel > 0.7 && Math.random() > 0.5) {
            response += " 😊";
        }
        
        return response;
    }

    storeResponse(message, response, userProfile) {
        const key = message.toLowerCase().slice(0, 50);
        this.personalizedResponses.set(key, {
            response,
            timestamp: Date.now(),
            user: userProfile.name || 'anonymous'
        });
        
        // Keep only recent responses
        if (this.personalizedResponses.size > 1000) {
            const oldestKey = this.personalizedResponses.keys().next().value;
            this.personalizedResponses.delete(oldestKey);
        }
    }
}

// MARK: - Main AI Brain
class SmartAIBrain {
    constructor() {
        this.emotionAnalyzer = new EmotionAnalyzer();
        this.intentClassifier = new IntentClassifier();
        this.contextManager = new ContextManager();
        this.userProfile = new UserProfileSystem();
        this.responseGenerator = new ResponseGenerator();
        
        this.conversationHistory = [];
        this.learningEnabled = true;
        this.personality = {
            name: 'Claude',
            traits: {
                empathy: 0.8,
                curiosity: 0.9,
                humor: 0.6,
                formality: 0.5
            }
        };
    }

    processMessage(message, isUser = true) {
        if (!message || message.trim() === '') return null;
        
        // Analyze message
        const emotion = this.emotionAnalyzer.analyzeEmotion(message);
        const intent = this.intentClassifier.classifyIntent(message, {
            lastIntent: this.contextManager.conversationContext.flow.slice(-1)[0]?.intent
        });
        const topics = this.intentClassifier.extractTopics(message);
        const entities = this.intentClassifier.extractEntities(message);
        
        // Store in context
        const messageObj = this.contextManager.addMessage(message, isUser, {
            emotion: emotion.primary,
            intent: intent.primary,
            topics,
            entities,
            messageLength: message.length
        });
        
        // Update user profile if it's a user message
        if (isUser && this.learningEnabled) {
            this.userProfile.learnFromInteraction(message, {
                emotion: emotion.primary,
                intensity: emotion.intensity,
                intent: intent.primary,
                topics,
                entities,
                messageLength: message.length
            });
        }
        
        // Store in conversation history
        this.conversationHistory.push(messageObj);
        
        return {
            emotion,
            intent,
            topics,
            entities,
            messageObj
        };
    }

    generateResponse(message, analysis) {
        const context = this.contextManager.getContextSummary();
        const userContext = this.userProfile.getPersonalizedContext();
        
        // Combine contexts
        const fullContext = {
            ...context,
            user: userContext,
            personality: this.personality
        };
        
        // Generate response
        const response = this.responseGenerator.generateResponse(
            message,
            analysis.intent,
            fullContext,
            this.userProfile
        );
        
        // Store bot message in context
        this.contextManager.addMessage(response, false, {
            intent: 'response',
            topics: analysis.topics
        });
        
        return response;
    }

    async chat(message) {
        // Process user message
        const analysis = this.processMessage(message);
        
        // Generate response
        const response = this.generateResponse(message, analysis);
        
        // Learn from this interaction
        if (this.learningEnabled) {
            this.learnFromInteraction(message, response, analysis);
        }
        
        return {
            response,
            analysis,
            context: this.contextManager.getContextSummary(),
            userProfile: this.userProfile.getPersonalizedContext()
        };
    }

    learnFromInteraction(message, response, analysis) {
        // Store in long-term memory if significant
        if (analysis.emotion.intensity === 'high' || 
            analysis.intent.confidence > 0.8) {
            this.userProfile.identifySignificantEvents(message, {
                ...analysis,
                response
            });
        }
    }

    getStats() {
        return {
            conversationLength: this.conversationHistory.length,
            topicsLearned: this.userProfile.profile.interests.size,
            factsLearned: this.userProfile.profile.facts.size,
            emotionalStates: this.userProfile.profile.emotionalHistory.length,
            memories: this.userProfile.profile.memory.longTerm.size,
            userProfile: this.userProfile.toJSON()
        };
    }

    saveState() {
        const state = {
            userProfile: this.userProfile.toJSON(),
            conversationHistory: this.conversationHistory.slice(-100), // Last 100 messages
            context: this.contextManager.getContextSummary(),
            timestamp: Date.now()
        };
        
        localStorage.setItem('smartAIState', JSON.stringify(state));
        return state;
    }

    loadState() {
        const saved = localStorage.getItem('smartAIState');
        if (saved) {
            try {
                const state = JSON.parse(saved);
                this.userProfile.fromJSON(state.userProfile);
                this.conversationHistory = state.conversationHistory || [];
                return true;
            } catch (e) {
                console.error('Failed to load state:', e);
                return false;
            }
        }
        return false;
    }

    clearState() {
        localStorage.removeItem('smartAIState');
        this.userProfile = new UserProfileSystem();
        this.conversationHistory = [];
        this.contextManager = new ContextManager();
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SmartAIBrain,
        EmotionAnalyzer,
        IntentClassifier,
        ContextManager,
        UserProfileSystem,
        ResponseGenerator
    };
}
