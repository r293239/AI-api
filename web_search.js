// web_search.js - Web search without API keys
// Using DuckDuckGo HTML API (no key needed) and fallback methods

class WebSearcher {
    constructor() {
        this.searchEngines = [
            this.searchDuckDuckGo,
            this.searchGoogleFallback
        ];
        this.cache = new Map();
        this.rateLimit = 1000; // ms between searches
        this.lastSearch = 0;
    }
    
    async search(query, maxResults = 3) {
        // Check cache first
        if (this.cache.has(query)) {
            const cached = this.cache.get(query);
            if (Date.now() - cached.timestamp < 3600000) { // 1 hour cache
                return cached.results;
            }
        }
        
        // Rate limiting
        const now = Date.now();
        if (now - this.lastSearch < this.rateLimit) {
            await this.sleep(this.rateLimit - (now - this.lastSearch));
        }
        
        // Try different search methods
        for (const engine of this.searchEngines) {
            try {
                const results = await engine.call(this, query, maxResults);
                if (results && results.length > 0) {
                    this.cache.set(query, { results, timestamp: Date.now() });
                    this.lastSearch = Date.now();
                    return results;
                }
            } catch (e) {
                console.log(`Search engine failed:`, e);
                continue;
            }
        }
        
        return ["Web search unavailable. Try again later."];
    }
    
    async searchDuckDuckGo(query, maxResults) {
        try {
            // Use DuckDuckGo HTML API (no key needed)
            const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
            
            // Using a CORS proxy for browser environment
            const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
            const response = await fetch(proxyUrl);
            const data = await response.json();
            
            // Parse HTML results (simplified)
            const parser = new DOMParser();
            const doc = parser.parseFromString(data.contents, 'text/html');
            
            const results = [];
            const links = doc.querySelectorAll('.result__a');
            const snippets = doc.querySelectorAll('.result__snippet');
            
            for (let i = 0; i < Math.min(links.length, maxResults); i++) {
                if (links[i] && snippets[i]) {
                    results.push({
                        title: links[i].textContent.trim(),
                        link: links[i].href,
                        snippet: snippets[i].textContent.trim()
                    });
                }
            }
            
            return this.formatResults(results, 'DuckDuckGo');
        } catch (e) {
            console.error('DuckDuckGo search failed:', e);
            return null;
        }
    }
    
    async searchGoogleFallback(query, maxResults) {
        try {
            // Alternative: use a public search API
            const response = await fetch(
                `https://en.wikipedia.org/w/api.php?` +
                `action=query&list=search&srsearch=${encodeURIComponent(query)}` +
                `&format=json&origin=*`
            );
            
            const data = await response.json();
            
            if (data.query && data.query.search) {
                const results = data.query.search.slice(0, maxResults).map(item => ({
                    title: item.title,
                    snippet: item.snippet.replace(/<\/?[^>]+(>|$)/g, ""), // Strip HTML
                    link: `https://en.wikipedia.org/wiki/${encodeURIComponent(item.title)}`
                }));
                
                return this.formatResults(results, 'Wikipedia');
            }
            return null;
        } catch (e) {
            console.error('Wikipedia search failed:', e);
            return null;
        }
    }
    
    formatResults(results, source) {
        if (results.length === 0) return ["No results found"];
        
        const formatted = [`🔍 Search results from ${source}:\n`];
        results.forEach((r, i) => {
            formatted.push(`${i+1}. ${r.title}`);
            formatted.push(`   ${r.snippet.substring(0, 100)}...`);
            formatted.push(`   🔗 ${r.link}\n`);
        });
        
        return formatted;
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
