// knowledge_base.js - Encyclopedia-scale knowledge database

const KnowledgeBase = {
    // Countries data
    countries: [
        { name: "United States", capital: "Washington, D.C.", population: "331 million", language: "English", currency: "US Dollar", continent: "North America" },
        { name: "Japan", capital: "Tokyo", population: "125 million", language: "Japanese", currency: "Yen", continent: "Asia" },
        { name: "France", capital: "Paris", population: "67 million", language: "French", currency: "Euro", continent: "Europe" },
        { name: "Brazil", capital: "Brasília", population: "213 million", language: "Portuguese", currency: "Real", continent: "South America" },
        { name: "India", capital: "New Delhi", population: "1.38 billion", language: "Hindi, English", currency: "Rupee", continent: "Asia" },
        { name: "Egypt", capital: "Cairo", population: "104 million", language: "Arabic", currency: "Pound", continent: "Africa" },
        { name: "Australia", capital: "Canberra", population: "25.7 million", language: "English", currency: "Australian Dollar", continent: "Oceania" },
        { name: "Canada", capital: "Ottawa", population: "38 million", language: "English, French", currency: "Canadian Dollar", continent: "North America" },
        { name: "Germany", capital: "Berlin", population: "83 million", language: "German", currency: "Euro", continent: "Europe" },
        { name: "Mexico", capital: "Mexico City", population: "126 million", language: "Spanish", currency: "Peso", continent: "North America" }
    ],
    
    // Books and literature
    books: [
        { title: "1984", author: "George Orwell", year: 1949, genre: "Dystopian", description: "Totalitarian surveillance state" },
        { title: "To Kill a Mockingbird", author: "Harper Lee", year: 1960, genre: "Fiction", description: "Racial injustice in the American South" },
        { title: "The Great Gatsby", author: "F. Scott Fitzgerald", year: 1925, genre: "Tragedy", description: "Jazz Age American story" },
        { title: "Pride and Prejudice", author: "Jane Austen", year: 1813, genre: "Romance", description: "Love and social class in England" },
        { title: "The Hobbit", author: "J.R.R. Tolkien", year: 1937, genre: "Fantasy", description: "Bilbo Baggins' adventure" },
        { title: "Dune", author: "Frank Herbert", year: 1965, genre: "Science Fiction", description: "Desert planet politics" },
        { title: "The Catcher in the Rye", author: "J.D. Salinger", year: 1951, genre: "Coming-of-age", description: "Teenage alienation" }
    ],
    
    // Historical events
    history: [
        { event: "World War II", date: "1939-1945", description: "Global conflict" },
        { event: "Moon Landing", date: "1969", description: "First humans on moon" },
        { event: "French Revolution", date: "1789-1799", description: "Overthrow of monarchy" },
        { event: "Industrial Revolution", date: "1760-1840", description: "Machines and factories" },
        { event: "Fall of Berlin Wall", date: "1989", description: "End of Cold War" }
    ],
    
    // Science facts
    science: [
        { topic: "Gravity", fact: "Newton discovered gravity when an apple fell" },
        { topic: "Evolution", fact: "Darwin's theory of natural selection" },
        { topic: "DNA", fact: "Discovered by Watson and Crick in 1953" },
        { topic: "Black Holes", fact: "Regions where gravity prevents escape" },
        { topic: "Photosynthesis", fact: "Plants convert sunlight to energy" }
    ],
    
    // Technology
    technology: [
        { invention: "Internet", year: "1983", inventor: "Vint Cerf, Bob Kahn" },
        { invention: "Smartphone", year: "2007", inventor: "Apple (iPhone)" },
        { invention: "Artificial Intelligence", year: "1956", description: "Machines that think" },
        { invention: "World Wide Web", year: "1989", inventor: "Tim Berners-Lee" }
    ],
    
    // Search function
    search(query) {
        query = query.toLowerCase();
        let results = [];
        
        // Search countries
        this.countries.forEach(country => {
            if (country.name.toLowerCase().includes(query) || 
                country.capital.toLowerCase().includes(query)) {
                results.push(`🌍 ${country.name}: Capital ${country.capital}, Population ${country.population}, Language: ${country.language}`);
            }
        });
        
        // Search books
        this.books.forEach(book => {
            if (book.title.toLowerCase().includes(query) || 
                book.author.toLowerCase().includes(query)) {
                results.push(`📚 ${book.title} by ${book.author} (${book.year}) - ${book.genre}`);
            }
        });
        
        // Search history
        this.history.forEach(item => {
            if (item.event.toLowerCase().includes(query)) {
                results.push(`📅 ${item.event} (${item.date}): ${item.description}`);
            }
        });
        
        // Search science
        this.science.forEach(item => {
            if (item.topic.toLowerCase().includes(query)) {
                results.push(`🔬 ${item.topic}: ${item.fact}`);
            }
        });
        
        return results.length > 0 ? results : null;
    },
    
    // Get random fact
    randomFact() {
        const categories = [this.countries, this.books, this.history, this.science, this.technology];
        const category = categories[Math.floor(Math.random() * categories.length)];
        const item = category[Math.floor(Math.random() * category.length)];
        
        if (category === this.countries) return `🌍 Did you know? ${item.name}'s capital is ${item.capital}`;
        if (category === this.books) return `📚 Book fact: ${item.title} by ${item.author} (${item.year})`;
        if (category === this.history) return `📅 History: ${item.event} (${item.date})`;
        if (category === this.science) return `🔬 Science: ${item.fact}`;
        return `💡 Random fact loaded!`;
    }
};
