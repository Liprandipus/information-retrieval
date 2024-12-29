from flask import Flask, request, render_template_string
import web
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

app = Flask(__name__)
final_tokens = web.get_tokens() 

def load_paragraphs():
    paragraphs = []
    with open('wikipedia_data.csv', 'r', encoding='utf-8') as file:
        for line in file:
            paragraphs.append(line.strip())
    return paragraphs

def search_paragraphs(result_set, inverted_index, paragraphs):
    result_paragraphs = []

    for doc_id in result_set:
        if 0 <= doc_id < len(paragraphs):
            result_paragraphs.append(paragraphs[doc_id])
        else:
            print(f"Invalid doc_id: {doc_id} is out of range.")
    
    if not result_paragraphs:
        return ["No valid paragraphs found."]
    return result_paragraphs



def load_inverted_index(filename, num_paragraphs):
    inverted_index = defaultdict(set)
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' | ')  
            token = parts[0].split(': ')[1] 
            doc_ids_str = parts[1].split(': ')[1]  
            
            doc_ids = map(int, doc_ids_str.split(', '))
            
            for doc_id in doc_ids:
                if 0 <= doc_id < num_paragraphs:  
                    inverted_index[token].add(doc_id)
    return inverted_index




@app.route("/", methods=["GET", "POST"])
def search():
    paragraphs = load_paragraphs()
    num_paragraphs = len(paragraphs)
    inverted_index = load_inverted_index('inverted_index.txt',num_paragraphs)  
    trie = web.load_inverted_index_with_trie('inverted_index.txt', num_paragraphs)
    result = None
    query = ""
    algorithm = "boolean" #default algorithm 
    queries = {"ronaldo","cristiano","ball","instagram","cr7","goal","scorer","foot","messi","ronaldo or cristiano","cristiano and goal"}
    retrieved_docs = set()
    relevant_docs_set = {}
    for query in queries:
        if query in inverted_index:
            relevant_docs_set[query] = inverted_index[query]
        else:
            relevant_docs_set[query] = set()  
    print(relevant_docs_set)

    if request.method == "POST":
        
        query = request.form.get("query", "").strip()
        algorithm = request.form.get("algorithm","boolean")
        
        if algorithm == "boolean":
            print(algorithm)
            retrieved_docs = web.query_processing(query, trie, debug=False)
            result_set = web.query_processing(query, trie, debug=False)
            result = [paragraphs[doc_id] for doc_id in result_set if 0 <= doc_id < len(paragraphs)]
        elif algorithm == "tfidf":
            print(algorithm)
            retrieved_docs = set(range(len(paragraphs)))
            result = web.search_tfidf(query,paragraphs,trie)
        elif algorithm == "bm25":
            print(algorithm)
            retrieved_docs = set(range(len(paragraphs)))
            result = web.search_bm25(query,paragraphs,trie) 

        relevant_docs = relevant_docs_set.get(query, set())
        precision, recall, f1 = web.calculate_metrics(retrieved_docs, relevant_docs)
        map_score = web.calculate_map(list(retrieved_docs), relevant_docs)

        print(f"Query: {query}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, MAP: {map_score:.2f}")

    return render_template_string("""
    <h1>Search Paragraphs</h1>
    <form method="POST">
    <input type="text" name="query" placeholder="Enter a word to search" value="{{ query }}">
    <label for="algorithm">Choose an algorithm:</label>
    <select name="algorithm">
        <option value="boolean">Boolean Retrieval</option>
        <option value="tfidf">TF-IDF (VSM)</option>
        <option value="bm25">Okapi BM25</option>
    </select>
    <button type="submit">Search</button>
</form>
    
    {% if result %}
        <h2>Search Results for "{{ query }}":</h2>
        <ul>
            {% if result == "No words found." %}
                <li>No paragraphs found for the given query.</li>
            {% else %}
                {% for paragraph in result %}
                    <li>{{ paragraph }}</li>
                {% endfor %}
            {% endif %}
        </ul>
    {% endif %}
    """, query=query, result=result)

if __name__ == "__main__":
    app.run(debug=True)
