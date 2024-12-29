import requests
import logging
import pandas as pd
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

trie = None
final_tokens = None 
inverted_index = defaultdict(set)
#Η δομή δεδομένων Trie
class TrieNode:
    def __init__(self):
        self.children = {}  # Οι χαρακτήρες των επόμενων κόμβων
        self.doc_ids = set()  # Σετ με τα doc_ids που περιέχουν τη λέξη

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, doc_id):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.doc_ids.add(doc_id)

    def search(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                return set()  # Επιστρέφουμε κενό αν δεν βρούμε τη λέξη
            current = current.children[char]
        return current.doc_ids





ps = PorterStemmer()
def default_run():
    try:
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        stop_words = set(stopwords.words('english'))
        print("Setup complete")
    except Exception as e:
        print("Error setting up : {e}")



#-------------------------------------------------------------------------------------------
#Βήμα 1 : Web Scrapping
def scrape_wikipedia(URL) :
    r = requests.get(URL)
    page = BeautifulSoup(r.text ,"html.parser")
    scrape_paragraphs(page)
    


def scrape_paragraphs(page):
    try:
        data = []
        paragraph_id = 0  # Το ID για κάθε παράγραφο
        for paragraph in page.select('p'):
            text = paragraph.getText().strip()  
            if text: 
                data.append(text)
                tokens = text.lower().split()  # Tokenize
                for token in tokens:
                    inverted_index[token].add(paragraph_id)  # Ενημερώνουμε το ευρετήριο
                paragraph_id += 1
        save_to_csv(data)
    except Exception as e:
        print(f"Error while processing paragraphs: {e}")



def save_to_csv(data):
    try:
        df = pd.DataFrame(data,columns=["Paragraph Text"])
        df.to_csv('wikipedia_data.csv', index= False , encoding='utf-8')
        print("Data saved Successfuly!")
    except Exception as e:
        print(f"Error Saving in csv File")

#-------------------------------------------------------------------------------------------
#Βήμα 2 : Text Processing

def text_processing(csvFile):
    
    try:
        df = pd.read_csv(csvFile, header=None)
        cleaned_data = []

        for text in df[0]:
            if text:
                cleaned_text = cleaning_text_and_save(text)
                cleaned_data.append({cleaned_text})

        cleaned_df = pd.DataFrame(cleaned_data)
        save_new_text(cleaned_data)
    except BaseException:
        logging.exception("An exception was thrown")


def cleaning_text_and_save(text):
    
        #Stop-word removal
        stop_words = set(stopwords.words('english'))
        #Tokenize
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
        cleaned_tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in stemmed_tokens]
        #Clearing Text
        cleaned_text = ''.join(stemmed_tokens)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
        #Final Tokens
        global final_tokens
        final_tokens = [token for token in cleaned_tokens if token]
        create_inverted_index(final_tokens)
        

        return cleaned_text



def save_new_text(cleaned_data):
    try:
        df = pd.DataFrame(cleaned_data)
        df.to_csv('cleaned_text.csv', index= False , encoding='utf-8')
        print("Cleaned data saved Successfuly!")
    except Exception as e:
        print(f"Error Saving in csv File")

#-------------------------------------------------------------------------------------------
#Βήμα 3 : Inverted Index και Υλοποιήση Δομής Δεδομένων και οτιδήποτε αυτό χρειαστει


def create_inverted_index(final_tokens):
    global inverted_index 
    inverted_index = defaultdict(set) 
    
    for idx, token in enumerate(final_tokens):
        token = token.lower()  
        if token: 
            inverted_index[token].add(idx)  
    
    with open("inverted_index.txt", 'a', encoding='utf-8') as file:
        for token, doc_ids in inverted_index.items():
            doc_ids_str = ', '.join(map(str, doc_ids))  
            file.write(f"Token: {token} | Document IDs: {doc_ids_str}\n")

  
        
def search_paragraphs(query, trie, paragraphs):
    tokens = query.lower().split()  
    result_ids = set()  

 
    for token in tokens:
        doc_ids = trie.search(token)  
        if doc_ids:
            result_ids.update(doc_ids)  

    if not result_ids:
        return ["No paragraphs found for the given query."]

    result_paragraphs = []
    invalid_doc_ids = []  
    
    for doc_id in result_ids:
        if 0 <= doc_id < len(paragraphs):
            result_paragraphs.append(paragraphs[doc_id])  
        else:
            invalid_doc_ids.append(doc_id)  

    if not result_paragraphs and invalid_doc_ids:
        return [f"Invalid doc_ids found: {invalid_doc_ids}. No valid paragraphs found."]
    

    if invalid_doc_ids:
        return [f"Some doc_ids are out of range: {invalid_doc_ids}. But no valid paragraphs found."] + result_paragraphs
    
    return result_paragraphs


def insert_inverted_index_to_trie(inverted_index, trie):
    for term, doc_ids in inverted_index.items():
        for doc_id in doc_ids:
            trie.insert(term, doc_id)


def load_inverted_index_with_trie(filename, num_paragraphs):
    trie = Trie()
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(' | ')  
                token = parts[0].split(': ')[1]  
                doc_ids_str = parts[1].split(': ')[1]  
                
                doc_ids = map(int, doc_ids_str.split(', '))
                valid_doc_ids = [doc_id for doc_id in doc_ids if 0 <= doc_id < num_paragraphs]  
                for doc_id in valid_doc_ids:
                    trie.insert(token, doc_id)  
    except Exception as e:
        print(f"Error loading the inverted index: {e}")
        return None
    
    return trie



#-------------------------------------------------------------------------------------------
#Bήμα 4 : Μηχανή αναζήτησης (η διεπαφή είναι υλοποιημένη σε άλλο αρχείο ονόματι web.py)
#α)



def search_in_inverted_index(token, inverted_index):
    return set(inverted_index.get(token, []))

def query_processing(query, trie, debug=False):
    query = query.strip()
    tokens = query.lower().split()
    
    if debug:
        print("Tokens from query:", tokens)
    
    tokens_set = set(tokens)
    
    if debug:
        print("Unique tokens set:", tokens_set)
    
    result_set = None
    current_operator = "AND"
    
    for token in tokens:
        if debug:
            print(f"Processing token: {token}")
        
        if token == "and":
            current_operator = "AND"
            continue
        elif token == "or":
            current_operator = "OR"
            continue
        elif token == "not":
            current_operator = "NOT"
            continue
        
        doc_ids = trie.search(token)  # Αναζητούμε το token στο trie
        
        if debug:
            print(f"Found doc_ids for '{token}': {doc_ids}")
        
        if result_set is None:
            result_set = doc_ids
        elif current_operator == "AND":
            result_set &= doc_ids
        elif current_operator == "OR":
            result_set |= doc_ids
        elif current_operator == "NOT":
            result_set -= doc_ids
    
    
    
    return result_set

def search_in_inverted_index(token,inverted_index):
    token = token.lower()
    print("Token in function :", token)
    if token in inverted_index:
        return inverted_index[token]
    else:
        return set()

#Βήμα 4β) Υλοποιήση αλγορίθμων

def search_tfidf(query,paragraphs,trie):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = similarity_scores.argsort()[::-1]
    ranked_paragraphs = [paragraphs[i] for i in ranked_indices if similarity_scores[i] > 0]

    return ranked_paragraphs

def search_bm25(query, paragraphs, trie): 
    tokenized_paragraphs = [word_tokenize(paragraph.lower()) for paragraph in paragraphs]
    tokenized_query = word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_paragraphs)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_paragraphs = [paragraphs[i] for i in ranked_indices if scores[i] > 0]
    
    return ranked_paragraphs


#Αξιολόγηση συστήματος 5)

def calculate_metrics(retrieved_docs, relevant_docs):
    if not retrieved_docs and not relevant_docs:
        return 0.0, 0.0, 0.0

    all_docs = retrieved_docs.union(relevant_docs)
    if not all_docs:
        return 0.0, 0.0, 0.0
    
    
    retrieved = np.array([1 if doc in retrieved_docs else 0 for doc in range(max(retrieved_docs.union(relevant_docs)) + 1)])
    relevant = np.array([1 if doc in relevant_docs else 0 for doc in range(max(retrieved_docs.union(relevant_docs)) + 1)])
    
    precision = np.sum(retrieved & relevant) / np.sum(retrieved) if np.sum(retrieved) > 0 else 0
    recall = np.sum(retrieved & relevant) / np.sum(relevant) if np.sum(relevant) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def calculate_map(retrieved_docs, relevant_docs):
    if not relevant_docs:
        return 0.0
    avg_precision = 0
    hits = 0
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            hits += 1
            avg_precision += hits / i
    return avg_precision / len(relevant_docs)

def evaluate_system(queries, relevant_docs_set, trie, paragraphs):
    results = []
    for query in queries:
        print(f"Evaluating query: {query}")
        
        retrieved_docs = query_processing(query, trie) or set()
        
        relevant_docs = relevant_docs_set.get(query, set())
        
      
        precision, recall, f1 = calculate_metrics(retrieved_docs, relevant_docs)
        map_score = calculate_map(list(retrieved_docs), relevant_docs)
        
        results.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "map": map_score
        })
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, MAP: {map_score:.2f}")
    return results



#getters    
def get_tokens():
    global final_tokens
    return final_tokens


def get_inverted_index():
    global inverted_index
    return inverted_index





def main():
    default_run() #Περιλαμβάνει το κατέβασμα των πακέτων, το setup του stemmer κτλ
    scrape_wikipedia("https://en.wikipedia.org/wiki/cristiano_ronaldo")  #Bήμα 1
    text_processing("wikipedia_data.csv") #Βήμα 2
    global inverted_index
    inverted_index = get_inverted_index()
    
    

if __name__ == "__main__":
    main()
    
    
