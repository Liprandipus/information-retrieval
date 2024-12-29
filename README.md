This repository contains both "web.py" and "app.py".
The file called "web.py" contains every single step that was asked in the work statement.
However, "app.py" contains the front-end part.


In more detail, "web.py" web scraps an wiki page. In our case scenario, this app scraps "https://en.wikipedia.org/wiki/Cristiano_Ronaldo".
This program scrapes each paragraph, it tokenizes, filters and stems each word and saves every word in an inverted index.
Afterwards, the inverted index is converted to a Trie.
Last but not least, "web.py" edits query that are given by a user using different kind of algorithms (boolean retrieval, Okapi BM25 and TF-IDF) and returns every paragraph in which the query exists.

When it comes to "app.py", it contains the code of a front-end based model that creates a simple web user interface.
The user can type words, terms or phrases separated by logical operators. Afterwards, the user chooses an algorithm that he wants his query to be found by. By clicking "Search", the results are going to be shown on the screen.



Package downloading commands :
pip install requests
pip install pandas
pip install nltk
pip install beautifulsoup4
pip install scikit-learn
pip install rank-bm25
pip install numpy

Compile commands :
python web.py
python app.py 


IP of the web UI :
http://127.0.0.1:5000
