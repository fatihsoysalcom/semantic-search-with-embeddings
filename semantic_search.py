import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents in Turkish
documents = [
    "Telefonumun şarjı çok hızlı bitiyor, ne yapmalıyım?",
    "Yeni bir akıllı telefon almak istiyorum, tavsiyeleriniz nelerdir?",
    "Batarya ömrünü uzatmak için ipuçları arıyorum.",
    "En iyi fotoğraf çeken telefon hangisi?",
    "Android telefonlarda pil tasarrufu nasıl yapılır?",
    "Bilgisayarım yavaş çalışıyor, çözüm önerileri.",
    "En hızlı işlemciye sahip dizüstü bilgisayar."
]

# --- Semantic Search Implementation ---
# This example uses TF-IDF as a proxy for semantic understanding.
# In a real-world scenario, you'd use pre-trained embeddings (like Sentence-BERT).

# Initialize TF-IDF Vectorizer
# This converts text into numerical vectors.
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform them into TF-IDF vectors
document_vectors = vectorizer.fit_transform(documents)

def search_documents(query, top_n=3):
    """Performs a semantic search on the documents.

    Args:
        query (str): The user's search query.
        top_n (int): The number of top results to return.

    Returns:
        list: A list of tuples (document_index, similarity_score) for the top N results.
    """
    # Transform the query into a TF-IDF vector using the same vectorizer
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query vector and all document vectors
    # Cosine similarity measures the angle between two vectors, indicating their similarity.
    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Get the indices of the top N most similar documents
    # argsort returns indices that would sort an array. We take the last 'top_n' for highest scores.
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    # Create a list of results with document index and score
    results = [(idx, similarity_scores[idx]) for idx in top_indices if similarity_scores[idx] > 0.01] # Filter out very low scores
    return results

# --- Example Usage ---

# Query 1: Similar to battery issues
query1 = "Telefonumun pili çabuk bitiyor, ne yapabilirim?"
results1 = search_documents(query1)

print(f"Query: '{query1}'")
print("Top Results:")
for idx, score in results1:
    print(f"- Score: {score:.4f}, Document: '{documents[idx]}'")
print("\n" + "-" * 30 + "\n")

# Query 2: Similar to buying a new phone
query2 = "Yeni bir cep telefonu almak istiyorum."
results2 = search_documents(query2)

print(f"Query: '{query2}'")
print("Top Results:")
for idx, score in results2:
    print(f"- Score: {score:.4f}, Document: '{documents[idx]}'")
print("\n" + "-" * 30 + "\n")

# Query 3: Less related query
query3 = "En iyi oyun bilgisayarı."
results3 = search_documents(query3)

print(f"Query: '{query3}'")
print("Top Results:")
for idx, score in results3:
    print(f"- Score: {score:.4f}, Document: '{documents[idx]}'")
