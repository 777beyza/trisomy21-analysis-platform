import requests

def fetch_pubmed_articles(query, max_results=5):
    """
    Fetches the latest articles from PubMed based on the search query.
    Uses NCBI E-utilities (esearch and esummary).
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Step 1: Search for article IDs (esearch)
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "pub_date" # Get the most recent ones
    }
    
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
            
        # Step 2: Fetch metadata for those IDs (esummary)
        summary_url = f"{base_url}esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        summary_response = requests.get(summary_url, params=summary_params)
        summary_response.raise_for_status()
        summary_data = summary_response.json()
        
        articles = []
        result = summary_data.get("result", {})
        
        for pmid in id_list:
            item = result.get(pmid)
            if item:
                # Extract relevant information
                title = item.get("title", "No Title")
                pub_date = item.get("pubdate", "Unknown Date")
                source = item.get("source", "Unknown Journal")
                
                # Format authors
                authors_list = item.get("authors", [])
                authors = ", ".join([auth.get("name") for auth in authors_list]) if authors_list else "Unknown Authors"
                
                articles.append({
                    "title": title,
                    "date": pub_date,
                    "journal": source,
                    "authors": authors,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
                
        return articles
        
    except Exception as e:
        print(f"PubMed API Error: {e}")
        return []
