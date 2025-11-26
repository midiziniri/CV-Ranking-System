import requests

HEADERS = {
    "User-Agent": "MyApp/1.0 (your-email@example.com)"  # <-- replace with your app/email
}

def get_search_results(search_query):
    endpoint = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&redirects=1&srprop=size&srsearch={search_query}"
    try:
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        print(f"⚠️ Error in get_search_results: {e}")
        print("Response text (first 300 chars):", response.text[:300] if 'response' in locals() else "No response")
        return None

    results = data.get("query", {}).get("search", [])
    if results:
        title = results[0].get("title", "")
        if title:
            return get_summary(title)
    return None


def get_summary(title):
    endpoint = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exsentences=5&explaintext=&titles={title}"
    try:
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        print(f"⚠️ Error in get_summary: {e}")
        print("Response text (first 300 chars):", response.text[:300] if 'response' in locals() else "No response")
        return None

    results = data.get("query", {}).get("pages", {})
    for result in results.values():
        return result.get("extract", "")
    return None
