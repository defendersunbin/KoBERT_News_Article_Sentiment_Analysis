import os
from flask import Flask, request, render_template
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Initialize Flask app
app = Flask(__name__)

# Check template path (for debugging purposes)
print(f"Template path: {os.path.join(os.getcwd(), 'templates')}")

# Cassandra connection setup
auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
cluster = Cluster(['127.0.0.1'], auth_provider=auth_provider)
session = cluster.connect('advancedcassandra')

# Create a simple route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    keyword = ""
    
    # If the user submits the search form
    if request.method == "POST":
        keyword = request.form.get("keyword")
        if keyword:
            # Fetch all records from Cassandra
            rows = session.execute("SELECT content, result, date, link FROM advancedcassandra.news_sentiment_official")
            # Filter results in Python based on the keyword
            results = [
                {
                    'content': row.content,
                    'result': row.result,
                    'date': row.date,
                    'link': row.link
                }
                for row in rows if keyword.lower() in row.content.lower()
            ]

    # Render the template with the search form and results
    return render_template("index.html", results=results, keyword=keyword)

if __name__ == "__main__":
    app.run(debug=True)