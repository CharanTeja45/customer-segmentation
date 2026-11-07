# CustomerLens — Streamlit Customer Segmentation

A  mini project using K-Means clustering, Streamlit, and SQLite.
 
Put the CSV at `data/Mall_Customers.csv`. Required columns:
- Annual Income (k$)
- Spending Score (1-100)

Common additional columns are CustomerID, Gender, and Age.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Include `data/Mall_Customers.csv` in the repository.
3. Visit https://share.streamlit.io and sign in with GitHub.
4. Choose **Create app**, select the repository and branch.
5. Set the main file path to `streamlit_app.py`.
6. Click **Deploy**.

## Database note
This example uses SQLite (`customers.db`) for a simple college demonstration. Streamlit Community Cloud does not guarantee persistence of local files, so prediction history can be lost when the app restarts or is redeployed. Use a hosted PostgreSQL database for persistent production storage.

## Method note
K-Means is unsupervised. Accuracy is not a suitable metric without ground-truth labels. The app reports silhouette score and cluster profiles. Cluster labels are descriptive, not objectively verified customer categories.
