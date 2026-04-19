# FeeKhoj Karnataka

A narrow legal-tech prototype that estimates court fees for selected Karnataka civil suits.

## What it does
- Uses a structured knowledge base derived from selected provisions of the Karnataka Court-Fees and Suits Valuation Act, 1958.
- Uses Gemini (optional) to convert plain-language user descriptions into structured facts.
- Uses deterministic logic and the ad valorem fee table to calculate an estimated fee.

## Scope covered
- Money recovery / damages / compensation suits
- Declaration + possession of immovable property
- Declaration + consequential injunction concerning immovable property
- Other declaratory suits
- Injunction concerning immovable property where title is denied / in issue
- Other injunction suits

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Deploy on Streamlit Community Cloud. Add `GOOGLE_API_KEY` or paste the key into the sidebar at runtime.

## Important limitation
This is an educational prototype. It does **not** cover all Karnataka suit categories, mixed-relief problems, jurisdiction, valuation disputes, appeals, limitation, or filing strategy.
