# Process Document - FeeKhoj Karnataka

## 1. Problem selection
I chose a court-fee estimation tool rather than a generic legal chatbot. This better satisfies the assignment because it performs a concrete legal task: it takes a user’s dispute description, maps it to a statutory fee category, and produces a first-cut court-fee estimate. Court-fee questions are a good legal-tech target because they sit in a bounded, rule-heavy domain: some parts can be made deterministic, but ordinary users still struggle to translate their situation into the right statutory bucket.

## 2. Why this use case is suitable for legal tech
The problem has three distinct layers:
1. **Unstructured user description** – users describe their dispute in ordinary language.
2. **Legal classification** – the description must be mapped to the correct statutory charging bucket.
3. **Computation** – once the bucket and value are identified, the actual fee should be calculated by rule, not by free-form generation.

That structure makes it a good fit for a hybrid app. The LLM is useful for extracting structured facts from messy user input, but the fee itself should be produced by deterministic code.

## 3. Scope choice and why I narrowed it
A fully general all-India court-fee calculator would have been too broad for the assignment timeframe and would have encouraged superficial treatment of the knowledge base. I instead built a narrower Karnataka-focused prototype. Even within Karnataka, I limited the app to selected civil-suit categories that illustrate different charging logics:
- money suits,
- declaration + possession,
- declaration + injunction concerning immovable property,
- other declaratory suits,
- injunction suits concerning immovable property where title is denied,
- other injunction suits.

This narrower scope made the app legally cleaner. It also made the reflection sharper, because I could identify exactly where the knowledge base mattered and where the LLM could fail.

## 4. Knowledge base construction
The knowledge base is stored as structured JSON. It has three functions:
1. identify the supported suit categories,
2. map each category to its statutory provision and charging rule,
3. store the ad valorem fee slabs from Schedule I, Article 1.

The statutory basis used in the knowledge base was drawn from the English text of the Karnataka Court-Fees and Suits Valuation Act, 1958 as accessed through official/state-hosted PDFs on the date of work. The core provisions used were:
- **Section 21**: in a suit for money, fee is computed on the amount claimed;
- **Section 24(a)**: declaration + possession → market value or Rs 1,000, whichever is higher;
- **Section 24(b)**: declaration + consequential injunction concerning immovable property → one-half of market value or Rs 1,000, whichever is higher;
- **Section 24(d)**: other declaratory suits → plaintiff’s valuation of relief or Rs 1,000, whichever is higher;
- **Section 26(a)**: injunction concerning immovable property where title is denied / in issue → one-half of market value or Rs 1,000, whichever is higher;
- **Section 26(c)**: other injunction suits → plaintiff’s valuation of relief or Rs 1,000, whichever is higher;
- **Schedule I, Article 1**: slab-based ad valorem fee table.

The most important design decision was to represent the law not as paragraphs of text, but as machine-usable charging rules. That is what made the app operational rather than merely informational.

## 5. App architecture
The app has three layers.

### A. Structured legal layer
A JSON knowledge base stores supported categories and the charging logic for each one.

### B. LLM extraction layer
The app optionally uses Gemini to convert plain-language input into a narrow JSON schema. The model is not asked to calculate the fee. It is only asked to classify the user’s problem and extract the relevant monetary value.

Example schema:
```json
{
  "case_type": "money_suit",
  "claim_amount": 550000,
  "property_market_value": null,
  "relief_value": null,
  "title_denied": null,
  "confidence": "high"
}
```

This is a meaningful LLM use because it reduces the burden on users, but it is also intentionally narrow. The LLM does not decide the legal consequence beyond the initial classification step.

### C. Deterministic computation layer
Once the structured facts are extracted, Python code applies the statutory charging rule and then uses the Schedule I ad valorem slab table to calculate the fee. This was an important methodological choice. In a legal setting, arithmetic and schedule application should be deterministic wherever possible.

## 6. Tools used
- **Python + Streamlit** for the user interface and deployment-friendly app structure.
- **Gemini API (optional)** for natural-language extraction and classification.
- **JSON knowledge base** for statutory categories and fee slabs.

I chose Streamlit because it allowed me to build and deploy a functioning public-facing prototype quickly without spending disproportionate time on front-end engineering.

## 7. How I implemented the logic
The calculation engine works in the following sequence:
1. accept a plain-language user query;
2. send it either to the LLM parser or to a heuristic parser fallback;
3. obtain a structured output containing suit type and value inputs;
4. identify the correct charging rule from the knowledge base;
5. determine the fee-bearing value (e.g., amount claimed, full market value, half market value, or relief value);
6. apply the ad valorem table from Schedule I, Article 1;
7. return the estimated fee with a short explanation of the statutory bucket used.

## 8. Design choices made to improve legal reliability
I made several choices to avoid the app becoming a misleading chatbot:
- The LLM is not used to perform calculations.
- The app exposes the structured extraction step so the user can inspect what the parser thought the facts were.
- A manual override is included so that the user can bypass the parser and directly choose the suit category.
- The scope is explicitly narrow and the app displays warnings about unsupported complexity.

These choices were deliberate. In legal-tech design, reducing overclaim is often more valuable than adding feature breadth.

## 9. Main limitations
The app is intentionally incomplete. It does not presently handle:
- all Karnataka suit categories,
- state-to-state variation,
- mixed reliefs,
- valuation disputes,
- appeals,
- jurisdiction,
- court-fee exemptions,
- cases where the user cannot state a meaningful property value or relief valuation.

It also assumes that the user’s supplied valuation is the correct one for statutory purposes, which may not be true in practice.

## 10. What I would do next with more time
With more time, I would:
1. expand the knowledge base to more statutory sections and more states;
2. attach exact statutory extracts and citations in the interface;
3. build a verification layer requiring user confirmation of the extracted classification;
4. maintain a library of adversarial test queries to measure parser failure;
5. add filing-ready outputs such as a short fee note explaining the chosen provision and calculation path.

## 11. Conclusion
This app demonstrates a useful legal-tech principle: the strongest public-facing tools are often not those that ask the LLM to do everything, but those that use the LLM for a narrow translation task and reserve the actual legal consequence for structured, rule-based logic.
