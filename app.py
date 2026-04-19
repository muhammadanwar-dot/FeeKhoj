import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


APP_TITLE = "FeeKhoj Karnataka"
APP_SUBTITLE = "Prototype court-fee estimator for selected Karnataka civil suits"


@dataclass
class ParsedQuery:
    case_type: Optional[str]
    claim_amount: Optional[float] = None
    property_market_value: Optional[float] = None
    relief_value: Optional[float] = None
    title_denied: Optional[bool] = None
    explanation: str = ""
    confidence: str = "low"
    raw_mode: str = "heuristic"


@st.cache_data

def load_knowledge_base() -> Dict[str, Any]:
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "knowledge_base.json"), "r", encoding="utf-8") as f:
        return json.load(f)


KB = load_knowledge_base()
SUPPORTED = {item["id"]: item for item in KB["supported_case_types"]}
SLABS = KB["ad_valorem_slabs"]


def format_inr(value: float) -> str:
    rounded = round(value, 2)
    s = f"{rounded:,.2f}"
    integer, decimal = s.split(".")
    if len(integer) > 3:
        last3 = integer[-3:]
        rest = integer[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        integer = ",".join(parts + [last3])
    return f"₹{integer}.{decimal}"


def clean_number(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.lower().strip()
    multiplier = 1
    if "crore" in text or "cr" in text:
        multiplier = 10000000
    elif "lakh" in text or re.search(r"\blac\b|\blakhs?\b", text):
        multiplier = 100000
    elif "thousand" in text or "k" == text[-1:]:
        multiplier = 1000
    digits = re.sub(r"[^0-9.]", "", text)
    if not digits:
        return None
    try:
        return float(digits) * multiplier
    except ValueError:
        return None


AMOUNT_PATTERNS = [
    r"₹\s*[0-9,]+(?:\.[0-9]+)?(?:\s*(?:crore|cr|lakh|lakhs|thousand))?",
    r"rs\.?\s*[0-9,]+(?:\.[0-9]+)?(?:\s*(?:crore|cr|lakh|lakhs|thousand))?",
    r"[0-9,]+(?:\.[0-9]+)?\s*(?:crore|cr|lakh|lakhs|thousand)"
]


def extract_amounts(text: str) -> List[float]:
    values = []
    for pattern in AMOUNT_PATTERNS:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            val = clean_number(match)
            if val is not None:
                values.append(val)
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


CASE_HINTS = {
    "money_suit": ["recovery", "money", "damages", "compensation", "arrears", "refund", "loan", "dues"],
    "declaration_possession": ["declaration and possession", "declare title and possession", "declaration with possession", "possession"],
    "declaration_injunction_immovable": ["declaration and injunction", "declaration with injunction", "consequential injunction"],
    "other_declaration": ["declaration", "declaratory"],
    "injunction_title_denied": ["injunction", "title denied", "title dispute", "encroachment", "interference"],
    "other_injunction": ["injunction", "restrain", "stay"]
}


def heuristic_parse(user_text: str) -> ParsedQuery:
    text = user_text.lower()
    amounts = extract_amounts(text)
    title_denied = any(phrase in text for phrase in ["title denied", "title dispute", "denies my title", "ownership dispute", "encroachment"])

    if any(h in text for h in ["recovery", "money suit", "damages", "compensation", "arrears", "refund", "loan"]):
        return ParsedQuery(
            case_type="money_suit",
            claim_amount=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a money / recovery style claim.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text and "possession" in text:
        return ParsedQuery(
            case_type="declaration_possession",
            property_market_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a declaration + possession suit.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text and "injunction" in text and any(w in text for w in ["property", "site", "land", "flat", "house", "immovable"]):
        return ParsedQuery(
            case_type="declaration_injunction_immovable",
            property_market_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified declaration + consequential injunction concerning immovable property.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "injunction" in text and any(w in text for w in ["property", "site", "land", "flat", "house", "immovable"]):
        return ParsedQuery(
            case_type="injunction_title_denied" if title_denied else "other_injunction",
            property_market_value=amounts[0] if (title_denied and amounts) else None,
            relief_value=amounts[0] if (not title_denied and amounts) else None,
            title_denied=title_denied,
            explanation="Heuristic parser identified an injunction dispute. It used title-denial cues to choose between sections 26(a) and 26(c).",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text:
        return ParsedQuery(
            case_type="other_declaration",
            relief_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a declaratory suit but not one clearly tied to possession or consequential injunction.",
            confidence="low",
            raw_mode="heuristic"
        )

    return ParsedQuery(
        case_type=None,
        explanation="The parser could not confidently classify the query into one of the supported categories.",
        confidence="low",
        raw_mode="heuristic"
    )


LLM_PROMPT = """
You are helping a narrow legal-tech prototype that only supports selected Karnataka court-fee categories.
Read the user query and return STRICT JSON only with these keys:
- case_type: one of [money_suit, declaration_possession, declaration_injunction_immovable, other_declaration, injunction_title_denied, other_injunction, unsupported]
- claim_amount: number or null
- property_market_value: number or null
- relief_value: number or null
- title_denied: true, false, or null
- confidence: one of [high, medium, low]
- explanation: very short explanation

Rules:
- money_suit is for money recovery, damages, compensation, arrears, refund, loan recovery or similar money claims.
- declaration_possession is for declaration plus possession of immovable property.
- declaration_injunction_immovable is for declaration plus consequential injunction concerning immovable property.
- other_declaration is any other declaratory suit supported by section 24(d).
- injunction_title_denied is an injunction concerning immovable property where title is denied or ownership/title is in issue.
- other_injunction is an injunction suit not falling under the title-denied immovable property bucket.
- unsupported means the query is too ambiguous or outside these categories.
- Prefer null instead of guessing numbers.
- Convert rupee amounts into plain numeric values without commas.
"""


def llm_parse(user_text: str, api_key: str) -> Optional[ParsedQuery]:
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([LLM_PROMPT, user_text])
        raw = response.text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        case_type = data.get("case_type")
        if case_type == "unsupported":
            case_type = None
        return ParsedQuery(
            case_type=case_type,
            claim_amount=data.get("claim_amount"),
            property_market_value=data.get("property_market_value"),
            relief_value=data.get("relief_value"),
            title_denied=data.get("title_denied"),
            explanation=data.get("explanation", "LLM parser used."),
            confidence=data.get("confidence", "low"),
            raw_mode="llm"
        )
    except Exception:
        return None


def compute_ad_valorem(amount: float) -> Tuple[float, Dict[str, Any]]:
    if amount <= 0:
        raise ValueError("Amount must be positive.")
    for slab in SLABS:
        upper = slab["max_inclusive"]
        if upper is None or amount <= upper:
            fee = slab["base_fee"] + slab["rate"] * (amount - slab["base_amount"])
            return fee, slab
    raise RuntimeError("No slab matched.")


def compute_fee(parsed: ParsedQuery) -> Dict[str, Any]:
    if not parsed.case_type or parsed.case_type not in SUPPORTED:
        raise ValueError("Unsupported or unclassified case type.")

    meta = SUPPORTED[parsed.case_type]
    rule = meta["charging_rule"]
    result: Dict[str, Any] = {
        "case_type": meta["name"],
        "section": meta["section"],
        "basis_amount": None,
        "fee": None,
        "breakdown": "",
    }

    if rule == "ad_valorem_on_claim_amount":
        amount = parsed.claim_amount
        if not amount:
            raise ValueError("This category needs a claim amount.")
        fee, slab = compute_ad_valorem(amount)
        result.update({
            "basis_amount": amount,
            "fee": fee,
            "breakdown": f"Section 21 sends the suit into the ad valorem table. Under Schedule I Article 1, the amount claimed ({format_inr(amount)}) falls in the slab {format_inr(slab['base_amount'])} to {format_inr(slab['max_inclusive']) if slab['max_inclusive'] else 'above ' + format_inr(slab['base_amount'])}. Fee = base {format_inr(slab['base_fee'])} + {slab['rate']*100:.1f}% of the excess over {format_inr(slab['base_amount'])}."
        })
        return result

    if rule == "ad_valorem_on_market_value_min_1000":
        amount = parsed.property_market_value
        if not amount:
            raise ValueError("This category needs the market value of the property.")
        basis = max(amount, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section uses the full market value of the property, subject to a minimum basis of {format_inr(1000)}. On the present inputs, the fee-bearing value is {format_inr(basis)} and the Article 1 slab calculation applies from there."
        })
        return result

    if rule == "ad_valorem_on_half_market_value_min_1000":
        amount = parsed.property_market_value
        if not amount:
            raise ValueError("This category needs the market value of the property.")
        half_value = amount / 2
        basis = max(half_value, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section directs computation on one-half of the property market value, subject to a minimum basis of {format_inr(1000)}. Half of the supplied market value is {format_inr(half_value)}; the fee-bearing value is therefore {format_inr(basis)}. The Article 1 ad valorem table is then applied to that fee-bearing value."
        })
        return result

    if rule == "ad_valorem_on_relief_value_min_1000":
        amount = parsed.relief_value
        if not amount:
            raise ValueError("This category needs the value at which the relief is stated in the plaint.")
        basis = max(amount, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section uses the plaintiff's stated valuation of the relief, subject to a floor of {format_inr(1000)}. On these inputs, the fee-bearing value is {format_inr(basis)} and Article 1 governs the ad valorem amount."
        })
        return result

    raise RuntimeError("Unknown charging rule.")


st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API key (optional)", type="password", help="If added, the app uses Gemini for the fact-extraction/classification step. Without it, it falls back to a deterministic parser.")
    st.markdown("**Scope covered**")
    for item in KB["supported_case_types"]:
        st.write(f"- {item['name']}")
    st.info("This prototype is intentionally narrow. It handles selected Karnataka suit categories only.")

col1, col2 = st.columns([1.25, 1])

with col1:
    st.subheader("1) Describe the suit")
    sample = "I want to file a money recovery suit in Bengaluru for unpaid invoices of Rs. 5,50,000."
    user_query = st.text_area(
        "Enter a plain-language description",
        value=sample,
        height=180,
        help="Example: I want a declaration and injunction regarding a Bengaluru property worth Rs. 40 lakh."
    )

    st.subheader("2) Or override manually")
    manual_mode = st.toggle("Use manual override instead of parser", value=False)

    manual_case_type = None
    manual_claim = None
    manual_property = None
    manual_relief = None

    if manual_mode:
        options = {item["name"]: item["id"] for item in KB["supported_case_types"]}
        selected_name = st.selectbox("Suit category", list(options.keys()))
        manual_case_type = options[selected_name]
        needs = SUPPORTED[manual_case_type]["requires"]
        if "claim_amount" in needs:
            manual_claim = st.number_input("Claim amount (₹)", min_value=0.0, step=1000.0)
        if "property_market_value" in needs:
            manual_property = st.number_input("Property market value (₹)", min_value=0.0, step=1000.0)
        if "relief_value" in needs:
            manual_relief = st.number_input("Value placed on relief in plaint (₹)", min_value=0.0, step=1000.0)

    run = st.button("Estimate court fee", type="primary")

with col2:
    st.subheader("Knowledge base used")
    st.json({
        "jurisdiction": KB["jurisdiction"],
        "supported_case_types": [
            {"id": x["id"], "section": x["section"], "requires": x["requires"]}
            for x in KB["supported_case_types"]
        ]
    })

    with st.expander("Ad valorem slabs (Schedule I, Article 1)"):
        st.dataframe(SLABS, use_container_width=True)

if run:
    if manual_mode:
        parsed = ParsedQuery(
            case_type=manual_case_type,
            claim_amount=manual_claim if manual_claim and manual_claim > 0 else None,
            property_market_value=manual_property if manual_property and manual_property > 0 else None,
            relief_value=manual_relief if manual_relief and manual_relief > 0 else None,
            explanation="Manual override used.",
            confidence="high",
            raw_mode="manual"
        )
    else:
        parsed = llm_parse(user_query, api_key) or heuristic_parse(user_query)

    st.markdown("---")
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Structured extraction")
        st.json(parsed.__dict__)

    with b:
        try:
            result = compute_fee(parsed)
            st.subheader("Estimated court fee")
            st.metric("Approximate fee", format_inr(result["fee"]))
            st.write(f"**Category:** {result['case_type']}")
            st.write(f"**Statutory bucket used:** {result['section']}")
            st.write(f"**Fee-bearing value:** {format_inr(result['basis_amount'])}")
            st.write(result["breakdown"])
            st.warning("This is a prototype estimate for selected suit categories only. It does not determine jurisdiction, limitation, maintainability, valuation disputes, or mixed-relief complexities.")
        except Exception as exc:
            st.error(str(exc))
            st.info("The app could not compute a fee on the supplied inputs. Use the manual override or revise the query in more concrete terms.")

st.markdown("---")
st.subheader("Why this is not just a chatbot")
st.write(
    "The app uses a structured knowledge base for selected Karnataka suit categories, an optional LLM only for fact extraction/classification, and deterministic statutory logic for the actual fee calculation."
)

with st.expander("Source notes"):
    for source in KB["sources"]:
        st.write(f"- **{source['label']}**: {source['summary']} ({source['citation']})")
