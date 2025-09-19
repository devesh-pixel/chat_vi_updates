# vic.py
import json
import tempfile
import openai
from openai import OpenAI
import requests
import os
from pathlib import Path
from typing import List, Dict

# --- LangChain (keeping your original imports; deprecation warnings are fine) ---
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

print("vic.py started")

# =========================
# Data load
# =========================
tmp_path = "investment_updates.json"
with open(tmp_path, encoding='utf-8') as f:
    data = json.load(f)

# =========================
# Build simple high-level string (unused, preserved)
# =========================
company_list = ""
for r in range(len(data['data'])):
    company = (f"Company: {data['data'][r]['companyName']}\n")
    company_list += "--------------------------------------------------\n"
    company_list += f"{company}\n"
    for update in range(len(data['data'][r]['investmentUpdates'])):
        try:
            iu = data['data'][r]['investmentUpdates']
            update_month = iu[update]['textualData']['update_month']
            revenue_type = iu[update]['kpis']['revenueType']
            update_revenue = iu[update]['kpis']['revenue']
            update_year = iu[update]['receivedYear']
            company_data = (
                f"As of Date: {update_month}, {update_year} | Lastest {revenue_type}: {update_revenue}|"
            )
            company_list += f"{company_data}\n"
        except Exception:
            continue

# =========================
# Environment / Clients
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(">> vic.py loaded")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (locally: .env or PowerShell; cloud: Secrets).")

# Explicitly pass API key (keeps old packages happy)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Company index (FAISS)
# =========================
# Step 1: Build dictionary: company name → ID
company_id_map: Dict[str, str] = {}
company_names: List[str] = []
for item in data['data']:
    name = item['companyName']
    cid = item['id']
    company_id_map[name] = cid
    company_names.append(name)

# Step 2: Embed all company names
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
company_docs = [Document(page_content=name) for name in company_names]
company_vs = FAISS.from_documents(company_docs, embedding_model)
print("faiss done, hell yeah")

# Step 3: Search helpers
def search_company(query, k=1):
    results = company_vs.similarity_search(query, k=k)
    for doc in results:
        name = doc.page_content
        company_id = company_id_map[name]
        return name, company_id

def get_data_from_id(cid):
    for r in range(len(data['data'])):
        if data['data'][r]['id'] == cid:
            return data['data'][r]

def get_data_from_name(company_name):
    name, cid = search_company(company_name)
    return get_data_from_id(cid)

# =========================
# Memory (file-backed JSONL)
# =========================
MEMORY_PATH = Path("chat_memory.jsonl")
MAX_TURNS = 8  # keep last 8 user/assistant pairs (16 messages)

def _load_memory() -> List[Dict[str, str]]:
    """
    Returns prior messages as a list of dicts:
    [{'role':'user','content':...}, {'role':'assistant','content':...}, ...]
    """
    if not MEMORY_PATH.exists():
        return []
    msgs: List[Dict[str, str]] = []
    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except Exception:
                    continue
        # keep only the last 2*MAX_TURNS messages
        return msgs[-(2 * MAX_TURNS):]
    except Exception:
        return []

def _append_memory(user_text: str, assistant_text: str) -> None:
    """Append the latest user/assistant messages to memory (best-effort)."""
    try:
        with MEMORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": user_text}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"role": "assistant", "content": assistant_text}, ensure_ascii=False) + "\n")
    except Exception:
        pass

# =========================
# JSON schema string (unchanged)
# =========================
json_structure = """ 
// Root
interface Root {
  data: Deal[];
}

// Deal (each element in data[])
interface Deal {
  id: string;
  vcFirmId: string;
  investorId: string;
  vcFundId: string;
  dealStatus: string; // e.g. "ACTIVE"
  companyName: string;
  normalizeCompanyName: string;
  email: string | null;
  companyUrl: string | null;
  sector: string | null;
  commentsAndNotes: string | null;
  commentsAndNotesSummary: string | null;
  geography: string | null;
  countryOfOperation: string | null;
  tags: string | null; // comma-separated in examples
  lastRoundValuation: string | null; // numeric string, e.g. "25000000.00"
  lastFundingRound: string | null;
  lastFundingRoundYear: number | null;
  investedInstrumentType: string | null; // e.g. "SAFE"
  termsOfSafe: string | null; // e.g. "Valuation Capped"
  priceRoundValuation: string | null;
  mfnValuation: string | null;
  valuationCap: string | null;
  discountMfn: string | null;
  freezeMfnValuation: boolean;
  safeToEquityValuation: string | null;
  investedAmount: string | null;
  investmentDate: string | null; // ISO date or null
  currentValue: string | null;
  currentValuation: string | null;
  percentageOwned: string | null; // numeric string like "0.40"
  moic: string | null; // numeric string like "1.00"
  irr: string | null;
  investmentMemoLink: string | null;
  valuationRoundDetail?: ValuationRoundDetail[] | null;
  dealStage: string | null; // e.g. "LEAD"
  proposedAmount: string | null;
  proposedValuation: string | null;
  estimatedCloseDate: string | null;
  companyPitchdeckLink: string | null;
  exitValuation: string | null;
  exitDate: string | null;
  exitPercentage: string | null;
  rejectedReason: string | null;
  statusUpdatedAt: string | null; // ISO date
  statusUpdatedBy: string | null;
  coinvestDeal: boolean;
  amountAllocated: string | null;
  entityLegalName: string | null;
  subscriptionDocDesc: string | null;
  coinvestRemoved: boolean;
  coinvestRemovedAt: string | null;
  additionalFields: any | null;
  coinvestPreviewSettings: any | null;
  startupId: string | null;
  yardstickInviteId: string | null;
  switchedToStartupYardstick: boolean;
  inviteId: string | null;
  s3Key: string | null;
  dateOfEmail: string | null; // ISO date
  dealHistory: DealHistoryEntry[];
  other: any | null;
  remark: any | null;
  otherEmails: string | null;
  starred: boolean;
  companyOneLiner: string | null;
  companyBlurp: string | null;
  aiAnalyst: AiAnalyst | null;
  memoPopulationData: any | null;
  emailAiResponse: any | null;
  unlockedAiAnalyst: boolean;
  dealSource: string | null; // e.g. "IMPORT", "ADMIN_8VDX_DEAL"
  assignedTo: string | null;
  ycDeal: boolean;
  ycBatch: string | null; // e.g. "YCW25"
  adminDealRefId: string | null;
  hasPendingUpdateRef: boolean;
  investmentUpdateType: string | null; // e.g. "MONTHLY"
  lastInvestmentUpdateAddedAt: string | null; // ISO
  investmentType: string | null;
  useOfProceeds: any | null;
  size: any | null;
  sourceName: any | null;
  emailContent: any | null;
  sourceEmail: any | null;
  sourceFirmName: any | null;
  screeningMemoS3Key: string | null;
  screeningMemoDocS3Key: string | null;
  currency: string | null;
  status: string | null;
  sourceType: string | null;
  firmName: string | null;
  asset: string | null;
  termSheet: string | null;
  dealDeadReason: string | null;
  dealDeadDate: string | null;
  investmentCommitment: string | null;
  capitalOutstanding: string | null;
  grossAssetValue: string | null;
  hedgeValueIncMtm: string | null;
  realisedProceeds: string | null;
  grossTvpi: string | null;
  companyType: string | null; // e.g. "asset-backed-lending"
  cmu: boolean;
  googleSheetsId: string | null;
  googleSheetsMetaData: any | null;
  createdAt: string; // ISO
  createdBy: string;
  updatedAt: string; // ISO
  updatedBy: string;
  jobs: Job[];
  investmentUpdates: InvestmentUpdate[];
  owner: string; // e.g. "Vijay Lavhale"
  fundName: string; // e.g. "Eight Capital Fund I"
  newReports: number;
  pastThreeMonthReleventTags: Record<string, any>;
  allPeriodWiseKpis: AllPeriodWiseKpiEntry[];
}

interface ValuationRoundDetail {
  id: string;
  date: string | null;
  createdAt: string; // ISO
  roundName: string; // may be empty string
  valuation: string; // numeric string
  defaultRound: boolean;
}

interface DealHistoryEntry {
  updatedAt: string; // ISO
  currentValuation: string | null; // numeric string or null
}

interface AiAnalyst {
  ONE_LINER: string | null;
  COMPANY_BLURB: string | null;
  investmentNoteAdmin?: {
    INVESTMENT_NOTE_PDF: string; // URL (time-limited in sample)
    INVESTMENT_NOTE_DOCX: string; // URL or empty string
  } | null;
  INVESTMENT_NOTE_DOCX?: string | null; // (present in first deal under aiAnalyst)
}

interface Job {
  id: string;
  name: string; // e.g. "Sep_2025"
  status: string; // e.g. "COMPLETED"
  dealId: string;
  investmentUpdateId: string;
  module: string; // e.g. "INVESTMENT_UPDATE"
  addedBy: string;
  emailSend: string | null;
  errorResponse: string | null;
  createdAt: string; // ISO
  updatedAt: string; // ISO
}

interface InvestmentUpdate {
  id: string;
  investorId: string;
  dealId: string;
  kpis: Kpis;                     // see below
  name: string | null;
  period: string | null;
  type: string | null;
  lastUpdated: string | null;
  textualData: TextualData;       // see below
  receivedDate: string;           // "YYYY-MM-DD"
  receivedMonth: number;          // 1-12
  receivedYear: number;
  quarter: string | null;
  dateParsed: string;             // ISO
  source: string;                 // e.g. "EMAIL"
  lastViewedAt: string | null;    // ISO
  processing: any | null;
  s3Key: string;
  dateOfEmail: string;            // "YYYY-MM-DD"
  tags: Record<string, any>;
  createdAt: string;              // ISO
  createdBy: string;
  updatedAt: string;              // ISO
  updatedBy: string | null;
  jobs: Job[];
}

// KPIs block used inside InvestmentUpdate
interface Kpis {
  gmv: number | null;
  gtv: number | null;
  runway: number | null;
  revenue: number | null;
  arr_burn: {
    to_year?: number | null;
    to_month?: number | null;
    to_quarter?: string | null;
    arr_burn_amount?: number | null;
    is_burn_given_as_ARR: boolean;
  };
  currency: string; // e.g. "USD"
  pivoting: {
    is_pivoting: boolean;
    pivoting_details: string | null;
  };
  annual_burn: {
    to_year?: number | null;
    to_month?: number | null;
    to_quarter?: string | null;
    annual_burn_amount?: number | null;
    is_burn_given_as_annual: boolean;
  };
  arr_revenue: {
    to_year?: number | null;
    to_month?: number | null;
    to_quarter?: string | null;
    burn_amount?: number | null;
    arr_revenue_amount?: number | null;
    is_revenue_given_as_ARR: boolean;
  };
  monthly_burn?: number | null;
  annual_revenue: {
    to_year?: number | null;
    to_month?: number | null;
    to_quarter?: string | null;
    annual_revenue_amount?: number | null;
    is_revenue_given_as_annual: boolean;
  };
  period_wise_kpis: PeriodWiseKpi[];
  fundraising_plans: {
    is_raising_funds: boolean;
    fundraising_details: string | null;
  };
  current_cash_balance?: number | null;
  revenueType?: string | null; // e.g. "MONTHLY" | "ARR"
  // Optional percentage change fields (present in some updates)
  runwayPercentageChange?: string | null;
  revenuePercentageChange?: string | null;
  current_cash_balancePercentageChange?: string | null;
  monthly_burnPercentageChange?: string | null;
  customers?: number | null; // present in some updates
}

interface PeriodWiseKpi {
  period: string; // e.g. "MONTHLY"
  runway?: number | null;
  revenue: number | null;
  currency: string;
  customers?: number | null;
  grossMargin?: number | null;
  monthly_burn?: number | null;
  receivedYear: number;
  receivedMonth: number;
  current_cash_balance?: number | null;
  quarter?: string | null;
}

// Textual data block on each InvestmentUpdate
interface TextualData {
  overview: string | null;
  lowlights: string | null;
  PMF_details: any | null;
  attachments: { url: string; filename: string }[];
  update_month: string; // e.g. "September"
  relevantLinks: string[];
  hiring_details: string | null;
  is_PMF_achieved: boolean;
  product_updates: {
    product_usage: string | null;
    intellectual_property: string | null;
    new_features_and_bug_fixes: string | null;
    product_roadmap_and_future_plans: string | null;
  };
  business_updates: {
    partnerships: string | null;
    team_updates: string | null;
    new_customers: string | null;
    strategic_focus: string | null;
    market_expansion_and_strategy: string | null;
    market_trends_and_competitive_analysis: string | null;
  };
  is_company_hiring: boolean;
  is_founder_leaving: boolean;
  assistance_required: string | null;
  company_name_change: {
    new_name: string | null;
    is_company_changing_name: boolean;
    company_name_change_details: string | null;
  };
  founder_leaving_details: string | null;
  explanation_for_hiring_details: string | null;
}

// Flattened KPI snapshots at deal level
interface AllPeriodWiseKpiEntry {
  receivedYear: number;
  receivedMonth: number;
  revenueTooltip: string; // e.g. "Monthly", "MRR"
  currency: string;
  period: string; // e.g. "MONTHLY"
  quarter?: string | null;
  kpis: {
    grossMargin?: number | null;
    customers?: number | null;
    current_cash_balance?: number | null;
    monthly_burn?: number | null;
    revenue?: number | null;
    runway?: number | null;
  };
}
"""

# =========================
# Python-code tool (unchanged behavior)
# =========================
def run_python_query_on_json(query: str) -> str:
    """
    Delegate execution to OpenAI's code interpreter using the Responses API.
    Uploads in-memory JSON to a temp file and attaches it.
    """
    try:
        # Save current data to a temp file
        tpath = os.path.join(tempfile.gettempdir(), "investment_data.json")
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Upload the file
        up = client.files.create(file=open(tpath, "rb"), purpose="assistants")

        # Compose input
        prompt = f"""
Use Python for this task.
The User query is: {query}

You have to use the uploaded JSON file to answer the query.
The structure of the JSON file is as follows:
{json_structure}
Also show what code you wrote to get the answer.
        """

        # Call OpenAI Code Interpreter
        resp = client.responses.create(
            model="o3",
            input=prompt,
            tools=[{
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": [up.id]}
            }]
        )

        return str(resp.output_text) if getattr(resp, "output_text", None) else "[Code interpreter returned no output]"

    except Exception as e:
        return f"[Error running Python query]: {e}"

# =========================
# Main entry: unified_answer
# =========================
def unified_answer(user_input: str):
    SYSTEM = """
You are an analyst answering questions about startup investment updates.
- If the user is asking about a single company, use the `get_data_from_name` function.
- If the user is asking to filter, compare, or list *multiple companies*, use the `run_python_query_on_json` function.
Do not guess numbers. Always cite facts from the data.
"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_data_from_name",
                "description": "Return structured data for a company",
                "parameters": {
                    "type": "object",
                    "properties": {"company_name": {"type": "string"}},
                    "required": ["company_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_python_query_on_json",
                "description": "Run a Python-based query over all company data. Use when filtering, comparing, or analyzing multiple companies.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }
    ]

    # Include prior turns from memory before current user message
    prior = _load_memory()

    # First LLM call to decide which tool to use
    resp1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM}] + prior + [
            {"role": "user", "content": user_input}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0
    )

    msg1 = resp1.choices[0].message
    tcs = msg1.tool_calls or []

    if not tcs:
        fallback = msg1.content or (
            "I'm tuned for investment‑update questions. Try:\n"
            "• Give me a summary of Rollstack for the past year\n"
            "• Which companies have revenue more than $1m?"
        )
        _append_memory(user_input, fallback)
        return fallback

    # Build message list for second call (include memory)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM}] + prior + [
        {"role": "user", "content": user_input},
        msg1  # tool call decision message
    ]

    # Execute tools and add their outputs
    for tc in tcs:
        fn_name = tc.function.name
        args = json.loads(tc.function.arguments)

        if fn_name == "get_data_from_name":
            co_data = get_data_from_name(args["company_name"])
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(co_data)
            })

        elif fn_name == "run_python_query_on_json":
            result = run_python_query_on_json(args["query"])
            print("=== Code Interpreter Output ===")
            print(result)
            print("================================")
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": str(result) if result else "[No output]"
            })

    # Final response with tool outputs injected
    resp2 = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        tools=tools
    )
    final_text = resp2.choices[0].message.content or ""
    _append_memory(user_input, final_text)
    return final_text

print("all functions processed, waiting for UI")
