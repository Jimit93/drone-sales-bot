# Aerotech Drones - Autonomous Email ERP Agent

**Version 5.2 (Live Monitor & WAL Enabled)**

## 📋 Description

AI-powered Email-to-ERP automation system. Processes customer emails, classifies intent, matches products, verifies inventory, generates PDF quotes/invoices, manages sales pipeline, and tracks profitability—all autonomously. Combines Qwen LLM with ML classifiers (intent, spam) for robust natural language understanding. Admin approves discounts & bulk orders via email. Includes real-time dashboard with charts.

---

## 🎯 Core Capabilities

✅ **Email Processing** - IMAP ingestion, spam filtering, deduplication  
✅ **AI-Powered Intent Recognition** - LLM extraction + ML classification  
✅ **Product Matching** - Exact, fuzzy (KNN), and regex-based matching  
✅ **Inventory Management** - Stock verification, low-stock alerts, velocity calculation  
✅ **PDF Generation** - Professional quotations, POs, and invoices  
✅ **Sales Pipeline** - State machine from inquiry → invoice → ledger  
✅ **Owner Approval** - Discount negotiation, bulk order review via email  
✅ **Analytics Dashboard** - Real-time sales, inventory, and pipeline charts  
✅ **Audit Trail** - Immutable ledger, approval decisions, model feedback  

---

## 🤖 Why AI + ML = Efficiency

### **Problem Without AI/ML:**
```
if "quote" in email and "Mavic" in email:
    # Breaks with typos, context, synonyms
```
❌ Rules-based → brittle, unmaintainable, no learning  
❌ Manual categorization required  
❌ Handles <5% of real-world variations  

### **Solution With AI/ML:**

| Feature | Impact |
|---------|--------|
| **Qwen LLM** | Understands natural language context; extracts "5 units of Mavic 4 Pro" from "interested in buying 5 Mavic4 Pro" |
| **Logistic Regression** | Intent classification (~90% accuracy); catches domain-specific patterns (lpo, quotation, best price) |
| **Naive Bayes** | Spam filtering; blocks newsletters, auto-replies, phishing without false positives |
| **K-Nearest Neighbors** | Fuzzy product matching; "DJI Mavic4 Pro" → "DJI Mavic 4 Pro" (typo tolerance) |
| **Hybrid Fallback** | If LLM fails → use ML model → use regex rules → ask clarification (no single point of failure) |

### **Real-World Benefits:**

| Metric | Manual | Automated | Saving |
|--------|--------|-----------|--------|
| Quote generation | 5-10 min | 10 sec | **97%** ↓ |
| Intent classification | 2-5 min | 1 sec | **98%** ↓ |
| Inventory check | 3-5 min | Instant | **100%** ↓ |
| Email response | 10-20 min | 30 sec | **99%** ↓ |
| 24/7 operation | ❌ | ✅ | **Priceless** |

---

## 🗂️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Qwen2.5-1.5B GGUF (Hugging Face) |
| **ML** | scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, KNN) |
| **Workflow** | LangGraph state machine |
| **Email** | IMAP4/SMTP |
| **Database** | SQLite3 + WAL |
| **PDF** | WeasyPrint |
| **Dashboard** | Chart.js + HTML/CSS |
| **Async** | asyncio |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install langchain langchain-community langgraph llama-cpp-python \
            weasyprint scikit-learn python-dotenv huggingface-hub
```

### 2. Create `.env` File
```env
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
IMAP_SERVER=imap.gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
MAILBOX=INBOX
DB_FILE=aerotech_v5.db
ML_INTENT_MIN_CONFIDENCE=0.68
BULK_ORDER_THRESHOLD=10
```

### 3. Initialize Database & Run
```bash
python Admin_automation_Qwen.py
```

First run:
- Downloads Qwen2.5-1.5B model (~1.5 GB)
- Creates SQLite database
- Sets IMAP watermark (skips old emails)
- Starts polling every 10 seconds

---

## 📊 Sales Pipeline

```
Customer Email
    ↓
extract_requirements (LLM + ML intent)
    ↓
check_stock (inventory verification)
    ↓
[Stock OK?] ──Yes──> generate_quote → send (QUOTE_SENT)
     ↓
    [Bulk/Shortage?] ──Yes──> owner_approval → ask_owner_discount
                ↓
            (Owner replies: "APPROVE 10%")
                ↓
         apply_discount_and_requote
                ↓
    [Customer replies: "Approved"]
         quote_approval
                ↓
         generate_lpo (LPO_SENT)
                ↓
    [Customer: "Received"]
      delivery_confirmed
                ↓
      generate_invoice (INVOICE_SENT)
                ↓
     record_sale (Stock -qty, Ledger +revenue)
                ↓
    COMPLETED_AND_RECORDED
```

---

## 💾 Database Schema

| Table | Purpose |
|-------|---------|
| `orders` | Customer deal tracking |
| `order_items` | Line items (product, qty, discount) |
| `inventory` | Catalogue (price, stock, reorder level) |
| `sales` | Immutable ledger (revenue, profit) |
| `approval_audit` | Discount & bulk decisions |
| `backtest_runs` | ML model performance |
| `processed_messages` | Message-ID watermarks |

---

## 🔧 Owner API

### Add/Update Product
```python
upsert_product("DJI Mavic 4 Pro", buying_price=15000, 
               selling_price=18500, total_stock=50, reorder_level=10)
```

### Update Stock
```python
update_stock("DJI Mavic 4 Pro", quantity=35, mode="set")
update_stock("DJI Mavic 4 Pro", quantity=5, mode="subtract")
```

### Approve Discount
**Email Subject:** `DISCOUNT REQUEST: customer@email.com`  
**Email Body:** `APPROVE 10% TOTAL` or `APPROVE 15% DJI Mavic 4 Pro`

### Get Report
Send email with keywords: "sales this month", "low stock", "pipeline", "stock status"

### Interactive Dashboard
Send email: "send report" → Receives HTML dashboard with charts

---

## 📈 Dashboard Features

- **KPI Cards**: Today's units/revenue, monthly totals, profit, stock levels
- **Trend Charts**: 30-day revenue & profit trends
- **Model Performance**: Top-selling models (bar/pie charts)
- **Stock Visualization**: Current vs. reorder levels
- **Pipeline Value**: Deal stage distribution
- **Executive Summary**: Auto-generated business brief

---

## 🧪 Quality Assurance

### Backtest Results (10 test cases):
```
Intent Accuracy: 90%
Macro F1-Score: 0.88
Bulk Detection Precision: 100%
Bulk Detection Recall: 100%
```

### Safety Features:
✅ WAL (Write-Ahead Logging) for crash-safe transactions  
✅ Foreign key constraints  
✅ Atomic stock deduction  
✅ Immutable ledger  
✅ Message-ID deduplication  
✅ No credentials in logs  

---

## ⚙️ Configuration

```python
TAX_RATE = 0.18                      # 18% GST
CURRENCY = "₹"
BULK_ORDER_THRESHOLD = 10            # Units
MAX_EMAILS_PER_CYCLE = 10
POLL_SECONDS = 10
INVENTORY_SWEEP_SECONDS = 60
ML_INTENT_MIN_CONFIDENCE = 0.68
KNN_MAX_DISTANCE = 0.35              # Fuzzy match threshold
DEFAULT_REORDER_LEVEL = 10
VELOCITY_WINDOW_DAYS = 30
LOW_STOCK_URGENT_DAYS = 14
```

---

## 🔍 Example Workflow

### Customer Email:
```
Subject: Quote for DJI Mavic 4 Pro
Body: Hi, we need 5 units for our aerial business. Best price?
```

### System:
1. **Extract** → Intent: `new_inquiry`, Items: 5× DJI Mavic 4 Pro
2. **Verify Stock** → 50 in stock ✅
3. **Generate Quote** → 5 × ₹18,500 = ₹92,500 + 18% GST
4. **Send Email** → Quotation PDF attached
5. **Update DB** → Status: `QUOTE_SENT`

### Customer Approval:
```
Subject: Re: Quote for DJI Mavic 4 Pro
Body: Approved! Please proceed.
```

1. **Extract** → Intent: `quote_approval`
2. **Generate LPO** → Local Purchase Order PDF
3. **Send Email** → LPO attached
4. **Update DB** → Status: `LPO_SENT`

### Delivery Confirmation:
```
Subject: Re: Quote for DJI Mavic 4 Pro
Body: Shipment received. All units verified.
```

1. **Extract** → Intent: `delivery_confirmed`
2. **Generate Invoice** → Tax Invoice with GST
3. **Send Email** → Invoice attached
4. **Record Sale** → Stock -5, Revenue +₹92,500, Profit calculated
5. **Update DB** → Status: `COMPLETED_AND_RECORDED`

---

## 🔐 Security

- No credentials in code (uses `.env`)
- Immutable audit trail
- Transaction safety (WAL)
- Spam filtering + domain blacklist
- HTTPS-ready (SMTP STARTTLS)

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| Qwen won't download | Set `DISABLE_LOCAL_LLM=1` |
| IMAP login fails | Use App Passwords (Gmail), not plain password |
| Emails not processing | Check `POLL_SECONDS`, verify mailbox name |
| Stock not deducting | Verify product name matches catalogue exactly |

---

## 📄 License

Proprietary to Aerotech Drones. All rights reserved.

---

**Built with ❤️ using Python, Qwen LLM, scikit-learn, and asyncio magic.** ✨
