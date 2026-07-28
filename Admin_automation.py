import nest_asyncio
nest_asyncio.apply()

import os
import time
import asyncio
import imaplib
import email
import email.utils
import smtplib
import uuid
import logging
import sqlite3
import difflib
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Dict, Any, Literal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from pydantic import BaseModel, Field
from weasyprint import HTML
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
EMAIL_USER = os.environ.get("EMAIL_ACCOUNT", "jimit93@gmail.com")
EMAIL_PASS = os.environ.get("EMAIL_PASSWORD", "")
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
DB_FILE = "aerotech.db"

LOW_STOCK_THRESHOLD = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INITIAL_INVENTORY = [
    ("DJI Neo 2", 16000.0, 22999.0, 49, 1, "Weight: ~135g | 1-Axis 4K30 | 15 min flight"),
    ("DJI Mini 5 Pro", 52000.0, 68999.0, 40, 0, "Weight: <249g | 50MP 4K/60fps | 38 min flight"),
    ("HoverAir X1 Pro Max", 34000.0, 45500.0, 30, 0, "Weight: 193g | 8K Video | 16 min flight"),
    ("DJI Air 3S", 85000.0, 115000.0, 25, 0, "Weight: 724g | Dual 50MP/48MP | 45 min flight"),
    ("DJI Mavic 4 Pro", 130000.0, 175000.0, 15, 0, "Weight: 1063g | Triple Hasselblad | 51 min flight"),
    ("DJI Avata 3", 60000.0, 79999.0, 20, 0, "Weight: ~375g | 4K RockSteady | 140 km/h FPV"),
    ("AAF TurboFly X FPV", 55000.0, 74800.0, 30, 0, "Weight: 550g | 1080p Analog/Digital | 10 min flight"),
    ("Maverick 400 RTK", 350000.0, 479999.0, 5, 0, "Weight: 1.8 kg | RTK & LiDAR | 42 min flight"),
    ("DJI Matrice 4T (Thermal)", 480000.0, 660000.0, 4, 0, "Weight: 920g | Thermal & Laser | 45 min flight"),
    ("EFT Z50P (50-Litre Heavy Lifter)", 650000.0, 874999.0, 2, 0, "Payload: 50L | Autonomous Crop Spraying")
]

# ==========================================
# 2. SQLITE ERP DATABASE
# ==========================================
def initialize_database():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS inventory")
        cursor.execute("DROP TABLE IF EXISTS orders")
        
        cursor.execute("""
            CREATE TABLE orders (
                email TEXT PRIMARY KEY,
                client_name TEXT,
                items TEXT,
                status TEXT,
                discount REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE inventory (
                product_name TEXT PRIMARY KEY,
                buying_price REAL,
                selling_price REAL,
                stock INTEGER,
                sales INTEGER,
                specs TEXT
            )
        """)
        cursor.executemany(
            "INSERT INTO inventory (product_name, buying_price, selling_price, stock, sales, specs) VALUES (?, ?, ?, ?, ?, ?)",
            INITIAL_INVENTORY
        )
        conn.commit()

def get_inventory_catalog():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_name, selling_price, specs FROM inventory")
        return {row[0]: {"price": row[1], "specs": row[2]} for row in cursor.fetchall()}

def get_client_status(email_addr: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE LOWER(email) = ?", (email_addr.lower(),))
        row = cursor.fetchone()
        if row:
            return {"Email": row[0], "Client Name": row[1], "Requested Items": row[2], "Status": row[3], "Discount": row[4], "Last Updated": row[5]}
    return None

def update_client_status(email_addr: str, client_name: str, items: str, new_status: str, discount: float = 0.0):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO orders (email, client_name, items, status, discount, last_updated) 
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET 
            client_name=excluded.client_name, 
            items=COALESCE(NULLIF(excluded.items, ''), orders.items), 
            status=excluded.status,
            discount=COALESCE(NULLIF(excluded.discount, 0.0), orders.discount),
            last_updated=excluded.last_updated
        """, (email_addr.lower(), client_name, items, new_status, discount, current_time))
        conn.commit()

initialize_database()

# ==========================================
# 3. AGENT STATE & STRUCTURED SCHEMAS
# ==========================================
class AgentState(TypedDict):
    email_id: str
    sender_email: str
    display_name: str
    email_subject: str
    email_body: str
    current_db_status: Optional[str]
    intent: Literal["new_inquiry", "quote_approval", "price_negotiation", "delivery_confirmed", "clarification_needed", "out_of_stock", "owner_analytics", "owner_query", "unrelated"]
    owner_command: Optional[str]
    company_name: Optional[str]
    requested_items: List[Dict[str, Any]]
    unrecognized_item_name: Optional[str]
    generated_doc_path: Optional[str]
    doc_type_sent: Optional[str]
    reply_message: Optional[str]
    discount_applied: float

class RequestedItem(BaseModel):
    product: str = Field(description="Product name mentioned by client")
    quantity: int = Field(default=1, description="Quantity requested")

class EmailExtraction(BaseModel):
    is_drone_inquiry: bool = Field(description="True if business inquiry, negotiation, or analytics request.")
    intent: Literal["new_inquiry", "quote_approval", "price_negotiation", "delivery_confirmed", "clarification_needed", "out_of_stock", "owner_analytics", "unrelated"] = Field(
        description="Classify intent. If they complain about price or ask for a discount, use 'price_negotiation'."
    )
    items: List[RequestedItem] = Field(default=[])
    unrecognized_item: Optional[str] = Field(default=None)
    clarification_prompt: Optional[str] = Field(default=None)

# ==========================================
# 4. LANGGRAPH WORKFLOW NODES
# ==========================================
def extract_and_validate_intent(state: AgentState) -> dict:
    db_record = get_client_status(state['sender_email'])
    current_status = db_record["Status"] if db_record else "NEW_CLIENT"
    discount_applied = db_record["Discount"] if db_record else 0.0
    
    body_lower = state["email_body"].strip().lower()

    # OWNER COMMANDS
    if body_lower.startswith("jimit:"):
        command = body_lower.replace("jimit:", "").strip()
        if any(k in command for k in ["report", "dashboard", "analytics", "visual"]):
            return {"intent": "owner_analytics", "requested_items": []}
        else:
            return {"intent": "owner_query", "owner_command": command, "requested_items": []}

    # CLIENT INQUIRY
    catalog = get_inventory_catalog()
    catalog_summary = "\n".join([f"- {name}: ₹{data['price']}" for name, data in catalog.items()])
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(EmailExtraction)
    
    prompt = PromptTemplate.from_template(
        "You are Aerotech Drones AI.\nSender: {sender_email}\nBody:\n{body}\nCatalog:\n{catalog_summary}\n"
        "Classify intent. If they approve a quote, use 'quote_approval'. If they receive delivery, use 'delivery_confirmed'. If they say price is high, use 'price_negotiation'."
    )
    
    result = (prompt | structured_llm).invoke({"sender_email": state["sender_email"], "body": state["email_body"], "catalog_summary": catalog_summary})
        
    if not result.is_drone_inquiry or result.intent == "unrelated": return {"intent": "unrelated"}
    if result.intent == "clarification_needed": return {"intent": "clarification_needed", "reply_message": result.clarification_prompt}
    if result.intent == "out_of_stock": return {"intent": "out_of_stock", "reply_message": f"Sorry, {result.unrecognized_item} is out of stock."}

    extracted_items = []
    catalog_keys = list(catalog.keys())
    for item in result.items:
        raw_name = item.product.strip()
        matched = next((k for k in catalog_keys if raw_name.lower() in k.lower()), None)
        if not matched:
            closest = difflib.get_close_matches(raw_name, catalog_keys, n=1, cutoff=0.4)
            if closest: matched = closest[0]
        if matched: extracted_items.append({"product": matched, "quantity": item.quantity})

    if not extracted_items and db_record and db_record.get("Requested Items"):
        for block in db_record["Requested Items"].split(", "):
            if "x " in block:
                qty, prod = block.split("x ", 1)
                extracted_items.append({"product": prod, "quantity": int(qty)})

    return {"current_db_status": current_status, "intent": result.intent, "requested_items": extracted_items, "discount_applied": discount_applied}

def route_workflow(state: AgentState) -> str:
    i = state["intent"]
    if i == "owner_analytics": return "generate_analytics"
    elif i == "owner_query": return "answer_owner_query"
    elif i == "new_inquiry": return "generate_quote"
    elif i == "price_negotiation": return "escalate_to_owner"
    elif i == "quote_approval": return "generate_lpo"
    elif i == "delivery_confirmed": return "generate_invoice"
    elif i in ["clarification_needed", "out_of_stock"]: return "dispatch_direct_message"
    else: return "end"

# ----------------- OWNER INTELLIGENCE -----------------
def escalate_to_owner(state: AgentState) -> dict:
    logging.info("Escalating negotiation to owner...")
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = "jimit93@gmail.com"
    msg['Subject'] = f"Price related issue - {state['sender_email']}"
    
    body = f"Client: {state['sender_email']}\nMessage:\n{state['email_body']}\n\nRequested:\n"
    for item in state['requested_items']: body += f"- {item['quantity']}x {item['product']}\n"
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
    except Exception: pass
    
    return {"reply_message": "I have forwarded your request to my manager. He will review the pricing and get back to you shortly.", "doc_type_sent": "Negotiation Phase"}

def answer_owner_query(state: AgentState) -> dict:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_name, stock, sales, selling_price, buying_price FROM inventory")
        inv = cursor.fetchall()
        
    turnover = sum(s * sp for (_, _, s, sp, _) in inv)
    profit = sum(s * (sp - bp) for (_, _, s, sp, bp) in inv)
    low_stock = [f"{n} ({stk} left)" for (n, stk, _, _, _) in inv if stk <= LOW_STOCK_THRESHOLD]
    
    context = f"Total Turnover: ₹{turnover:,.2f}\nTotal Profit: ₹{profit:,.2f}\nLow Stock: {', '.join(low_stock) if low_stock else 'None'}"
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_output_tokens=60)
    response = llm.invoke(f"Context:\n{context}\n\nOwner Query: {state['owner_command']}\n\nAnswer directly based on context.")
    return {"reply_message": response.content.strip()}

def generate_analytics(state: AgentState) -> dict:
    logging.info("Executing Owner Dashboard Generation...")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_name, stock, sales, selling_price, buying_price FROM inventory")
        inv_data = cursor.fetchall()

    products = [row[0] for row in inv_data]
    stocks = [row[1] for row in inv_data]
    sales = [row[2] for row in inv_data]

    total_rev = sum(s * sp for (_, _, s, sp, _) in inv_data)
    total_prof = sum(s * (sp - bp) for (_, _, s, sp, bp) in inv_data)
    
    plt.figure(figsize=(9, 5), dpi=200)
    x = range(len(products))
    width = 0.35
    plt.bar([p - width/2 for p in x], sales, width=width, label='Total Sold', color='#2ecc71')
    plt.bar([p + width/2 for p in x], stocks, width=width, label='Current Stock', color='#3498db')
    plt.xlabel('Drone Models', fontsize=12, fontweight='bold', color='#333')
    plt.ylabel('Units', fontsize=12, fontweight='bold', color='#333')
    plt.title('Aerotech Drones - Live Inventory', fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    plt.xticks(x, [p.replace("DJI ", "") for p in products], rotation=45, ha='right', fontsize=9)
    plt.legend(frameon=True, facecolor='#f9f9f9')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    os.makedirs("./docs", exist_ok=True)
    html_path = f"./docs/Aerotech_Dashboard_{datetime.now().strftime('%y%m%d_%H%M%S')}.html"
    
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Aerotech Executive Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f7f6; margin: 0; padding: 10px; color: #333; }}
            .container {{ width: 100%; max-width: 900px; margin: auto; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
            h1 {{ color: #2c3e50; font-size: 20px; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-top: 0; text-align: center; }}
            .metrics-grid {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }}
            @media (min-width: 600px) {{ .metrics-grid {{ flex-direction: row; gap: 15px; }} }}
            .metric-box {{ flex: 1; background: #3498db; color: white; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-box.profit {{ background: #2ecc71; }}
            .metric-box h2 {{ margin: 0; font-size: 22px; }}
            .metric-box p {{ margin: 5px 0 0 0; text-transform: uppercase; font-size: 11px; font-weight: bold; letter-spacing: 0.5px; }}
            .chart-container {{ width: 100%; text-align: center; margin-top: 15px; overflow: hidden; }}
            .chart-container h3 {{ font-size: 16px; margin-bottom: 10px; }}
            .chart-container img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; display: block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Aerotech Executive Dashboard</h1>
            <div class="metrics-grid">
                <div class="metric-box"><h2>₹{total_rev:,.2f}</h2><p>Gross Revenue</p></div>
                <div class="metric-box profit"><h2>₹{total_prof:,.2f}</h2><p>Net Profit</p></div>
            </div>
            <div class="chart-container">
                <h3>Visual Analytics Breakdown</h3>
                <img src="data:image/png;base64,{img_b64}" alt="Aerotech Analytics Chart">
            </div>
        </div>
    </body>
    </html>"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return {"generated_doc_path": html_path, "doc_type_sent": "Analytics Dashboard", "reply_message": "HTML report attached and optimized for mobile viewing."}

# ----------------- DOCUMENT GENERATORS -----------------
def generate_professional_pdf(doc_type: str, state: AgentState) -> str:
    client_name = state.get("company_name", state.get("display_name", "Valued Client"))
    requested_items = state.get("requested_items", [])
    discount = state.get("discount_applied", 0.0)
    current_date = datetime.now()
    catalog = get_inventory_catalog()
    run_id = str(uuid.uuid4().hex[:6]).upper()

    subtotal = 0.0
    table_rows = ""
    for idx, item in enumerate(requested_items, start=1):
        prod = item.get("product", "Standard Drone")
        qty = item.get("quantity", 1)
        price = catalog.get(prod, {"price": 0.0})["price"]
        specs = catalog.get(prod, {"specs": ""})["specs"]
        line_total = qty * price
        subtotal += line_total
        table_rows += f"""<tr><td style="text-align:center;">{idx}</td><td><strong>{prod}</strong><br><span style="font-size:10px;color:#555;">{specs}</span></td><td style="text-align:center;">{qty}</td><td style="text-align:right;">₹{price:,.2f}</td><td style="text-align:right;">₹{line_total:,.2f}</td></tr>"""

    discount_row = ""
    grand_total = subtotal
    if discount > 0 and doc_type == "Tax Invoice":
        discount_amount = subtotal * (discount / 100)
        grand_total -= discount_amount
        discount_row = f"""<tr><td colspan="4" style="text-align:right;"><strong>Discount Allowed ({discount}%):</strong></td><td style="text-align:right; color:red;">-₹{discount_amount:,.2f}</td></tr>"""

    gst = grand_total * 0.18
    final_payable = grand_total + gst

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{{font-family:Helvetica;font-size:12px;padding:20px;}}</style></head><body>
        <h2>AEROTECH DRONES - {doc_type.upper()}</h2>
        <p><strong>To:</strong> {client_name}<br><strong>Date:</strong> {current_date.strftime("%d %b %Y")}<br><strong>Ref:</strong> {run_id}</p>
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;" border="1" cellpadding="8">
        <tr style="background:#f2f2f2;"><th>S.NO</th><th>ITEM & DESCRIPTION</th><th>QTY</th><th>RATE</th><th>AMOUNT</th></tr>
        {table_rows}
        {discount_row}
        </table>
        <h3 style="text-align:right;">Total: ₹{final_payable:,.2f} (Incl 18% GST)</h3>
        <p>Authorized Signatory<br>Jimit Talekar</p>
    </body></html>"""
    
    os.makedirs("./docs", exist_ok=True)
    path = f"./docs/{doc_type}_{run_id}.pdf"
    HTML(string=html).write_pdf(path)
    return path

def generate_quote(state: AgentState) -> dict: return {"generated_doc_path": generate_professional_pdf("Quotation", state), "doc_type_sent": "Quotation"}
def generate_lpo(state: AgentState) -> dict: return {"generated_doc_path": generate_professional_pdf("LPO", state), "doc_type_sent": "Local Purchase Order"}

def generate_invoice(state: AgentState) -> dict:
    requested_items = state.get("requested_items", [])
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for item in requested_items:
            prod, qty = item.get("product"), item.get("quantity", 1)
            cursor.execute("UPDATE inventory SET stock = MAX(stock - ?, 0), sales = sales + ? WHERE product_name = ?", (qty, qty, prod))
        conn.commit()
    return {"generated_doc_path": generate_professional_pdf("Tax Invoice", state), "doc_type_sent": "Tax Invoice"}

# ----------------- DISPATCHERS -----------------
def dispatch_direct_message(state: AgentState) -> dict:
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = state["sender_email"]
    
    safe_subject = state.get('email_subject') or "Aerotech Inquiry"
    msg['Subject'] = f"Re: {safe_subject}"
    
    reply_text = state.get("reply_message") or "We have received your message and will update you shortly."
    msg.attach(MIMEText(reply_text, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        logging.error(f"Direct Message SMTP Error: {e}")
    return {}

def dispatch_and_update(state: AgentState) -> dict:
    doc_type = state.get("doc_type_sent") or "Document"
    
    if doc_type == "Negotiation Phase":
        dispatch_direct_message(state)
        update_client_status(state["sender_email"], state.get("display_name") or "Client", "", "NEGOTIATING")
        return {}

    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = state["sender_email"]
    msg['Subject'] = f"Your {doc_type} from Aerotech Drones"
    
    reply_text = state.get("reply_message") or f"Please find your {doc_type} attached."
    msg.attach(MIMEText(reply_text, 'plain'))
    
    filepath = state.get("generated_doc_path")
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(filepath))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
        msg.attach(part)
        
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        logging.info(f"Successfully dispatched {doc_type} to {state['sender_email']}")
    except Exception as e:
        logging.error(f"CRITICAL SMTP SEND ERROR: {e}")

    status_map = {"Quotation": "QUOTE_SENT", "Local Purchase Order": "LPO_SENT", "Tax Invoice": "INVOICE_SENT"}
    new_status = status_map.get(doc_type, "UNKNOWN")
    
    requested_items = state.get("requested_items") or []
    items_str = ", ".join([f"{i.get('quantity', 1)}x {i.get('product', 'Item')}" for i in requested_items])
    
    update_client_status(state["sender_email"], state.get("display_name") or "Client", items_str, new_status)
    return {}

# ----------------- GRAPH COMPILATION -----------------
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_and_validate_intent)
workflow.add_node("answer_owner_query", answer_owner_query)
workflow.add_node("escalate_to_owner", escalate_to_owner)
workflow.add_node("generate_analytics", generate_analytics)
workflow.add_node("generate_quote", generate_quote)
workflow.add_node("generate_lpo", generate_lpo)
workflow.add_node("generate_invoice", generate_invoice)
workflow.add_node("dispatch_direct_message", dispatch_direct_message)
workflow.add_node("dispatch", dispatch_and_update)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_workflow, {
    "generate_analytics": "generate_analytics",
    "answer_owner_query": "answer_owner_query",
    "escalate_to_owner": "escalate_to_owner",
    "generate_quote": "generate_quote",
    "generate_lpo": "generate_lpo", 
    "generate_invoice": "generate_invoice",
    "dispatch_direct_message": "dispatch_direct_message",
    "end": END
})

workflow.add_edge("generate_analytics", "dispatch")
workflow.add_edge("answer_owner_query", "dispatch_direct_message")
workflow.add_edge("escalate_to_owner", END)
workflow.add_edge("generate_quote", "dispatch")
workflow.add_edge("generate_lpo", "dispatch")
workflow.add_edge("generate_invoice", "dispatch")
workflow.add_edge("dispatch_direct_message", END)
workflow.add_edge("dispatch", END)

app = workflow.compile()

# ==========================================
# 5. ASYNC POLLER & FOLLOW-UP ENGINE
# ==========================================
RATE_LIMIT_SECONDS = 15.0  
PROCESSED_UIDS = set()

def fetch_unread_emails():
    global PROCESSED_UIDS
    emails_data = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select('INBOX') 
        status, messages = mail.uid('search', None, 'UNSEEN')
        
        if status == 'OK' and messages[0]:
            uids = messages[0].split()
            
            for uid in uids[-5:]:
                if uid in PROCESSED_UIDS:
                    continue
                PROCESSED_UIDS.add(uid)
                
                res, msg_data = mail.uid('fetch', uid, '(RFC822)')
                mail.uid('store', uid, '+FLAGS', '(\\Seen)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        raw_from = msg.get('from', '')
                        display_name, sender_email = email.utils.parseaddr(raw_from)
                        
                        if any(domain in sender_email.lower() for domain in ["render", "no-reply", "temu", "support"]):
                            continue

                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

                        emails_data.append({
                            "email_id": uid.decode(), "sender_email": sender_email,
                            "display_name": display_name, "email_subject": msg.get('subject', ''), "email_body": body
                        })
        mail.logout()
    except Exception as e:
        logging.error(f"IMAP Fetch Error: {e}")
    return emails_data

async def follow_up_manager():
    while True:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT email, client_name, last_updated FROM orders WHERE status = 'QUOTE_SENT'")
                for row in cursor.fetchall():
                    email_addr, name, last_updated_str = row
                    last_date = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                    if datetime.now() - last_date > timedelta(days=3):
                        logging.info(f"Sending 3-day follow-up to {email_addr}")
                        
                        msg = MIMEMultipart()
                        msg['From'] = EMAIL_USER
                        msg['To'] = email_addr
                        msg['Subject'] = "Following up on your Aerotech Quote"
                        msg.attach(MIMEText("Sir, are you interested in proceeding with the quotation we sent?\n\nRegards,\nAerotech Drones", 'plain'))
                        
                        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
                        server.starttls()
                        server.login(EMAIL_USER, EMAIL_PASS)
                        server.send_message(msg)
                        server.quit()
                        
                        cursor.execute("UPDATE orders SET status = 'FOLLOWED_UP', last_updated = ? WHERE email = ?", 
                                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), email_addr))
                        conn.commit()
        except Exception as e:
            logging.error(f"Follow Up Error: {e}")
        await asyncio.sleep(3600)

async def email_poller(queue: asyncio.Queue):
    while True:
        new_emails = await asyncio.to_thread(fetch_unread_emails)
        if new_emails:
            for mail in new_emails: await queue.put(mail)
        await asyncio.sleep(5)

async def agent_worker(queue: asyncio.Queue):
    global LAST_REQUEST_TIME
    LAST_REQUEST_TIME = 0.0
    while True:
        try:
            mail_data = await queue.get()
            current_time = time.time()
            elapsed = current_time - LAST_REQUEST_TIME
            if elapsed < RATE_LIMIT_SECONDS:
                await asyncio.sleep(RATE_LIMIT_SECONDS - elapsed)
            LAST_REQUEST_TIME = time.time()

            initial_state = {
                "email_id": mail_data.get("email_id", ""), 
                "sender_email": mail_data.get("sender_email", ""),
                "display_name": mail_data.get("display_name", ""), 
                "email_subject": mail_data.get("email_subject", ""),
                "email_body": mail_data.get("email_body", ""), 
                "current_db_status": None, "intent": "unrelated", 
                "owner_command": None,
                "company_name": mail_data.get("display_name", ""), 
                "requested_items": [], "unrecognized_item_name": None, "generated_doc_path": None,
                "doc_type_sent": None, "reply_message": None, "discount_applied": 0.0
            }
            await asyncio.to_thread(app.invoke, initial_state)
            queue.task_done()
        except Exception as e:
            logging.error(f"Agent Worker Error: {e}")
            queue.task_done()
        await asyncio.sleep(5)

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Aerotech Headless CRM is Active!")

def start_dummy_server():
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 10000))), DummyHandler)
    server.serve_forever()

async def main():
    email_queue = asyncio.Queue()
    await asyncio.gather(
        email_poller(email_queue), 
        agent_worker(email_queue), 
        follow_up_manager()
    )

if __name__ == "__main__":
    threading.Thread(target=start_dummy_server, daemon=True).start()
    asyncio.run(main())