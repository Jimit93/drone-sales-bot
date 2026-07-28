import nest_asyncio
nest_asyncio.apply()

import os
import re
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

# Matplotlib backend fix for headless servers
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

LOW_STOCK_THRESHOLD = 5  # Reorder trigger limit

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
# 2. SQLITE ERP DATABASE & AUTO-FIX SCHEMA
# ==========================================
def initialize_database():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                email TEXT PRIMARY KEY,
                client_name TEXT,
                items TEXT,
                status TEXT,
                last_updated TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                product_name TEXT PRIMARY KEY,
                buying_price REAL,
                selling_price REAL,
                stock INTEGER,
                sales INTEGER,
                specs TEXT
            )
        """)
        
        # Safety check: Drop and recreate inventory table if old schema columns are missing
        cursor.execute("PRAGMA table_info(inventory)")
        columns = [col[1] for col in cursor.fetchall()]
        if "sales" not in columns or "buying_price" not in columns:
            logging.info("Old inventory schema detected. Recreating table with correct columns...")
            cursor.execute("DROP TABLE inventory")
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

        cursor.execute("SELECT COUNT(*) FROM inventory")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                "INSERT INTO inventory (product_name, buying_price, selling_price, stock, sales, specs) VALUES (?, ?, ?, ?, ?, ?)",
                INITIAL_INVENTORY
            )
            logging.info("Seeded live SQLite inventory database.")
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
            return {"Email": row[0], "Client Name": row[1], "Requested Items": row[2], "Status": row[3], "Last Updated": row[4]}
    return None

def update_client_status(email_addr: str, client_name: str, items: str, new_status: str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO orders (email, client_name, items, status, last_updated) 
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET 
            client_name=excluded.client_name, 
            items=COALESCE(NULLIF(excluded.items, ''), orders.items), 
            status=excluded.status, 
            last_updated=excluded.last_updated
        """, (email_addr.lower(), client_name, items, new_status, current_time))
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
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "clarification_needed", "out_of_stock", "owner_analytics", "unrelated"]
    company_name: Optional[str]
    requested_items: List[Dict[str, Any]]
    unrecognized_item_name: Optional[str]
    generated_doc_path: Optional[str]
    doc_type_sent: Optional[str]
    reply_message: Optional[str]
    error_message: Optional[str]
    reorder_items: List[str]

class RequestedItem(BaseModel):
    product: str = Field(description="Product name or query mentioned by client")
    quantity: int = Field(default=1, description="Quantity requested")

class EmailExtraction(BaseModel):
    is_drone_inquiry: bool = Field(description="True ONLY if email is a genuine business inquiry OR an owner analytics/dashboard request.")
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "clarification_needed", "out_of_stock", "owner_analytics", "unrelated"] = Field(
        description="Classify intent"
    )
    items: List[RequestedItem] = Field(default=[], description="List of items mentioned")
    unrecognized_item: Optional[str] = Field(default=None)
    clarification_prompt: Optional[str] = Field(default=None)

# ==========================================
# 4. LANGGRAPH WORKFLOW NODES
# ==========================================
def extract_and_validate_intent(state: AgentState) -> dict:
    logging.info(f"Analyzing intent for {state['display_name']} ({state['sender_email']})...")
    db_record = get_client_status(state['sender_email'])
    current_status = db_record["Status"] if db_record else "NEW_CLIENT"
    
    body_lower = state["email_body"].strip().lower()

    # --- STRICT OWNER COMMAND BYPASS ---
    if body_lower.startswith("jimit:"):
        command = body_lower.replace("jimit:", "").strip()
        logging.info(f"Owner command detected: {command}")
        if any(k in command for k in ["report", "dashboard", "analytics", "stock", "visual"]):
            return {"intent": "owner_analytics", "current_db_status": current_status, "requested_items": []}
        else:
            return {"intent": "unrelated", "reply_message": "Command received but not recognized."}

    # --- NORMAL CLIENT INQUIRY PROCESSING ---
    catalog = get_inventory_catalog()
    catalog_summary = "\n".join([f"- {name}: ₹{data['price']}" for name, data in catalog.items()])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EmailExtraction)
    
    prompt = PromptTemplate.from_template(
        "You are an AI sales engineer for Aerotech Drones.\n"
        "Sender: {sender_email}\nSubject: {subject}\nBody:\n{body}\n\n"
        "Available Catalog:\n{catalog_summary}\n\n"
        "Classify the intent strictly. Only output standard intents."
    )
    
    result = (prompt | structured_llm).invoke({
        "sender_email": state["sender_email"],
        "subject": state["email_subject"],
        "body": state["email_body"],
        "catalog_summary": catalog_summary
    })
        
    if not result.is_drone_inquiry or result.intent == "unrelated":
        return {"intent": "unrelated", "reply_message": None}

    if result.intent == "clarification_needed":
        return {"intent": "clarification_needed", "reply_message": result.clarification_prompt or "Could you specify the exact model?"}

    if result.intent == "out_of_stock":
        item_name = result.unrecognized_item or "the requested model"
        return {"intent": "out_of_stock", "reply_message": f"Sorry sir, {item_name} is not available in our stock right now.", "unrecognized_item_name": item_name}

    extracted_items = []
    catalog_keys = list(catalog.keys())
    for item in result.items:
        raw_name = item.product.strip()
        matched_name = None
        for cat_key in catalog_keys:
            if raw_name.lower() in cat_key.lower() or cat_key.lower() in raw_name.lower():
                matched_name = cat_key
                break
        if not matched_name:
            closest = difflib.get_close_matches(raw_name, catalog_keys, n=1, cutoff=0.4)
            if closest: matched_name = closest[0]
        if matched_name:
            extracted_items.append({"product": matched_name, "quantity": item.quantity})

    if not extracted_items and db_record and db_record.get("Requested Items"):
        for block in db_record["Requested Items"].split(", "):
            if "x " in block:
                qty, prod = block.split("x ", 1)
                extracted_items.append({"product": prod, "quantity": int(qty)})

    return {"current_db_status": current_status, "intent": result.intent, "requested_items": extracted_items}

def route_workflow(state: AgentState) -> str:
    i = state["intent"]
    if i == "owner_analytics": return "generate_analytics"
    elif i == "new_inquiry": return "generate_quote"
    elif i == "quote_approval": return "generate_lpo"
    elif i == "delivery_confirmed": return "ask_about_invoice"
    elif i == "invoice_response": return "generate_invoice"
    elif i in ["clarification_needed", "out_of_stock"]: return "dispatch_direct_message"
    else: return "end"

# ----------------- STRICT OWNER COMMUNICATION & PURE HTML DASHBOARD -----------------
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

# ----------------- STANDARD DOCUMENT GENERATORS -----------------
def generate_professional_pdf(doc_type: str, state: AgentState) -> str:
    client_name = state.get("company_name", state.get("display_name", "Valued Client"))
    requested_items = state.get("requested_items", [])
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

    gst = subtotal * 0.18
    grand_total = subtotal + gst
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{{font-family:Helvetica;font-size:12px;padding:20px;}}</style></head><body>
        <h2>AEROTECH DRONES - {doc_type.upper()}</h2>
        <p><strong>To:</strong> {client_name}<br><strong>Date:</strong> {current_date.strftime("%d %b %Y")}<br><strong>Ref:</strong> {run_id}</p>
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;" border="1" cellpadding="8">
        <tr style="background:#f2f2f2;"><th>S.NO</th><th>ITEM & SPECS</th><th>QTY</th><th>RATE</th><th>AMOUNT</th></tr>
        {table_rows}
        </table>
        <h3 style="text-align:right;">Total: ₹{grand_total:,.2f} (Incl 18% GST)</h3>
        <p>Authorized Signatory<br>Jimit Talekar</p>
    </body></html>"""
    
    os.makedirs("./docs", exist_ok=True)
    path = f"./docs/{doc_type}_{run_id}.pdf"
    HTML(string=html).write_pdf(path)
    return path

def generate_quote(state: AgentState) -> dict:
    return {"generated_doc_path": generate_professional_pdf("Quotation", state), "doc_type_sent": "Quotation"}

def generate_lpo(state: AgentState) -> dict:
    return {"generated_doc_path": generate_professional_pdf("LPO", state), "doc_type_sent": "Local Purchase Order"}

def ask_about_invoice(state: AgentState) -> dict:
    return {"doc_type_sent": "Invoice Inquiry", "reply_message": "Sir, have you received the delivery?"}

def generate_invoice(state: AgentState) -> dict:
    logging.info("Generating Tax Invoice & Checking Stock Thresholds...")
    requested_items = state.get("requested_items", [])
    reorder_list = []
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for item in requested_items:
            prod = item.get("product")
            qty = item.get("quantity", 1)
            
            cursor.execute("""
                UPDATE inventory 
                SET stock = MAX(stock - ?, 0), sales = sales + ? 
                WHERE product_name = ?
            """, (qty, qty, prod))
            
            cursor.execute("SELECT stock FROM inventory WHERE product_name = ?", (prod,))
            new_stock = cursor.fetchone()[0]
            if new_stock <= LOW_STOCK_THRESHOLD:
                reorder_list.append(f"{prod} (Current Stock: {new_stock})")
                
        conn.commit()
        
    return {
        "generated_doc_path": generate_professional_pdf("Invoice", state), 
        "doc_type_sent": "Tax Invoice", 
        "reply_message": "Here is your final tax invoice sir.",
        "reorder_items": reorder_list
    }

# ----------------- DISPATCHERS -----------------
def dispatch_direct_message(state: AgentState) -> dict:
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = state["sender_email"]
    msg['Subject'] = f"Re: {state['email_subject']}"
    msg.attach(MIMEText(state.get("reply_message", ""), 'plain'))
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
    doc_type = state["doc_type_sent"]
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = state["sender_email"]
    
    if doc_type == "Invoice Inquiry":
        msg['Subject'] = "Delivery Status"
        msg.attach(MIMEText(state["reply_message"], 'plain'))
        new_status = "ASKED_INVOICE_STATUS"
    elif doc_type == "Analytics Dashboard":
        msg['Subject'] = "Aerotech Inventory Update"
        msg.attach(MIMEText(state["reply_message"], 'plain'))
        filepath = state.get("generated_doc_path")
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(filepath))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
            msg.attach(part)
        new_status = "ANALYTICS_SENT"
    else:
        msg['Subject'] = f"Your {doc_type} from Aerotech Drones"
        msg.attach(MIMEText(state.get("reply_message", f"Please find your {doc_type} attached."), 'plain'))
        filepath = state.get("generated_doc_path")
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(filepath))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(filepath)}"'
            msg.attach(part)
        status_map = {"Quotation": "QUOTE_SENT", "Local Purchase Order": "LPO_SENT", "Tax Invoice": "INVOICE_SENT"}
        new_status = status_map.get(doc_type, "UNKNOWN")
        
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        logging.info(f"Successfully dispatched email to {state['sender_email']}")
    except Exception as e:
        logging.error(f"CRITICAL SMTP SEND ERROR: {e}")

    if doc_type != "Analytics Dashboard":
        items_str = ", ".join([f"{i['quantity']}x {i['product']}" for i in state["requested_items"]])
        update_client_status(state["sender_email"], state["display_name"], items_str, new_status)

    reorder_items = state.get("reorder_items", [])
    if reorder_items:
        alert_msg = MIMEMultipart()
        alert_msg['From'] = EMAIL_USER
        alert_msg['To'] = "jimit93@gmail.com" 
        alert_msg['Subject'] = "URGENT: Restock Alert"
        body = "Alert:\n"
        for item in reorder_items:
            body += f"{item}\n"
        alert_msg.attach(MIMEText(body, 'plain'))
        try:
            server.send_message(alert_msg)
        except Exception: pass

    try:
        server.quit()
    except Exception: pass
        
    return {}

# ----------------- GRAPH COMPILATION -----------------
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_and_validate_intent)
workflow.add_node("generate_analytics", generate_analytics)
workflow.add_node("generate_quote", generate_quote)
workflow.add_node("generate_lpo", generate_lpo)
workflow.add_node("ask_about_invoice", ask_about_invoice)
workflow.add_node("generate_invoice", generate_invoice)
workflow.add_node("dispatch_direct_message", dispatch_direct_message)
workflow.add_node("dispatch", dispatch_and_update)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_workflow, {
    "generate_analytics": "generate_analytics",
    "generate_quote": "generate_quote",
    "generate_lpo": "generate_lpo", 
    "ask_about_invoice": "ask_about_invoice",
    "generate_invoice": "generate_invoice",
    "dispatch_direct_message": "dispatch_direct_message",
    "end": END
})

workflow.add_edge("generate_analytics", "dispatch")
workflow.add_edge("generate_quote", "dispatch")
workflow.add_edge("generate_lpo", "dispatch")
workflow.add_edge("ask_about_invoice", "dispatch")
workflow.add_edge("generate_invoice", "dispatch")
workflow.add_edge("dispatch_direct_message", END)
workflow.add_edge("dispatch", END)

app = workflow.compile()

# ==========================================
# 5. ASYNC POLLER & DUMMY RENDER WEB SERVER
# ==========================================
LAST_CHECKED_ID = None
LAST_REQUEST_TIME = 0.0
RATE_LIMIT_SECONDS = 5.0  

def fetch_unread_emails():
    global LAST_CHECKED_ID
    emails_data = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select('INBOX') 
        status, messages = mail.uid('search', None, 'UNSEEN')
        if status == 'OK' and messages[0]:
            uids = messages[0].split()
            if LAST_CHECKED_ID is None:
                LAST_CHECKED_ID = int(uids[-1])
                mail.logout()
                return []
            new_uids = [uid for uid in uids if int(uid) > LAST_CHECKED_ID]
            for uid in new_uids[-3:]:
                LAST_CHECKED_ID = max(LAST_CHECKED_ID, int(uid))
                res, msg_data = mail.uid('fetch', uid, '(RFC822)')
                mail.uid('store', uid, '+FLAGS', '\\Seen')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        raw_from = msg.get('from', '')
                        display_name, sender_email = email.utils.parseaddr(raw_from)
                        
                        # ----- IGNORE AUTOMATED JUNK/RENDER EMAILS -----
                        if any(domain in sender_email.lower() for domain in ["render", "no-reply", "temu", "support"]):
                            continue
                        # -----------------------------------------------

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
        else:
            if LAST_CHECKED_ID is None: LAST_CHECKED_ID = 0
        mail.logout()
    except Exception as e:
        logging.error(f"IMAP Fetch Error: {e}")
    return emails_data

async def email_poller(queue: asyncio.Queue):
    while True:
        new_emails = await asyncio.to_thread(fetch_unread_emails)
        for mail in new_emails: await queue.put(mail)
        await asyncio.sleep(3)

async def agent_worker(queue: asyncio.Queue):
    global LAST_REQUEST_TIME
    while True:
        try:
            mail_data = await queue.get()
            
            current_time = time.time()
            elapsed = current_time - LAST_REQUEST_TIME
            if elapsed < RATE_LIMIT_SECONDS:
                wait_time = RATE_LIMIT_SECONDS - elapsed
                await asyncio.sleep(wait_time)
            
            LAST_REQUEST_TIME = time.time()

            initial_state = {
                "email_id": mail_data.get("email_id", ""), 
                "sender_email": mail_data.get("sender_email", ""),
                "display_name": mail_data.get("display_name", ""), 
                "email_subject": mail_data.get("email_subject", ""),
                "email_body": mail_data.get("email_body", ""), 
                "current_db_status": None, "intent": "unrelated", "company_name": mail_data.get("display_name", ""), 
                "requested_items": [], "unrecognized_item_name": None, "generated_doc_path": None,
                "doc_type_sent": None, "reply_message": None, "error_message": None, "reorder_items": []
            }
            await asyncio.to_thread(app.invoke, initial_state)
            queue.task_done()
        except Exception as e:
            logging.error(f"Agent Worker Error: {e}")
            queue.task_done()
        await asyncio.sleep(3)

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Aerotech Headless Email Bot is Active!")

def start_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

async def main():
    email_queue = asyncio.Queue()
    asyncio.gather(email_poller(email_queue), agent_worker(email_queue))

if __name__ == "__main__":
    threading.Thread(target=start_dummy_server, daemon=True).start()
    asyncio.run(main())