import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import imaplib
import email
import email.utils
import smtplib
import uuid
import logging
import csv
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Dict, Any, Literal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from pydantic import BaseModel, Field
from weasyprint import HTML
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Securely loading environment variables instead of hardcoding to pass GitHub Push Protection
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY") 
EMAIL_USER = os.environ.get("EMAIL_ACCOUNT", "jimit93@gmail.com")  
EMAIL_PASS = os.environ.get("EMAIL_PASSWORD")     
IMAP_SERVER = "imap.gmail.com"       
SMTP_SERVER = "smtp.gmail.com"       
SMTP_PORT = 587

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
DB_FILE = "orders.csv"
HEADERS = ["Email", "Client Name", "Requested Items", "Status", "Last Updated", "Quotation Sent", "LPO Sent", "Invoice Sent"]

def initialize_database():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
        logging.info(f"Initialized database at '{DB_FILE}'")

def get_client_status(email: str) -> dict:
    if not os.path.exists(DB_FILE): return None
    with open(DB_FILE, mode='r', newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if row["Email"].strip().lower() == email.strip().lower(): return row
    return None

def update_client_status(email: str, client_name: str, items: str, new_status: str):
    rows = []
    updated = False
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(DB_FILE):
        with open(DB_FILE, mode='r', newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                if row["Email"].strip().lower() == email.strip().lower():
                    row["Status"] = new_status
                    # Protect existing items if client just replies "yes" or "approved"
                    if items: row["Requested Items"] = items
                    row["Last Updated"] = current_time
                    if new_status == "QUOTE_SENT": row["Quotation Sent"] = f"Yes - {current_time}"
                    elif new_status == "LPO_SENT": row["LPO Sent"] = f"Yes - {current_time}"
                    elif new_status == "INVOICE_SENT": row["Invoice Sent"] = f"Yes - {current_time}"
                    updated = True
                rows.append(row)
                
    if not updated:
        rows.append({
            "Email": email.lower(),
            "Client Name": client_name,
            "Requested Items": items,
            "Status": new_status,
            "Last Updated": current_time,
            "Quotation Sent": f"Yes - {current_time}" if new_status == "QUOTE_SENT" else "No",
            "LPO Sent": f"Yes - {current_time}" if new_status == "LPO_SENT" else "No",
            "Invoice Sent": f"Yes - {current_time}" if new_status == "INVOICE_SENT" else "No"
        })
        
    with open(DB_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"DATABASE UPDATE: {email} status changed to {new_status}")

initialize_database()

# ==========================================
# 3. STATE & SCHEMAS (WITH SEMANTIC UPGRADES)
# ==========================================
class AgentState(TypedDict):
    email_id: str
    sender_email: str
    display_name: str
    email_subject: str
    email_body: str
    current_db_status: Optional[str]
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "unrelated"]
    company_name: Optional[str]
    requested_items: List[Dict[str, Any]]
    generated_doc_path: Optional[str]
    doc_type_sent: Optional[str]
    reply_message: Optional[str]
    error_message: Optional[str]

class RequestedItem(BaseModel):
    product: str = Field(description="Product name matching inventory")
    quantity: int = Field(description="Quantity")

class EmailExtraction(BaseModel):
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "unrelated"] = Field(
        description="""Classify the email intent strictly based on semantic meaning:
        - 'new_inquiry': Client asks for a quotation, price, specifications, or details (e.g., 'send quote', 'what is the price', 'send specs').
        - 'quote_approval': Client approves the quote (e.g., 'Ok I want this', 'Approved', 'Agreed').
        - 'delivery_confirmed': Client confirms they received the order (e.g., 'Order received', 'Got it', 'Delivered', 'Yes I did' when asked about the order).
        - 'invoice_response': Client answers 'yes' or 'no' when asked if they received the invoice.
        - 'unrelated': Spam or unrelated chatter."""
    )
    items: List[RequestedItem] = Field(description="List of requested items. Leave empty if they are just replying yes/no/approved.")

# ==========================================
# 4. LANGGRAPH NODES
# ==========================================
def extract_requirements(state: AgentState) -> dict:
    logging.info(f"Analyzing intent for {state['display_name']}...")
    db_record = get_client_status(state['sender_email'])
    current_status = db_record["Status"] if db_record else "NEW_CLIENT"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    relevant_docs = retriever.invoke(state["email_body"])
    inventory_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EmailExtraction)
    
    prompt = PromptTemplate.from_template(
        "You are an AI sales agent. Analyze the client's email based on their status: {current_status}\n\n"
        "Inventory:\n{inventory_context}\n\n"
        "Sender: {sender}\nSubject: {subject}\nBody: {body}\n\n"
        "Determine intent and extract requested items (if mentioned)."
    )
    
    result = (prompt | structured_llm).invoke({
        "current_status": current_status,
        "inventory_context": inventory_context,
        "sender": state["display_name"],
        "subject": state["email_subject"],
        "body": state["email_body"]
    })
    
    requested_items = [{"product": item.product, "quantity": item.quantity} for item in result.items]
    
    # If LLM didn't extract items but we have them in the DB, recover them (so we can build the PDF)
    if not requested_items and db_record and db_record.get("Requested Items"):
        items_str = db_record["Requested Items"]
        for block in items_str.split(", "):
            if "x " in block:
                qty, prod = block.split("x ", 1)
                requested_items.append({"product": prod, "quantity": int(qty)})
                
    return {
        "current_db_status": current_status,
        "intent": result.intent,
        "company_name": state["display_name"],
        "requested_items": requested_items
    }

def route_workflow(state: AgentState) -> str:
    i = state["intent"]
    if i == "new_inquiry": return "generate_quote"
    elif i == "quote_approval": return "generate_lpo"
    elif i == "delivery_confirmed": return "ask_about_invoice"
    elif i == "invoice_response": return "generate_invoice"
    else: return "end"

def generate_professional_pdf(doc_type: str, state: AgentState) -> str:
    """Generates perfectly aligned, template-specific PDFs with automated 18% GST."""
    client_name = state.get("company_name", "Valued Client")
    requested_items = state.get("requested_items", [])
    current_date = datetime.now()
    date_str = current_date.strftime("%d %b %Y")
    valid_until = (current_date + timedelta(days=30)).strftime("%d %b %Y")
    run_id = str(uuid.uuid4().hex[:6]).upper()
    
    CATALOG = {
        "DJI Neo 2": {"price": 22999.00, "specs": "Weight: ~135g | 1-Axis 4K30 | 15 min flight"},
        "DJI Mini 5 Pro": {"price": 68999.00, "specs": "Weight: <249g | 50MP 4K/60fps | 38 min flight"},
        "HoverAir X1 Pro Max": {"price": 45500.00, "specs": "Weight: 193g | 8K Video | 16 min flight"},
        "DJI Air 3S": {"price": 115000.00, "specs": "Weight: 724g | Dual 50MP/48MP | 45 min flight"},
        "DJI Mavic 4 Pro": {"price": 175000.00, "specs": "Weight: 1063g | Triple Hasselblad | 51 min flight"},
        "DJI Avata 3": {"price": 79999.00, "specs": "Weight: ~375g | 4K RockSteady | 140 km/h FPV"},
        "AAF TurboFly X FPV": {"price": 74800.00, "specs": "Weight: 550g | 1080p Analog/Digital | 10 min flight"},
        "Maverick 400 RTK": {"price": 479999.00, "specs": "Weight: 1.8 kg | RTK & LiDAR | 42 min flight"},
        "DJI Matrice 4T (Thermal)": {"price": 660000.00, "specs": "Weight: 920g | Thermal & Laser | 45 min flight"},
        "EFT Z50P (50-Litre Heavy Lifter)": {"price": 874999.00, "specs": "Payload: 50L | Autonomous Crop Spraying"}
    }

    subtotal = 0.0
    table_rows = ""
    for idx, item in enumerate(requested_items, start=1):
        prod = item.get("product", "Unknown Item")
        qty = item.get("quantity", 1)
        item_data = CATALOG.get(prod, {"price": 0.00, "specs": "Standard specifications apply."})
        price = item_data["price"]
        specs = item_data["specs"]
        
        line_total = qty * price
        subtotal += line_total
        
        table_rows += f"""
        <tr>
            <td style="text-align: center;">{idx}</td>
            <td><strong>{prod}</strong><br><span class="specs">{specs}</span></td>
            <td style="text-align: center;">{qty}</td>
            <td style="text-align: right;">₹{price:,.2f}</td>
            <td style="text-align: right;">₹{line_total:,.2f}</td>
        </tr>
        """

    gst = subtotal * 0.18
    cgst = subtotal * 0.09
    sgst = subtotal * 0.09
    grand_total = subtotal + gst

    css = """
        body { font-family: 'Helvetica', sans-serif; color: #111; font-size: 12px; margin: 0; padding: 20px; }
        .header-container { border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 20px; }
        .company-title { font-size: 24px; font-weight: bold; color: #2c3e50; margin: 0; }
        .doc-title { font-size: 20px; font-weight: bold; text-align: right; text-transform: uppercase; color: #555; }
        .info-grid { display: table; width: 100%; margin-bottom: 20px; }
        .info-col { display: table-cell; width: 50%; vertical-align: top; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th { background-color: #f2f2f2; border: 1px solid #000; padding: 8px; font-size: 11px; text-transform: uppercase; }
        td { border: 1px solid #000; padding: 8px; vertical-align: top; }
        .specs { font-size: 10px; color: #555; }
        .totals-table { width: 40%; float: right; border-collapse: collapse; }
        .totals-table td { border: 1px solid #000; padding: 6px; }
        .totals-table .bold { font-weight: bold; background-color: #f9f9f9; }
        .footer-grid { display: table; width: 100%; margin-top: 40px; }
        .footer-col { display: table-cell; width: 50%; vertical-align: bottom; }
        .signature-line { border-top: 1px solid #000; width: 200px; padding-top: 5px; text-align: center; margin-top: 50px; }
        .page-num { position: fixed; bottom: 20px; width: 100%; text-align: center; font-size: 10px; }
    """

    if doc_type == "Quotation":
        html = f"""
        <html><head><style>{css}</style></head><body>
            <div class="header-container">
                <table style="border:none; margin:0; padding:0;"><tr>
                    <td style="border:none; padding:0;"><h1 class="company-title">AEROTECH DRONES</h1><p style="margin:0;">BKC, Mumbai<br>support@aerotechdrones.com</p></td>
                    <td style="border:none; padding:0;" align="right"><div class="doc-title">QUOTATION</div></td>
                </tr></table>
            </div>
            <div class="info-grid">
                <div class="info-col">
                    <strong>Customer Name:</strong><br>{client_name}
                </div>
                <div class="info-col" style="text-align: right;">
                    <strong>Quotation No:</strong> QT-{current_date.strftime("%y%m")}-{run_id}<br>
                    <strong>Quote Date:</strong> {date_str}<br>
                    <strong>Valid Until:</strong> {valid_until}<br>
                    <strong>Payment Terms:</strong> 100% Advance
                </div>
            </div>
            <table>
                <tr><th width="5%">S.NO</th><th width="45%">ITEM DESCRIPTION & SPECIFICATIONS</th><th width="10%">QTY</th><th width="20%">UNIT PRICE</th><th width="20%">TOTAL PRICE</th></tr>
                {table_rows}
            </table>
            <div>
                <div style="float: left; width: 50%;">
                    <strong>Terms & Conditions:</strong><br>
                    Quotation valid for 30 days. Delivery within 7-14 business days subject to availability.
                </div>
                <table class="totals-table">
                    <tr><td>Subtotal</td><td style="text-align: right;">₹{subtotal:,.2f}</td></tr>
                    <tr><td>Tax / GST (18%)</td><td style="text-align: right;">₹{gst:,.2f}</td></tr>
                    <tr class="bold"><td>Total Amount</td><td style="text-align: right;">₹{grand_total:,.2f}</td></tr>
                </table>
            </div>
            <div style="clear: both;"></div>
            <div class="footer-grid">
                <div class="footer-col"><div class="signature-line">Prepared By<br>Jimit Talekar</div></div>
                <div class="footer-col" align="right"><div class="signature-line" style="float:right;">Authorized Signatory</div></div>
            </div>
            <div class="page-num">Page 1</div>
        </body></html>
        """
        file_prefix = "Quotation"

    elif doc_type == "LPO":
        html = f"""
        <html><head><style>{css}</style></head><body>
            <div class="header-container">
                <table style="border:none; margin:0; padding:0;"><tr>
                    <td style="border:none; padding:0;"><h1 class="company-title">AEROTECH DRONES</h1><p style="margin:0;">Procurement Division<br>Nariman Point, Mumbai</p></td>
                    <td style="border:none; padding:0;" align="right"><div class="doc-title">LOCAL PURCHASE ORDER</div></td>
                </tr></table>
            </div>
            <div class="info-grid">
                <div class="info-col">
                    <strong>Vendor Name:</strong><br>{client_name}<br><br>
                    <strong>Quote Ref:</strong> QT-{current_date.strftime("%y%m")}-{run_id}
                </div>
                <div class="info-col" style="text-align: right;">
                    <strong>LPO Number:</strong> LPO-{current_date.strftime("%y%m")}-{run_id}<br>
                    <strong>Order Date:</strong> {date_str}<br>
                    <strong>Delivery Date:</strong> {(current_date + timedelta(days=7)).strftime("%d %b %Y")}
                </div>
            </div>
            <table>
                <tr><th width="5%">S.NO</th><th width="45%">ITEM DESCRIPTION & SPECIFICATIONS</th><th width="10%">QTY</th><th width="20%">UNIT RATE</th><th width="20%">TOTAL AMOUNT</th></tr>
                {table_rows}
            </table>
            <div>
                <div style="float: left; width: 50%;">
                    <strong>Special Instructions & Delivery Terms:</strong><br>
                    Ensure secure packaging. Notify via email upon dispatch.
                </div>
                <table class="totals-table">
                    <tr><td>Subtotal</td><td style="text-align: right;">₹{subtotal:,.2f}</td></tr>
                    <tr><td>Tax/GST (18%)</td><td style="text-align: right;">₹{gst:,.2f}</td></tr>
                    <tr class="bold"><td>Order Total</td><td style="text-align: right;">₹{grand_total:,.2f}</td></tr>
                </table>
            </div>
            <div style="clear: both;"></div>
            <div class="footer-grid">
                <div class="footer-col"><div class="signature-line">Purchasing Officer</div></div>
                <div class="footer-col" align="right"><div class="signature-line" style="float:right;">Head of Procurement</div></div>
            </div>
            <div class="page-num">Page 1</div>
        </body></html>
        """
        file_prefix = "LPO"

    elif doc_type == "Invoice":
        html = f"""
        <html><head><style>{css}</style></head><body>
            <div class="header-container">
                <table style="border:none; margin:0; padding:0;"><tr>
                    <td style="border:none; padding:0;"><h1 class="company-title">AEROTECH DRONES</h1><p style="margin:0;">BKC, Mumbai<br>GSTIN: 27AAAAA0000A1Z5</p></td>
                    <td style="border:none; padding:0;" align="right"><div class="doc-title">TAX INVOICE</div></td>
                </tr></table>
            </div>
            <div class="info-grid">
                <div class="info-col">
                    <strong>Billed To (Name):</strong><br>{client_name}<br><br>
                    <strong>GSTIN/Tax ID:</strong> UNREGISTERED
                </div>
                <div class="info-col" style="text-align: right;">
                    <strong>Invoice No:</strong> INV-{current_date.strftime("%y%m")}-{run_id}<br>
                    <strong>Invoice Date:</strong> {date_str}<br>
                    <strong>LPO/PO Ref:</strong> LPO-{current_date.strftime("%y%m")}-{run_id}
                </div>
            </div>
            <table>
                <tr><th width="5%">S.NO</th><th width="45%">ITEM DESCRIPTION & PARTICULARS</th><th width="10%">QTY</th><th width="20%">RATE</th><th width="20%">AMOUNT</th></tr>
                {table_rows}
            </table>
            <div>
                <div style="float: left; width: 50%;">
                    <strong>Bank Details for Payment:</strong><br>
                    Bank: HDFC Bank, BKC Branch<br>
                    A/C No: 50200012345678<br>
                    IFSC: HDFC0000543
                </div>
                <table class="totals-table">
                    <tr><td>Subtotal</td><td style="text-align: right;">₹{subtotal:,.2f}</td></tr>
                    <tr><td>CGST (9%)</td><td style="text-align: right;">₹{cgst:,.2f}</td></tr>
                    <tr><td>SGST (9%)</td><td style="text-align: right;">₹{sgst:,.2f}</td></tr>
                    <tr class="bold"><td>Grand Total</td><td style="text-align: right;">₹{grand_total:,.2f}</td></tr>
                </table>
            </div>
            <div style="clear: both;"></div>
            <div class="footer-grid">
                <div class="footer-col"><div class="signature-line">Received By</div></div>
                <div class="footer-col" align="right"><div class="signature-line" style="float:right;">Authorized Signatory<br>Jimit Talekar</div></div>
            </div>
            <div class="page-num">Page 1</div>
        </body></html>
        """
        file_prefix = "Invoice"

    os.makedirs("./docs", exist_ok=True)
    path = f"./docs/{file_prefix}_{run_id}.pdf"
    HTML(string=html).write_pdf(path)
    return path

def generate_quote(state: AgentState) -> dict:
    logging.info("Phase 1: Generating Quotation...")
    return {"generated_doc_path": generate_professional_pdf("Quotation", state), "doc_type_sent": "Quotation"}

def generate_lpo(state: AgentState) -> dict:
    logging.info("Phase 2: Generating Local Purchase Order...")
    return {"generated_doc_path": generate_professional_pdf("LPO", state), "doc_type_sent": "Local Purchase Order"}

# NEW CONVERSATIONAL NODE
def ask_about_invoice(state: AgentState) -> dict:
    logging.info("Phase 3: Asking if they received the invoice...")
    return {"doc_type_sent": "Invoice Inquiry", "reply_message": "Sir, have you received the invoice?", "generated_doc_path": None}

def generate_invoice(state: AgentState) -> dict:
    logging.info("Phase 4: Generating Tax Invoice...")
    return {"generated_doc_path": generate_professional_pdf("Invoice", state), "doc_type_sent": "Tax Invoice", "reply_message": "Here is your invoice sir."}

def dispatch_and_update(state: AgentState) -> dict:
    doc_type = state["doc_type_sent"]
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = state["sender_email"]
    
    if doc_type == "Invoice Inquiry":
        msg['Subject'] = "Regarding the invoice for your delivery"
        msg.attach(MIMEText(state["reply_message"], 'plain'))
        new_status = "ASKED_INVOICE_STATUS"
    else:
        msg['Subject'] = f"Your {doc_type} from Aerotech Drones"
        body = state.get("reply_message") or f"Dear {state['display_name']},\n\nPlease find your requested {doc_type} attached.\n\nBest Regards,\nJimit Talekar"
        msg.attach(MIMEText(body, 'plain'))
        
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
        server.quit()
    except Exception as e:
        return {"error_message": str(e)}

    items_str = ", ".join([f"{i['quantity']}x {i['product']}" for i in state["requested_items"]])
    update_client_status(state["sender_email"], state["display_name"], items_str, new_status)
    return {"error_message": None}

workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_requirements)
workflow.add_node("generate_quote", generate_quote)
workflow.add_node("generate_lpo", generate_lpo)
workflow.add_node("ask_about_invoice", ask_about_invoice)
workflow.add_node("generate_invoice", generate_invoice)
workflow.add_node("dispatch", dispatch_and_update)

workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_workflow, {
    "generate_quote": "generate_quote", "generate_lpo": "generate_lpo", 
    "ask_about_invoice": "ask_about_invoice", "generate_invoice": "generate_invoice", "end": END
})
workflow.add_edge("generate_quote", "dispatch")
workflow.add_edge("generate_lpo", "dispatch")
workflow.add_edge("ask_about_invoice", "dispatch")
workflow.add_edge("generate_invoice", "dispatch")
workflow.add_edge("dispatch", END)
app = workflow.compile()

# ==========================================
# 5. ASYNC WORKERS & PROACTIVE CHASER
# ==========================================
LAST_CHECKED_ID = None

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
                logging.info(f"Baseline set to UID {LAST_CHECKED_ID}. Send a NEW email now...")
                mail.logout()
                return []
            
            new_uids = [uid for uid in uids if int(uid) > LAST_CHECKED_ID]
            for uid in new_uids[-3:]:
                LAST_CHECKED_ID = max(LAST_CHECKED_ID, int(uid))
                res, msg_data = mail.uid('fetch', uid, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        raw_from = msg.get('from', '')
                        display_name, sender_email = email.utils.parseaddr(raw_from)
                        if not display_name: display_name = sender_email.split('@')[0]
                            
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
                            "display_name": display_name, "email_subject": msg['subject'], "email_body": body
                        })
        else:
            if LAST_CHECKED_ID is None:
                stat_all, msg_all = mail.uid('search', None, 'ALL')
                if stat_all == 'OK' and msg_all[0]: LAST_CHECKED_ID = int(msg_all[0].split()[-1])
                else: LAST_CHECKED_ID = 0
                logging.info("Baseline set with 0 unread emails. Send a NEW email now...")
        mail.logout()
    except Exception as e:
        pass
    return emails_data

async def email_poller(queue: asyncio.Queue):
    while True:
        new_emails = await asyncio.to_thread(fetch_unread_emails)
        for mail in new_emails:
            logging.info(f"Picked up {mail['email_subject']} from {mail['sender_email']}")
            await queue.put(mail)
        await asyncio.sleep(10)

async def agent_worker(queue: asyncio.Queue):
    while True:
        try:
            mail_data = await queue.get()
            initial_state = {
                "email_id": mail_data.get("email_id", ""), "sender_email": mail_data.get("sender_email", ""),
                "display_name": mail_data.get("display_name", ""), "email_subject": mail_data.get("email_subject", ""),
                "email_body": mail_data.get("email_body", ""), "current_db_status": None, "intent": "unrelated",
                "company_name": mail_data.get("display_name", ""), "requested_items": [], "generated_doc_path": None,
                "doc_type_sent": None, "reply_message": None, "error_message": None
            }
            await asyncio.to_thread(app.invoke, initial_state)
            queue.task_done()
            await asyncio.sleep(30)
        except Exception as e:
            queue.task_done()
            await asyncio.sleep(10)

# NEW: The Proactive AI Loop
async def lpo_chaser():
    logging.info("Starting Proactive LPO Chaser...")
    while True:
        await asyncio.sleep(60) # Scans DB every 60 seconds
        if not os.path.exists(DB_FILE): continue
        
        current_time = datetime.now()
        with open(DB_FILE, mode='r', newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                status = row["Status"]
                last_updated_str = row["Last Updated"]
                try:
                    last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                    # TRIGGER: If LPO was sent more than 2 minutes ago (Change 120 to 86400 for 24 hours)
                    if status == "LPO_SENT" and (current_time - last_updated).total_seconds() > 120:
                        logging.info(f"Client {row['Email']} went silent. Sending proactive follow-up...")
                        
                        # Dispatch Follow-up Email
                        msg = MIMEMultipart()
                        msg['From'] = EMAIL_USER
                        msg['To'] = row['Email']
                        msg['Subject'] = "Regarding to your lpo"
                        msg.attach(MIMEText("Sir, have you received your order?", 'plain'))
                        
                        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
                        server.starttls()
                        server.login(EMAIL_USER, EMAIL_PASS)
                        server.send_message(msg)
                        server.quit()
                        
                        # Update DB so we don't spam them every minute
                        update_client_status(row['Email'], row['Client Name'], row['Requested Items'], "ASKED_DELIVERY_STATUS")
                except Exception as e:
                    pass

async def main():
    email_queue = asyncio.Queue()
    # Run all three AI systems simultaneously
    await asyncio.gather(email_poller(email_queue), agent_worker(email_queue), lpo_chaser())

if __name__ == "__main__":
    # Changed from 'await main()' which crashes in standard Python scripts
    asyncio.run(main())