import nest_asyncio
nest_asyncio.apply()

import os
import re
import time
import asyncio
import uuid
import logging
import sqlite3
import difflib
import base64
from io import BytesIO
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field
from weasyprint import HTML
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Web Server Imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
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

initialize_database()

# ==========================================
# 3. AGENT STATE & STRUCTURED SCHEMAS
# ==========================================
class AgentState(TypedDict):
    sender_email: str
    display_name: str
    email_body: str
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "clarification_needed", "out_of_stock", "owner_analytics", "unrelated"]
    requested_items: List[Dict[str, Any]]
    unrecognized_item_name: Optional[str]
    reply_message: Optional[str]

class RequestedItem(BaseModel):
    product: str = Field(description="Product name or query mentioned by client")
    quantity: int = Field(default=1, description="Quantity requested")

class EmailExtraction(BaseModel):
    is_drone_inquiry: bool = Field(description="True ONLY if email is a genuine business inquiry OR an owner analytics/dashboard request.")
    intent: Literal["new_inquiry", "quote_approval", "delivery_confirmed", "invoice_response", "clarification_needed", "out_of_stock", "owner_analytics", "unrelated"] = Field(
        description="Classify intent strictly."
    )
    items: List[RequestedItem] = Field(default=[], description="List of items mentioned")
    unrecognized_item: Optional[str] = Field(default=None)

# ==========================================
# 4. LANGGRAPH WORKFLOW NODES
# ==========================================
def extract_and_validate_intent(state: AgentState) -> dict:
    catalog = get_inventory_catalog()
    catalog_summary = "\n".join([f"- {name}: ₹{data['price']}" for name, data in catalog.items()])
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(EmailExtraction)
    
    prompt = PromptTemplate.from_template(
        "You are an AI sales engineer for Aerotech Drones.\n"
        "Sender: {sender_email}\nMessage:\n{body}\n\n"
        "CRITICAL RULE: If the sender is 'jimit93@gmail.com' and they ask about stock, dashboard, report, or analytics, set intent to 'owner_analytics' and is_drone_inquiry to True.\n\n"
        "Available Catalog:\n{catalog_summary}\n\n"
        "Classify the intent strictly."
    )
    
    result = (prompt | structured_llm).invoke({
        "sender_email": state["sender_email"],
        "body": state["email_body"],
        "catalog_summary": catalog_summary
    })
    
    body_lower = state["email_body"].lower()
    if state["sender_email"].lower() == "jimit93@gmail.com" and any(k in body_lower for k in ["stock", "sold", "report", "how much", "dashboard", "visual"]):
        return {"intent": "owner_analytics"}
        
    if not result.is_drone_inquiry or result.intent == "unrelated":
        return {"intent": "unrelated", "reply_message": "Command not recognized."}

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

    return {"intent": result.intent, "requested_items": extracted_items}

def route_workflow(state: AgentState) -> str:
    i = state["intent"]
    if i == "owner_analytics": return "generate_analytics"
    else: return "end"

def generate_analytics(state: AgentState) -> dict:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_name, stock, sales, selling_price, buying_price FROM inventory")
        inv_data = cursor.fetchall()

    body_text = state["email_body"].lower()
    wants_report = any(k in body_text for k in ["report", "dashboard", "analytics", "visual"])

    if wants_report:
        products = [row[0] for row in inv_data]
        stocks = [row[1] for row in inv_data]
        sales = [row[2] for row in inv_data]

        total_rev = sum(s * sp for (_, _, s, sp, _) in inv_data)
        total_prof = sum(s * (sp - bp) for (_, _, s, sp, bp) in inv_data)
        
        plt.figure(figsize=(8, 4), dpi=150)
        x = range(len(products))
        width = 0.35
        plt.bar([p - width/2 for p in x], sales, width=width, label='Total Sold', color='#2ecc71')
        plt.bar([p + width/2 for p in x], stocks, width=width, label='Current Stock', color='#3498db')
        plt.xlabel('Models', fontsize=10, fontweight='bold', color='#333')
        plt.ylabel('Units', fontsize=10, fontweight='bold', color='#333')
        plt.title('Live ERP Inventory Analytics', fontsize=12, fontweight='bold', pad=10, color='#2c3e50')
        plt.xticks(x, [p.replace("DJI ", "") for p in products], rotation=45, ha='right', fontsize=8)
        plt.legend(frameon=True, facecolor='#f9f9f9')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Pure HTML response (No files saved, injected directly to chat)
        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <style>
                body {{ font-family: sans-serif; background: #fff; margin: 0; padding: 10px; color: #333; }}
                .metrics-grid {{ display: flex; gap: 10px; margin-bottom: 15px; }}
                .metric-box {{ flex: 1; background: #3498db; color: white; padding: 10px; border-radius: 8px; text-align: center; }}
                .metric-box.profit {{ background: #2ecc71; }}
                .metric-box h2 {{ margin: 0; font-size: 18px; }}
                .metric-box p {{ margin: 5px 0 0 0; font-size: 10px; text-transform: uppercase; }}
                img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="metrics-grid">
                <div class="metric-box"><h2>₹{total_rev:,.2f}</h2><p>Revenue</p></div>
                <div class="metric-box profit"><h2>₹{total_prof:,.2f}</h2><p>Profit</p></div>
            </div>
            <img src="data:image/png;base64,{img_b64}" alt="Chart">
        </body>
        </html>"""
        return {"reply_message": html_content}
    else:
        # Strict 50-token answer
        targeted_text = ""
        for name, stock, sale, sp, bp in inv_data:
            if name.lower().replace("dji ", "") in body_text or name.lower() in body_text:
                targeted_text += f"{name}: {stock} in stock.\n"
        if not targeted_text: targeted_text = "Item not recognized."
        return {"reply_message": targeted_text.strip()}

# Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_and_validate_intent)
workflow.add_node("generate_analytics", generate_analytics)
workflow.set_entry_point("extract")
workflow.add_conditional_edges("extract", route_workflow, {"generate_analytics": "generate_analytics", "end": END})
workflow.add_edge("generate_analytics", END)
app_logic = workflow.compile()

# ==========================================
# 5. FASTAPI WEB SERVER & CHAT INTERFACE
# ==========================================
app_api = FastAPI()
app_api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app_api.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Aerotech AI Terminal</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: white; margin: 0; display: flex; flex-direction: column; height: 100vh; }
            .header { background: #1e293b; padding: 15px; text-align: center; font-weight: bold; font-size: 1.2rem; border-bottom: 1px solid #334155; }
            .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
            .msg { max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 15px; line-height: 1.5; word-wrap: break-word; }
            .user { background: #3b82f6; align-self: flex-end; border-bottom-right-radius: 2px; }
            .bot { background: #1e293b; align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #334155; }
            .input-area { padding: 15px; background: #1e293b; display: flex; gap: 10px; border-top: 1px solid #334155; }
            input { flex: 1; padding: 12px; border-radius: 8px; border: none; background: #0f172a; color: white; font-size: 16px; outline: none; }
            button { background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
            button:active { background: #2563eb; }
            iframe { width: 100%; height: 400px; border: none; border-radius: 8px; background: white; margin-top: 10px;}
        </style>
    </head>
    <body>
        <div class="header">Aerotech AI ERP</div>
        <div class="chat-container" id="chat"></div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask about stock or request dashboard..." autocomplete="off" onkeypress="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            function appendMessage(sender, text) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = 'msg ' + sender;
                
                if (text.includes("<!DOCTYPE html>")) {
                    const iframe = document.createElement('iframe');
                    iframe.srcdoc = text;
                    div.appendChild(iframe);
                } else {
                    div.innerText = text;
                }
                
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const text = input.value.trim();
                if (!text) return;
                
                appendMessage('user', text);
                input.value = '';
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                
                const data = await response.json();
                appendMessage('bot', data.reply);
            }
        </script>
    </body>
    </html>
    """

@app_api.post("/api/chat")
async def process_chat(req: ChatRequest):
    initial_state = {
        "sender_email": "jimit93@gmail.com",
        "display_name": "Jimit",
        "email_body": req.message,
        "intent": "unrelated",
        "requested_items": [],
        "reply_message": ""
    }
    result = await asyncio.to_thread(app_logic.invoke, initial_state)
    return {"reply": result.get("reply_message", "Processing completed.")}

if __name__ == "__main__":
    # Local testing fallback
    uvicorn.run(app_api, host="0.0.0.0", port=10000)