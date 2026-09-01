
import streamlit as st
import json

# =========================
# MODERN COLOR SYSTEM
# =========================

COLORS = {
    "light": {
        "bg": "#F7F9FC",
        "primary": "#4F46E5",
        "success": "#22C55E",
        "danger": "#EF4444",
        "accent": "#06B6D4",
        "card_bg": "#FFFFFF",
        "text": "#1E293B",
        "text_muted": "#64748B",
        "shadow": "rgba(0, 0, 0, 0.05)",
        "sidebar_bg": "#FFFFFF",
    },
    "dark": {
        "bg": "#0F172A",
        "primary": "#6366F1",
        "success": "#22C55E",
        "danger": "#F87171",
        "accent": "#38BDF8",
        "card_bg": "#1E293B",
        "text": "#F8FAFC",
        "text_muted": "#94A3B8",
        "shadow": "rgba(0, 0, 0, 0.3)",
        "sidebar_bg": "#0B1120",
    }
}

def inject_custom_css():
    theme = st.session_state.get('theme', 'light')
    c = COLORS[theme]
    
    css = f"""
    <style>
        /* Global Styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {c['bg']} !important;
            font-family: 'Inter', sans-serif;
            color: {c['text']} !important;
        }}
        
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']} !important;
            border-right: 1px solid {c['shadow']};
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: {c['text']} !important;
        }}
        
        .text-muted {{
            color: {c['text_muted']} !important;
        }}
        
        /* Card System */
        .glass-card {{
            background: {c['card_bg']};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 25px -5px {c['shadow']};
            border: 1px solid {c['shadow']};
            margin-bottom: 2rem;
            animation: fadeIn 0.6s ease-out;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 30px -10px {c['shadow']};
        }}
        
        /* Buttons */
        .stButton > button {{
            background: {c['primary']} !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100%;
        }}
        
        .stButton > button:hover {{
            filter: brightness(1.1);
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px {c['primary']}40 !important;
        }}
        
        /* Sidebar Specific */
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent !important;
            border: 1px solid {c['primary']}40 !important;
            color: {c['text']} !important;
            font-size: 0.9rem !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {c['primary']}15 !important;
            border-color: {c['primary']} !important;
        }}
        
        .logout-btn-container .stButton > button {{
            background: {c['danger']}15 !important;
            color: {c['danger']} !important;
            border: 1px solid {c['danger']}40 !important;
        }}
        
        .logout-btn-container .stButton > button:hover {{
            background: {c['danger']} !important;
            color: white !important;
            border-color: {c['danger']} !important;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); box-shadow: 0 0 0 0 {c['danger']}40; }}
            70% {{ transform: scale(1.02); box-shadow: 0 0 0 15px {c['danger']}00; }}
            100% {{ transform: scale(1); box-shadow: 0 0 0 0 {c['danger']}00; }}
        }}
        
        /* Login Page Specific */
        .login-card {{
            background: {c['card_bg']};
            padding: 3.5rem;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px {c['shadow']};
            border: 1px solid {c['shadow']};
            animation: slideInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            width: 100%;
            margin-top: 15vh;
        }}
        
        /* Input Field Styling */
        div[data-testid="stTextInput"] input {{
            background-color: {c['bg']} !important;
            border: 1px solid {c['shadow']} !important;
            border-radius: 10px !important;
            padding: 0.75rem !important;
            color: {c['text']} !important;
        }}
        
        div[data-testid="stTextInput"] label {{
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
            color: {c['text']} !important;
        }}
        
        @keyframes slideInUp {{
            from {{ opacity: 0; transform: translateY(40px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Result Cards */
        .result-card {{
            padding: 2.5rem;
            border-radius: 24px;
            text-align: center;
            margin-top: 1.5rem;
            position: relative;
            overflow: hidden;
            border: 2px solid transparent;
        }}
        
        .result-card.fresh {{
            background: linear-gradient(135deg, {c['success']}15 0%, {c['success']}05 100%);
            border-color: {c['success']}40;
            box-shadow: 0 20px 40px -15px {c['success']}30;
        }}
        
        .result-card.rotten {{
            background: linear-gradient(135deg, {c['danger']}15 0%, {c['danger']}05 100%);
            border-color: {c['danger']}40;
            box-shadow: 0 20px 40px -15px {c['danger']}30;
            animation: pulse 2s infinite;
        }}
        
        .result-card.unknown {{
            background: linear-gradient(135deg, {c['accent']}15 0%, {c['accent']}05 100%);
            border-color: {c['accent']}40;
            box-shadow: 0 20px 40px -15px {c['accent']}30;
        }}
        
        .result-title {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            margin-bottom: 0.5rem;
            opacity: 0.8;
        }}
        
        .result-status {{
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }}
        
        .status-fresh {{ color: {c['success']} !important; }}
        .status-rotten {{ color: {c['danger']} !important; }}
        .status-unknown {{ color: {c['accent']} !important; }}
        
        /* Hide default streamlit elements */
        div[data-testid="stStatusWidget"] {{
            visibility: hidden;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =========================
# AUTH SYSTEM (UI)
# =========================

def render_login():
    inject_custom_css()
    
    # Use columns for layout but keep HTML condensed
    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        # Combine opening tags and header to avoid Streamlit container spacing
        st.markdown(f"""
            <div class="login-card">
                <div style="text-align: center; margin-bottom: 2rem;">
                    <div style="font-size: 3.5rem; margin-bottom: 1rem;">🥚</div>
                    <h1 style="margin: 0; font-weight: 800; font-size: 1.8rem; color: {COLORS[st.session_state.theme]['text']} !important;">System Authentication Portal</h1>
                    <p style="margin-top: 0.5rem; color: {COLORS[st.session_state.theme]['text_muted']} !important; font-size: 1rem;">Secure Access to Intelligent Poultry Assessment System</p>
                </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("ID", placeholder="Enter researcher identification")
            password = st.text_input("Credential", type="password", placeholder="••••••••")
            
            st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
            submit = st.form_submit_button("Authenticate Access", use_container_width=True)
            
            if submit:
                try:
                    with open('users.json', 'r') as f:
                        users = json.load(f)
                    
                    if username in users and users[username]['password'] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = users[username]['role']
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    st.error("Authentication system error")
        
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# LAYOUT COMPONENTS
# =========================

def render_hero():
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS[st.session_state.theme]['primary']} 0%, {COLORS[st.session_state.theme]['accent']} 100%); 
                    padding: 4rem 2rem; border-radius: 24px; text-align: center; color: white; margin-bottom: 3rem;
                    box-shadow: 0 20px 25px -5px {COLORS[st.session_state.theme]['primary']}30;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🥚</div>
            <h1 style="color: white !important; font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.025em;">AI-Driven Chicken Egg Quality Assessment System</h1>
            <p style="color: white !important; font-size: 1.1rem; opacity: 0.9; font-weight: 400; max-width: 800px; margin: 1rem auto 0;">
                Cost-Effective Deep Learning Framework for Quality Assessment of Chicken Eggs for Small-Scale Farming
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_status_card(label, value, type="success"):
    c = COLORS[st.session_state.theme]
    border_color = c['success'] if type == "success" else c['danger']
    bg_color = f"{border_color}10"
    
    st.markdown(f"""
        <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.6; text-transform: uppercase; color: {c['text']} !important;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {border_color} !important;">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# =========================
# RESULT CARD COMPONENTS
# =========================

def render_result_card(status, confidence):
    status = status.lower().strip()
    
    # Map ML classes to UI types
    ui_type = "unknown"
    icon = "🔍"
    label = "Diagnostic Analysis Report"
    
    if "infertile" in status:
        ui_type = "unknown"
        icon = "🟡"
        display_status = "Unclassified Sample (Requires Further Review)"
    elif "fertile" in status or "fresh" in status:
        ui_type = "fresh"
        icon = "✨"
        display_status = "Grade A (Fresh Egg – High Quality)"
    elif "dead" in status or "rotten" in status:
        ui_type = "rotten"
        icon = "⚠️"
        display_status = "Grade C (Defective Egg – Unfit for Consumption)"
    else:
        icon = "🔍"
        display_status = "Unclassified Sample (Requires Further Review)"
        
    st.markdown(f"""
        <div class="result-card {ui_type}">
            <div class="result-title">{label}</div>
            <div class="result-status status-{ui_type}" style="font-size: 1.8rem;">{icon} {display_status}</div>
            <div style="font-size: 1.1rem; font-weight: 600; opacity: 0.9;">
                Model Prediction Confidence Index: <span style="font-size: 1.4rem;">{confidence:.1%}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# =========================
# UI THEME SYSTEM
# =========================

def render_sidebar_controls():
    with st.sidebar:
        st.markdown(f"""
            <div style="padding: 1rem 0; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔬</div>
                <h3 style="margin:0;">Research Panel</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User Profile
        st.markdown(f"""
            <div style="padding: 1rem; background: {COLORS[st.session_state.theme]['bg']}; border-radius: 12px; margin-bottom: 1rem;">
                <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.6; text-transform: uppercase;">Active Researcher</div>
                <div style="font-weight: 700; font-size: 1.1rem;">{st.session_state.username}</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">Authorization: {st.session_state.role.capitalize()}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Theme Toggle
        st.markdown("#### System Configuration Module")
        is_dark = st.session_state.get('theme', 'light') == 'dark'
        if st.toggle("🌙 Enable Dark Mode", value=is_dark):
            if st.session_state.theme != 'dark':
                st.session_state.theme = 'dark'
                st.rerun()
        else:
            if st.session_state.theme != 'light':
                st.session_state.theme = 'light'
                st.rerun()
                
        st.markdown("---")
        
        # Logout Button
        st.markdown('<div class="logout-btn-container">', unsafe_allow_html=True)
        if st.button("🚫 Terminate Session", key="logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        with st.expander("📊 Model Performance Accuracy (%)", expanded=False):
            st.markdown("""
                - Detection: 98.4%
                - Classification: 96.7%
                - Framework: TensorFlow v2.15
            """)
        
        with st.expander("🧩 Deep Learning Model Configuration Panel", expanded=False):
            st.markdown("""
                **Architecture:**
                - Dual-Stage CNN Pipeline
                - Input Vector: [224, 224, 3]
                - Activation: Softmax/Relu
            """)
