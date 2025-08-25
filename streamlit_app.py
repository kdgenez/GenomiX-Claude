import streamlit as st
import os
from src.genomix_agent import GenomiXAgent
from src.knowledge_base import BiologyKnowledgeBase
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="🧬 GenomiX - Agente Inteligente de Biología",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con paleta de colores GenomiX
st.markdown("""
<style>
    /* Paleta de colores GenomiX */
    :root {
        --azul-profundo: #1B365D;
        --cian-brillante: #00C2D1;
        --verde-esmeralda: #2ECC71;
        --gris-neutro: #4D4D4D;
    }
    
    /* Header principal */
    .main-header {
        font-size: 3.5rem;
        color: var(--azul-profundo);
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: var(--cian-brillante);
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .slogan {
        font-size: 1.1rem;
        color: var(--gris-neutro);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Mensajes del chat */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
    }
    
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left-color: var(--azul-profundo);
        box-shadow: 0 2px 10px rgba(27, 54, 93, 0.1);
    }
    
    .agent-message {
        background: linear-gradient(135deg, #E8F8F5 0%, #D5F4E6 100%);
        border-left-color: var(--verde-esmeralda);
        box-shadow: 0 2px 10px rgba(46, 204, 113, 0.1);
    }
    
    /* Cajas de estadísticas */
    .stats-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid var(--cian-brillante);
        transition: transform 0.3s ease;
    }
    
    .stats-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0, 194, 209, 0.2);
    }
    
    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(135deg, var(--azul-profundo) 0%, var(--cian-brillante) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 194, 209, 0.4);
    }
    
    /* API Key input styling */
    .api-key-container {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFF3CD 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FFE066;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: var(--azul-profundo);
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--gris-neutro);
        padding: 2rem;
        border-top: 1px solid #E0E0E0;
        margin-top: 3rem;
    }
    
    /* DNA Helix Animation */
    @keyframes dna-rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .dna-icon {
        animation: dna-rotate 4s linear infinite;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def validate_groq_api_key(api_key: str) -> bool:
    """Validar la API key de Groq"""
    if not api_key or len(api_key) < 10:
        return False
    
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=10
        )
        # Test simple
        response = llm.invoke([{"role": "user", "content": "Hi"}])
        return True
    except Exception as e:
        st.error(f"Error validando API Key: {str(e)}")
        return False

@st.cache_resource
def initialize_agent(api_key: str):
    """Inicializar el agente GenomiX (cached para mejor rendimiento)"""
    try:
        knowledge_base = BiologyKnowledgeBase()
        agent = GenomiXAgent(knowledge_base, api_key)
        return agent
    except Exception as e:
        st.error(f"Error al inicializar GenomiX: {str(e)}")
        return None

def display_api_key_setup():
    """Mostrar configuración de API Key"""
    st.markdown("""
    <div class="api-key-container">
        <h3>🔑 Configuración de API Key</h3>
        <p>Para usar GenomiX, necesitas una API key gratuita de Groq:</p>
        <ol>
            <li>Visita <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
            <li>Regístrate o inicia sesión</li>
            <li>Genera una nueva API key</li>
            <li>Copia y pega la clave aquí abajo</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input(
        "Ingresa tu API Key de Groq:",
        type="password",
        placeholder="gsk_...",
        help="Tu API key se mantiene segura y solo se usa durante esta sesión"
    )
    
    if api_key:
        with st.spinner("🔍 Validando API Key..."):
            if validate_groq_api_key(api_key):
                st.success("✅ API Key válida! GenomiX está listo para usar.")
                st.session_state.groq_api_key = api_key
                st.session_state.api_key_valid = True
                st.rerun()
            else:
                st.error("❌ API Key inválida. Por favor verifica e intenta nuevamente.")
    
    return api_key if api_key else None

def display_chat_history():
    """Mostrar el historial de chat con estilo GenomiX"""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for i, (user_msg, agent_msg, timestamp) in enumerate(st.session_state.chat_history):
            # Mensaje del usuario
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 Usuario ({timestamp}):</strong><br>
                {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            # Respuesta de GenomiX
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong><span class="dna-icon">🧬</span> GenomiX:</strong><br>
                {agent_msg}
            </div>
            """, unsafe_allow_html=True)

def create_genomix_dashboard():
    """Crear dashboard visual para GenomiX"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-box">
            <h2>🔬</h2>
            <h4>Conceptos</h4>
            <p>Biológicos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-box">
            <h2>🌿</h2>
            <h4>Identificación</h4>
            <p>de Especies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-box">
            <h2>🧬</h2>
            <h4>Procesos</h4>
            <p>Genómicos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-box">
            <h2>🌍</h2>
            <h4>Ecología</h4>
            <p>y Evolución</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header principal con identidad GenomiX
    st.markdown('<h1 class="main-header"><span class="dna-icon">🧬</span> GenomiX</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Agente Inteligente de Biología</p>', unsafe_allow_html=True)
    st.markdown('<p class="slogan"><em>"Descifrando la vida, gen por gen"</em></p>', unsafe_allow_html=True)
    
    # Verificar si hay API key válida
    if not st.session_state.get("api_key_valid", False):
        display_api_key_setup()
        return
    
    # Inicializar el agente
    agent = initialize_agent(st.session_state.groq_api_key)
    
    if agent is None:
        st.error("❌ No se pudo inicializar GenomiX. Verifica tu API key.")
        if st.button("🔄 Reintentar"):
            st.session_state.api_key_valid = False
            st.rerun()
        return
    
    # Dashboard visual
    create_genomix_dashboard()
    
    # Sidebar con información de GenomiX
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">ℹ️ Acerca de GenomiX</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **GenomiX** es tu compañero inteligente para explorar el fascinante mundo de la biología. 
        Con rigor académico y tecnología de vanguardia, GenomiX te ayuda a:
        """)
        
        st.markdown("### 🎯 Capacidades Principales")
        capabilities = [
            "🔬 **Conceptos Biológicos**: Explicaciones claras de procesos complejos",
            "🌿 **Identificación de Especies**: Análisis detallado por características",
            "🧬 **Genómica**: Procesos genéticos y moleculares",
            "📊 **Taxonomía**: Clasificación y jerarquías completas",
            "🌍 **Ecología**: Relaciones ecosistémicas y biodiversidad",
            "🔬 **Bioquímica**: Procesos metabólicos y enzimáticos"
        ]
        
        for capability in capabilities:
            st.markdown(f"• {capability}")
        
        st.markdown("---")
        
        st.markdown("### 💡 Preguntas de Ejemplo")
        example_questions = [
            "¿Cómo funciona la fotosíntesis a nivel molecular?",
            "Identifica: animal pequeño, peludo, cola larga, vive en árboles",
            "Explica el proceso de mitosis paso a paso",
            "¿Cuál es la diferencia entre ADN y ARN?",
            "¿Qué es CRISPR y cómo funciona?",
            "Describe la clasificación taxonómica del ser humano"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📝 {question[:30]}...", key=f"example_{i}", help=question):
                st.session_state.user_input = question
        
        st.markdown("---")
        
        # Información técnica
        st.markdown("### ⚙️ Información Técnica")
        st.info(f"""
        **Modelo**: Mixtral-8x7B (Groq)
        **Framework**: LangChain
        **Conocimiento**: Base vectorial especializada
        **Estado**: ✅ Operativo
        """)
    
    # Inicializar session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Área principal de chat
    st.markdown("---")
    st.header("💬 Chat con GenomiX")
    
    # Mostrar historial de chat
    display_chat_history()
    
    # Input del usuario
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Haz tu consulta biológica a GenomiX:",
            value=st.session_state.get("user_input", ""),
            key="current_input",
            placeholder="Ejemplo: ¿Puedes explicar cómo funciona la respiración celular?"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Espaciado
        send_button = st.button("🚀 Consultar", type="primary", use_container_width=True)
    
    # Botones adicionales
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        clear_button = st.button("🗑️ Limpiar Chat", use_container_width=True)
    
    with col2:
        if st.button("🔄 Nueva Sesión", use_container_width=True):
            st.session_state.api_key_valid = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Procesar input del usuario
    if send_button and user_input.strip():
        with st.spinner("🧬 GenomiX está analizando tu consulta..."):
            try:
                # Obtener respuesta del agente
                start_time = time.time()
                response = agent.process_query(user_input)
                response_time = time.time() - start_time
                
                # Formatear timestamp
                timestamp = time.strftime("%H:%M:%S")
                
                # Agregar al historial
                st.session_state.chat_history.append(
                    (user_input, response, timestamp)
                )
                
                # Limpiar input
                st.session_state.user_input = ""
                
                # Mostrar métricas de respuesta
                st.success(f"✅ Análisis completado en {response_time:.2f} segundos")
                
                # Rerun para actualizar la UI
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error procesando consulta: {str(e)}")
                st.info("💡 Intenta reformular tu pregunta o verifica tu conexión a internet.")
    
    # Limpiar historial
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.success("🗑️ Chat limpiado correctamente")
        st.rerun()
    
    # Footer con información de GenomiX
    st.markdown("""
    <div class="footer">
        <h3>🧬 GenomiX - Donde la biología se encuentra con la inteligencia</h3>
        <p><strong>Desarrollado con ❤️ usando:</strong> LangChain • Groq API • Streamlit • FAISS</p>
        <p><em>"El conocimiento biológico, amplificado por IA"</em></p>
        <br>
        <p style="font-size: 0.9rem; color: #999;">
            🔬 Rigor Académico • 🚀 Innovación Tecnológica • 📚 Didáctica Clara
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
