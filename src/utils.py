"""
Utilidades y funciones de apoyo para GenomiX
Agente inteligente especializado en Biología
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import unicodedata
from datetime import datetime

logger = logging.getLogger(__name__)

def format_species_info(species: Dict[str, Any]) -> str:
    """
    Formatear información de especies con estilo GenomiX
    
    Args:
        species: Diccionario con datos de la especie
        
    Returns:
        Texto formateado con información de la especie
    """
    try:
        # Información básica
        name = species.get('name', 'Especie no identificada')
        scientific_name = species.get('scientific_name', 'N/A')
        confidence = species.get('confidence', 'N/A')
        
        # Taxonomía
        kingdom = species.get('kingdom', 'N/A')
        phylum = species.get('phylum', 'N/A')
        class_name = species.get('class', 'N/A')
        order = species.get('order', 'N/A')
        family = species.get('family', 'N/A')
        
        # Características
        characteristics = species.get('characteristics', 'Sin descripción disponible')
        habitat = species.get('habitat', 'Hábitat no especificado')
        behavior = species.get('behavior', 'Comportamiento no documentado')
        diet = species.get('diet', 'Dieta no especificada')
        size = species.get('size', 'Tamaño no especificado')
        
        # Formatear con estilo GenomiX
        formatted_info = f"""
🧬 **{name}** (*{scientific_name}*)
{'─' * 50}

**📊 Clasificación Taxonómica**
├─ Reino: {kingdom}
├─ Filo: {phylum}  
├─ Clase: {class_name}
├─ Orden: {order}
└─ Familia: {family}

**🔬 Características Morfológicas**
{characteristics}

**🌍 Hábitat y Distribución**
{habitat}

**⚡ Comportamiento**
{behavior}

**🍽️ Dieta**
{diet}

**📏 Dimensiones**
{size}

**💡 Confianza del Análisis GenomiX:** {confidence}%
"""
        
        return formatted_info.strip()
        
    except Exception as e:
        logger.error(f"Error formateando información de especie: {e}")
        return f"Error procesando información de {species.get('name', 'especie desconocida')}"

def extract_biological_concepts(text: str) -> List[str]:
    """
    Extraer conceptos biológicos clave de un texto
    
    Args:
        text: Texto del cual extraer conceptos
        
    Returns:
        Lista de conceptos biológicos identificados
    """
    # Diccionario de conceptos biológicos con patrones
    biological_patterns = {
        # Procesos celulares
        r'\b(?:mitosis|meiosis|citocinesis|apoptosis)\b': 'División Celular',
        r'\b(?:fotosíntesis|respiración celular|glucólisis)\b': 'Metabolismo',
        r'\b(?:transcripción|traducción|replicación)\b': 'Expresión Génica',
        
        # Moléculas biológicas
        r'\b(?:ADN|ARN|proteína|enzima|ATP)\b': 'Moléculas Biológicas',
        r'\b(?:carbohidrato|lípido|aminoácido)\b': 'Biomoléculas',
        
        # Genética
        r'\b(?:gen|genoma|cromosoma|alelo|mutación)\b': 'Genética',
        r'\b(?:herencia|dominante|recesivo|fenotipo|genotipo)\b': 'Herencia',
        
        # Evolución
        r'\b(?:evolución|selección natural|adaptación|especiación)\b': 'Evolución',
        r'\b(?:ancestro común|filogenia|deriva genética)\b': 'Filogenia',
        
        # Ecología
        r'\b(?:ecosistema|biodiversidad|nicho ecológico)\b': 'Ecología',
        r'\b(?:cadena alimentaria|productor|consumidor|descomponedor)\b': 'Redes Tróficas',
        
        # Anatomía y fisiología
        r'\b(?:sistema nervioso|sistema circulatorio|homeostasis)\b': 'Fisiología',
        r'\b(?:tejido|órgano|célula|organelo)\b': 'Anatomía',
        
        # Taxonomía
        r'\b(?:reino|filo|clase|orden|familia|género|especie)\b': 'Taxonomía',
        r'\b(?:clasificación|nomenclatura binomial)\b': 'Sistemática'
    }
    
    concepts_found = []
    text_lower = text.lower()
    
    for pattern, category in biological_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            concepts_found.extend([(match, category) for match in matches])
    
    # Remover duplicados manteniendo orden
    unique_concepts = []
    seen = set()
    
    for concept, category in concepts_found:
        if concept not in seen:
            unique_concepts.append(f"{concept.capitalize()} ({category})")
            seen.add(concept)
    
    return unique_concepts

def clean_biological_text(text: str) -> str:
    """
    Limpiar y normalizar texto biológico
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio y normalizado
    """
    if not text:
        return ""
    
    # Normalizar unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remover caracteres de control
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    
    # Limpiar caracteres especiales pero mantener científicos
    text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]°′″αβγδεμπ]', '', text)
    
    return text.strip()

def validate_species_data(species_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validar estructura de datos de especies
    
    Args:
        species_data: Diccionario con datos de especie
        
    Returns:
        Tupla (es_válido, lista_de_errores)
    """
    errors = []
    
    # Campos obligatorios
    required_fields = ['name', 'scientific_name', 'family']
    
    for field in required_fields:
        if not species_data.get(field):
            errors.append(f"Campo obligatorio faltante: {field}")
    
    # Validar nombre científico (formato binomial básico)
    scientific_name = species_data.get('scientific_name', '')
    if scientific_name and not re.match(r'^[A-Z][a-z]+ [a-z]+', scientific_name):
        errors.append("Nombre científico no sigue nomenclatura binomial")
    
    # Validar taxonomía
    taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family']
    for rank in taxonomic_ranks:
        value = species_data.get(rank, '')
        if value and not isinstance(value, str):
            errors.append(f"Rango taxonómico {rank} debe ser texto")
    
    return len(errors) == 0, errors

def generate_species_summary(species: Dict[str, Any]) -> str:
    """
    Generar resumen conciso de una especie
    
    Args:
        species: Datos de la especie
        
    Returns:
        Resumen formateado
    """
    name = species.get('name', 'Especie no identificada')
    scientific_name = species.get('scientific_name', '')
    family = species.get('family', 'Familia desconocida')
    characteristics = species.get('characteristics', '')[:100] + "..." if len(species.get('characteristics', '')) > 100 else species.get('characteristics', '')
    
    summary = f"**{name}**"
    if scientific_name:
        summary += f" (*{scientific_name}*)"
    
    summary += f"\n📚 {family}"
    
    if characteristics:
        summary += f"\n🔍 {characteristics}"
    
    return summary

def format_concept_explanation(concept: Dict[str, Any]) -> str:
    """
    Formatear explicación de concepto biológico
    
    Args:
        concept: Datos del concepto
        
    Returns:
        Explicación formateada con estilo GenomiX
    """
    try:
        name = concept.get('name', 'Concepto sin nombre')
        category = concept.get('category', 'Sin categoría')
        definition = concept.get('definition', 'Sin definición')
        description = concept.get('description', '')
        importance = concept.get('importance', '')
        examples = concept.get('examples', [])
        related_concepts = concept.get('related_concepts', [])
        
        formatted_explanation = f"""
🧬 **{name}**
📂 *Categoría: {category}*
{'─' * 40}

**🔬 Definición Científica**
{definition}

"""
        
        if description:
            formatted_explanation += f"""**⚡ Descripción Detallada**
{description}

"""
        
        if importance:
            formatted_explanation += f"""**🎯 Importancia Biológica**
{importance}

"""
        
        if examples:
            formatted_explanation += f"""**📊 Ejemplos Prácticos**
{chr(10).join([f"• {example}" for example in examples])}

"""
        
        if related_concepts:
            formatted_explanation += f"""**🔗 Conceptos Relacionados**
{', '.join(related_concepts)}
"""
        
        return formatted_explanation.strip()
        
    except Exception as e:
        logger.error(f"Error formateando explicación de concepto: {e}")
        return f"Error procesando concepto {concept.get('name', 'desconocido')}"

def create_taxonomic_hierarchy(species_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crear jerarquía taxonómica a partir de datos de especies
    
    Args:
        species_data: Lista de especies
        
    Returns:
        Diccionario con jerarquía taxonómica
    """
    hierarchy = {}
    
    for species in species_data:
        # Construir path taxonómico
        kingdom = species.get('kingdom', 'Unknown')
        phylum = species.get('phylum', 'Unknown')
        class_name = species.get('class', 'Unknown')
        order = species.get('order', 'Unknown')
        family = species.get('family', 'Unknown')
        
        # Crear estructura jerárquica
        if kingdom not in hierarchy:
            hierarchy[kingdom] = {}
        
        if phylum not in hierarchy[kingdom]:
            hierarchy[kingdom][phylum] = {}
        
        if class_name not in hierarchy[kingdom][phylum]:
            hierarchy[kingdom][phylum][class_name] = {}
        
        if order not in hierarchy[kingdom][phylum][class_name]:
            hierarchy[kingdom][phylum][class_name][order] = {}
        
        if family not in hierarchy[kingdom][phylum][class_name][order]:
            hierarchy[kingdom][phylum][class_name][order][family] = []
        
        # Agregar especie a la familia
        hierarchy[kingdom][phylum][class_name][order][family].append({
            'name': species.get('name', 'Sin nombre'),
            'scientific_name': species.get('scientific_name', 'Sin nombre científico')
        })
    
    return hierarchy

def generate_genomix_response_template(query_type: str) -> str:
    """
    Generar plantilla de respuesta según el tipo de consulta
    
    Args:
        query_type: Tipo de consulta ('species', 'concept', 'process', 'general')
        
    Returns:
        Plantilla de respuesta con formato GenomiX
    """
    templates = {
        'species': """
🧬 **Análisis de Especies GenomiX**

**🔍 Identificación Sistemática**
[Información de identificación]

**📊 Clasificación Taxonómica**
[Jerarquía taxonómica]

**🔬 Características Distintivas**
[Descripción morfológica y funcional]

**💡 Perspectiva GenomiX**
[Insights adicionales y conexiones]
""",
        
        'concept': """
🧬 **Explicación Conceptual GenomiX**

**🔬 Definición Científica**
[Definición precisa]

**⚙️ Mecanismo Biológico**
[Cómo funciona el proceso/concepto]

**🎯 Importancia y Aplicaciones**
[Relevancia biológica y aplicaciones]

**🔗 Conexiones Biológicas**
[Relación con otros conceptos]

**💡 Perspectiva GenomiX**
[Insights tecnológicos y futuros]
""",
        
        'process': """
🧬 **Análisis de Proceso GenomiX**

**⚡ Descripción del Proceso**
[Descripción general]

**🔄 Mecánica Molecular**
[Pasos detallados del proceso]

**📊 Regulación y Control**
[Mecanismos de control]

**🔬 Metodología de Estudio**
[Cómo se estudia este proceso]

**💡 Aplicaciones GenomiX**
[Aplicaciones biotecnológicas]
""",
        
        'general': """
🧬 **Respuesta GenomiX**

[Contenido de respuesta adaptado al contexto]

**💡 Insight GenomiX**
[Perspectiva única o conexión interdisciplinaria]

*"Descifrando la vida, gen por gen"*
"""
    }
    
    return templates.get(query_type, templates['general'])

def log_genomix_interaction(query: str, response: str, success: bool = True):
    """
    Registrar interacciones de GenomiX para análisis
    
    Args:
        query: Consulta del usuario
        response: Respuesta de GenomiX
        success: Si la interacción fue exitosa
    """
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100] + "..." if len(query) > 100 else query,
            'response_length': len(response),
            'success': success,
            'query_type': classify_query_type(query)
        }
        
        logger.info(f"GenomiX Interaction: {json.dumps(log_entry)}")
        
    except Exception as e:
        logger.error(f"Error logging GenomiX interaction: {e}")

def classify_query_type(query: str) -> str:
    """
    Clasificar tipo de consulta para optimizar respuesta
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Tipo de consulta identificado
    """
    query_lower = query.lower()
    
    # Patrones para identificación de especies
    species_indicators = [
        'identifica', 'qué animal', 'qué planta', 'describe', 'características',
        'pequeño', 'grande', 'vive en', 'hábitat', 'comportamiento'
    ]
    
    # Patrones para conceptos
    concept_indicators = [
        'qué es', 'explica', 'define', 'cómo funciona', 'proceso de',
        'fotosíntesis', 'respiración', 'mitosis', 'evolución', 'gen'
    ]
    
    # Patrones para procesos
    process_indicators = [
        'paso a paso', 'etapas', 'fases', 'mecanismo', 'ciclo',
        'replicación', 'transcripción', 'traducción'
    ]
    
    # Patrones para taxonomía
    taxonomy_indicators = [
        'clasificación', 'taxonomía', 'reino', 'filo', 'familia',
        'nombre científico', 'sistemática'
    ]
    
    if any(indicator in query_lower for indicator in species_indicators):
        return 'species'
    elif any(indicator in query_lower for indicator in process_indicators):
        return 'process'
    elif any(indicator in query_lower for indicator in taxonomy_indicators):
        return 'taxonomy'
    elif any(indicator in query_lower for indicator in concept_indicators):
        return 'concept'
    else:
        return 'general'

def format_error_message(error: Exception, context: str = "") -> str:
    """
    Formatear mensajes de error con estilo GenomiX
    
    Args:
        error: Excepción capturada
        context: Contexto del error
        
    Returns:
        Mensaje de error formateado
    """
    error_messages = {
        "connection": """
🔬 **GenomiX - Problema de Conexión**

Los sistemas GenomiX están experimentando dificultades de conectividad. Como un organismo adaptándose a condiciones adversas, estamos reconfigurando nuestros sistemas.

💡 **Mientras tanto, puedes:**
• Verificar tu conexión a internet
• Intentar tu consulta nuevamente en unos momentos
• Reformular tu pregunta de manera más específica

*Los sistemas biológicos también enfrentan disrupciones, pero siempre encuentran formas de adaptarse.*
""",
        
        "api_key": """
🔬 **GenomiX - Configuración de API**

Como un organismo necesita nutrientes para funcionar, GenomiX requiere una API key válida para acceder a sus sistemas avanzados.

💡 **Solución:**
• Obtén una API key gratuita en console.groq.com
• Verifica que la clave esté correctamente ingresada
• Asegúrate de tener conexión a internet

*La precisión en la configuración es tan importante como la precisión en la ciencia.*
""",
        
        "general": f"""
🔬 **GenomiX - Análisis Temporal No Disponible**

Los sistemas GenomiX han encontrado una situación inesperada durante el análisis{': ' + context if context else ''}.

💡 **Como un científico experimentado, GenomiX sugiere:**
• Reformular la consulta con términos más específicos
• Proporcionar más contexto biológico
• Intentar la consulta nuevamente

*En la ciencia, cada obstáculo es una oportunidad para refinar nuestro enfoque.*

Error técnico: {str(error)}
"""
    }
    
    # Determinar tipo de error
    if "connection" in str(error).lower() or "network" in str(error).lower():
        return error_messages["connection"]
    elif "api" in str(error).lower() or "key" in str(error).lower():
        return error_messages["api_key"]
    else:
        return error_messages["general"]
