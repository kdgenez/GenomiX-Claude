import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain_groq import ChatGroq
from .knowledge_base import BiologyKnowledgeBase
from .prompts import GENOMIX_SYSTEM_PROMPT, SPECIES_IDENTIFICATION_PROMPT, GENOMIX_PERSONALITY
from .utils import format_species_info, extract_biological_concepts
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenomiXAgent:
    """
    GenomiX - Agente inteligente especializado en Biología
    "Descifrando la vida, gen por gen"
    
    Características de personalidad:
    - Académico y confiable (rigor científico)
    - Innovador y visionario (toque tecnológico)
    - Claro y didáctico (explica conceptos complejos de manera comprensible)
    """
    
    def __init__(self, knowledge_base: BiologyKnowledgeBase, groq_api_key: str):
        """
        Inicializar el agente GenomiX
        
        Args:
            knowledge_base: Base de conocimiento biológico
            groq_api_key: API key de Groq proporcionada por el usuario
        """
        self.knowledge_base = knowledge_base
        self.groq_api_key = groq_api_key
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.tools = self._create_tools()
        self.agent = self._initialize_agent()
        
        logger.info("GenomiX inicializado correctamente - 'Descifrando la vida, gen por gen'")
    
    def _initialize_llm(self) -> ChatGroq:
        """Inicializar el modelo LLM de Groq con la API key del usuario"""
        if not self.groq_api_key:
            raise ValueError("API key de Groq requerida para inicializar GenomiX")
        
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.2,  # Temperatura moderada para balance entre precisión y creatividad
            max_tokens=3000,
            top_p=0.9,
            streaming=False
        )
    
    def _create_tools(self) -> List[Tool]:
        """Crear herramientas especializadas para GenomiX"""
        tools = [
            Tool(
                name="genomix_species_analyzer",
                description="Herramienta avanzada de GenomiX para identificar y analizar especies. "
                          "Usa análisis sistemático basado en características morfológicas, "
                          "comportamentales y ecológicas. Perfecto para descripciones detalladas.",
                func=self.identify_species
            ),
            Tool(
                name="genomix_concept_explainer", 
                description="Explicador de conceptos biológicos de GenomiX. Desglosa procesos "
                          "complejos como fotosíntesis, respiración celular, genética, evolución "
                          "usando analogías tecnológicas y ejemplos claros.",
                func=self.explain_concept
            ),
            Tool(
                name="genomix_molecular_analyzer",
                description="Analizador molecular de GenomiX. Especializado en procesos a nivel "
                          "celular y molecular: ADN, ARN, proteínas, enzimas, metabolismo.",
                func=self.analyze_molecular_process
            ),
            Tool(
                name="genomix_taxonomy_classifier",
                description="Clasificador taxonómico de GenomiX. Proporciona jerarquías "
                          "taxonómicas completas y relaciones evolutivas usando principios "
                          "de sistemática moderna.",
                func=self.get_taxonomy_info
            ),
            Tool(
                name="genomix_ecology_consultant",
                description="Consultor ecológico de GenomiX. Analiza ecosistemas, cadenas "
                          "alimentarias, relaciones interespecíficas y conceptos de biodiversidad "
                          "con enfoque de sistemas complejos.",
                func=self.explain_ecology
            ),
            Tool(
                name="genomix_genomics_expert",
                description="Experto en genómica de GenomiX. Especializado en genética moderna, "
                          "biotecnología, CRISPR, secuenciación genética y bioinformática.",
                func=self.explain_genomics
            )
        ]
        return tools
    
    def _initialize_agent(self):
        """Inicializar el agente con personalidad GenomiX"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
            agent_kwargs={
                "system_message": GENOMIX_SYSTEM_PROMPT,
                "human_message": "Usuario: {input}\n\nGenomiX:",
                "ai_message": "Perfecto. Como GenomiX, analicemos esta fascinante consulta biológica."
            }
        )
    
    def identify_species(self, description: str) -> str:
        """
        Identificar especies usando el enfoque sistemático de GenomiX
        
        Args:
            description: Descripción de las características del organismo
            
        Returns:
            Análisis detallado de posibles especies con enfoque GenomiX
        """
        try:
            # Buscar en base de conocimiento
            relevant_species = self.knowledge_base.search_species(description)
            
            if not relevant_species:
                return self._generate_genomix_species_analysis(description)
            
            # Formatear con estilo GenomiX
            result = """🧬 **Análisis GenomiX de Especies**

Basándome en tu descripción, he procesado las características mediante algoritmos de clasificación sistemática. Aquí están las especies más probables:

"""
            
            for i, species in enumerate(relevant_species[:3], 1):
                result += f"""**{i}. {species.get('name', 'Especie no identificada')}**
├─ *Nombre científico*: {species.get('scientific_name', 'Por determinar')}
├─ *Clasificación*: {species.get('family', 'Familia no especificada')}
├─ *Nicho ecológico*: {species.get('habitat', 'Hábitat variable')}
├─ *Características distintivas*: {species.get('characteristics', 'En análisis')}
└─ *Confianza del análisis*: {species.get('confidence', 85)}%

"""
            
            result += "\n💡 **Recomendación GenomiX**: Para mayor precisión, proporciona características adicionales como tamaño, coloración, comportamiento o ubicación geográfica."
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis GenomiX de especies: {e}")
            return f"🔬 Análisis temporal no disponible. Sistema GenomiX en reconfiguración: {str(e)}"
    
    def _generate_genomix_species_analysis(self, description: str) -> str:
        """Generar análisis de especies usando el LLM con personalidad GenomiX"""
        prompt = f"""{SPECIES_IDENTIFICATION_PROMPT}

Descripción del organismo: {description}

Como GenomiX, proporciona un análisis sistemático y tecnológicamente informado."""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return f"🧬 **Análisis GenomiX**\n\n{response.content}"
        except Exception as e:
            logger.error(f"Error generando análisis GenomiX: {e}")
            return "🔬 Análisis no disponible temporalmente. Reiniciando sistemas GenomiX..."
    
    def explain_concept(self, concept: str) -> str:
        """
        Explicar conceptos biológicos con el enfoque didáctico de GenomiX
        
        Args:
            concept: Concepto biológico a explicar
            
        Returns:
            Explicación detallada con analogías y enfoque GenomiX
        """
        try:
            # Buscar información en base de conocimiento
            concept_info = self.knowledge_base.search_concepts(concept)
            
            if concept_info:
                info = concept_info[0]
                explanation = f"""🧬 **GenomiX Explica: {concept.title()}**

**🔬 Definición Científica**
{info.get('definition', 'Concepto en análisis por sistemas GenomiX')}

**⚡ ¿Por qué es importante?**
{info.get('importance', 'Fundamental para comprender la complejidad biológica')}

**🔧 Analogía Tecnológica**
Si pensamos en este proceso como un sistema computacional...
"""
                
                if info.get('examples'):
                    explanation += f"""
**📊 Ejemplos Prácticos**
• {' • '.join(info['examples'])}
"""
                
                if info.get('related_concepts'):
                    explanation += f"""
**🔗 Conexiones en la Red Biológica**
Conceptos relacionados: {', '.join(info['related_concepts'])}
"""
                
                return explanation
            else:
                # Generar explicación usando LLM con personalidad GenomiX
                prompt = f"""Como GenomiX, explica el concepto biológico: {concept}

Estructura tu respuesta usando mi personalidad académica pero didáctica:

1. **🔬 Esencia del Concepto**: Definición precisa
2. **⚙️ Mecanismo**: Cómo funciona (usa analogías tecnológicas)
3. **🎯 Relevancia Biológica**: Por qué es crucial
4. **🔗 Conexiones**: Relación con otros procesos
5. **💡 Perspectiva GenomiX**: Insight único o aplicación moderna

Mantén el rigor científico pero hazlo accesible. Usa analogías con sistemas, redes o tecnología cuando sea apropiado."""

                response = self.llm.invoke([{"role": "user", "content": prompt}])
                return response.content
                
        except Exception as e:
            logger.error(f"Error explicando concepto: {e}")
            return f"🔬 Sistemas GenomiX temporalmente no disponibles para análisis conceptual: {str(e)}"
    
    def analyze_molecular_process(self, process: str) -> str:
        """
        Analizar procesos moleculares con enfoque GenomiX
        
        Args:
            process: Proceso molecular/celular a analizar
            
        Returns:
            Análisis detallado del proceso molecular
        """
        try:
            prompt = f"""Como GenomiX, analiza el proceso molecular/celular: {process}

Estructura el análisis así:
🧬 **Análisis Molecular GenomiX**

**⚛️ Nivel Molecular**
- Componentes moleculares clave
- Interacciones químicas específicas

**🔄 Mecánica del Proceso**
- Pasos secuenciales detallados
- Regulación y control

**⚡ Energética**
- Requerimientos energéticos
- Eficiencia del proceso

**🔬 Tecnología Actual**
- Cómo estudiamos este proceso
- Aplicaciones biotecnológicas

**🚀 Perspectiva Futura**
- Implicaciones para biotecnología/medicina

Usa terminología precisa pero con explicaciones claras."""

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            logger.error(f"Error analizando proceso molecular: {e}")
            return f"🔬 Análisis molecular GenomiX no disponible: {str(e)}"
    
    def get_taxonomy_info(self, organism: str) -> str:
        """
        Obtener información taxonómica con sistemática moderna GenomiX
        
        Args:
            organism: Organismo del cual obtener información taxonómica
            
        Returns:
            Clasificación taxonómica detallada con enfoque GenomiX
        """
        try:
            prompt = f"""Como GenomiX, proporciona la clasificación taxonómica de: {organism}

Formato de respuesta:
🧬 **Clasificación Sistemática GenomiX**

**📊 Jerarquía Taxonómica**
```
Reino (Kingdom): 
Filo/División (Phylum/Division): 
Clase (Class): 
Orden (Order): 
Familia (Family): 
Género (Genus): 
Especie (Species): 
```

**🔬 Características Distintivas por Nivel**
- Reino: [características fundamentales]
- Filo: [características del filo]
- Clase: [características de clase]
- etc.

**🧬 Información Genómica**
- Tamaño aproximado del genoma
- Características genómicas únicas

**🌐 Relaciones Evolutivas**
- Grupos hermanos
- Divergencias evolutivas importantes

**💡 Datos GenomiX**
- Curiosidades taxonómicas
- Revisiones sistemáticas recientes"""

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            logger.error(f"Error obteniendo taxonomía: {e}")
            return f"🔬 Base de datos taxonómicos GenomiX temporalmente no disponible: {str(e)}"
    
    def explain_ecology(self, topic: str) -> str:
        """
        Explicar conceptos ecológicos con enfoque de sistemas GenomiX
        
        Args:
            topic: Tópico ecológico a explicar
            
        Returns:
            Explicación ecológica con perspectiva de sistemas complejos
        """
        try:
            prompt = f"""Como GenomiX, explica el concepto ecológico: {topic}

Estructura con enfoque de sistemas:
🌍 **Análisis Ecológico GenomiX**

**🔗 Arquitectura del Sistema**
- Componentes principales
- Flujos de información/energía

**⚙️ Dinámicas de Interacción**
- Retroalimentaciones
- Puntos de equilibrio

**📊 Métricas del Ecosistema**
- Indicadores clave
- Métodos de medición

**🔬 Herramientas de Análisis**
- Tecnologías de monitoreo
- Modelado computacional

**🚀 Perspectiva GenomiX**
- Aplicación de big data
- Genómica ambiental
- Predicciones futuras

Conecta conceptos ecológicos clásicos con tecnología moderna."""

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            logger.error(f"Error explicando ecología: {e}")
            return f"🌍 Sistemas ecológicos GenomiX en mantenimiento: {str(e)}"
    
    def explain_genomics(self, topic: str) -> str:
        """
        Explicar temas de genómica y biotecnología moderna
        
        Args:
            topic: Tópico de genómica a explicar
            
        Returns:
            Explicación especializada en genómica
        """
        try:
            prompt = f"""Como GenomiX, especialista en genómica, explica: {topic}

Estructura especializada:
🧬 **Informe Genómico GenomiX**

**🔬 Fundamento Molecular**
- Base genética/molecular
- Mecanismos moleculares

**⚡ Tecnología Actual**
- Herramientas y técnicas
- Plataformas tecnológicas

**📊 Datos y Análisis**
- Tipos de datos generados
- Métodos de análisis

**🚀 Aplicaciones Prácticas**
- Medicina personalizada
- Biotecnología
- Agricultura

**🔮 Futuro de la Genómica**
- Tendencias emergentes
- Desafíos técnicos
- Implicaciones éticas

**💡 Perspectiva GenomiX**
- Insights únicos
- Conexiones interdisciplinarias

Mantén alta precisión técnica con claridad explicativa."""

            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content
            
        except Exception as e:
            logger.error(f"Error explicando genómica: {e}")
            return f"🧬 Sistemas genómicos GenomiX en actualización: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Procesar consulta del usuario con personalidad GenomiX
        
        Args:
            query: Pregunta o consulta del usuario
            
        Returns:
            Respuesta de GenomiX con su personalidad característica
        """
        try:
            logger.info(f"GenomiX procesando consulta: {query[:50]}...")
            
            # Agregar contexto de personalidad GenomiX a la consulta
            contextualized_query = f"""Como GenomiX, el agente inteligente de biología que combina rigor académico con claridad didáctica y perspectiva tecnológica, responde a esta consulta:

{query}

Recuerda mi personalidad:
- Académico y confiable (rigor científico)
- Innovador y visionario (toque tecnológico)  
- Claro y didáctico (analogías comprensibles)
- Slogan: "Descifrando la vida, gen por gen"

Usa herramientas apropiadas y mantén el tono GenomiX."""
            
            # Ejecutar agente
            result = self.agent.run(input=contextualized_query)
            
            # Limpiar y formatear respuesta
            if isinstance(result, dict) and "output" in result:
                response = result["output"]
            else:
                response = str(result)
            
            # Asegurar que la respuesta tenga el tono GenomiX
            if not any(marker in response.lower() for marker in ['genomix', '🧬', '🔬']):
                response = f"🧬 **GenomiX responde:**\n\n{response}\n\n*Descifrando la vida, gen por gen.*"
            
            logger.info("Consulta procesada exitosamente por GenomiX")
            return response
            
        except Exception as e:
            logger.error(f"Error procesando consulta GenomiX: {e}")
            
            # Respuesta de respaldo con personalidad GenomiX
            try:
                backup_prompt = f"""Soy GenomiX, tu agente inteligente de biología. Mi misión es descifrar la vida, gen por gen.

Consulta del usuario: {query}

Como experto en biología con enfoque académico pero didáctico, proporciono una respuesta clara y científicamente rigurosa. Uso analogías tecnológicas cuando es apropiado y mantengo mi perspectiva innovadora.

Respuesta GenomiX:"""
                
                backup_response = self.llm.invoke([{"role": "user", "content": backup_prompt}])
                return f"🧬 **GenomiX (Modo Directo):**\n\n{backup_response.content}\n\n*Sistemas GenomiX temporalmente en configuración básica*"
            except:
                return """🔬 **GenomiX - Sistema en Mantenimiento**

Mis sistemas avanzados están temporalmente no disponibles. Sin embargo, como GenomiX, puedo decirte que la biología es un campo fascinante donde cada proceso, desde la replicación del ADN hasta las complejas redes ecológicas, representa una maravillosa ingeniería molecular.

*Descifrando la vida, gen por gen.*

Por favor, intenta tu consulta nuevamente en unos momentos."""
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Obtener historial de conversación GenomiX"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Reiniciar memoria de GenomiX"""
        self.memory.clear()
        logger.info("Memoria GenomiX reiniciada - Listo para nuevas consultas biológicas")
