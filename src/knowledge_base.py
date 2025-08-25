import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class BiologyKnowledgeBase:
    """
    Base de conocimiento biológico especializada para GenomiX
    Utiliza embeddings vectoriales para búsqueda semántica avanzada
    """
    
    # Configuración de categorías de datos
    DATA_CATEGORIES = {
        'species': {
            'file': 'species_data.json',
            'default_data': None,
            'index': None,
            'embeddings': None
        },
        'concepts': {
            'file': 'biology_concepts.json', 
            'default_data': None,
            'index': None,
            'embeddings': None
        },
        'processes': {
            'file': 'biological_processes.json',
            'default_data': None,
            'index': None,
            'embeddings': None
        }
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.model = None
        self.data = {category: [] for category in self.DATA_CATEGORIES}
        
        # Configurar datos por defecto
        self.DATA_CATEGORIES['species']['default_data'] = self._get_default_species_data
        self.DATA_CATEGORIES['concepts']['default_data'] = self._get_default_concepts_data
        self.DATA_CATEGORIES['processes']['default_data'] = self._get_default_processes_data
        
        try:
            self._initialize_knowledge_base()
            logger.info("Base de conocimiento GenomiX inicializada exitosamente")
        except Exception as e:
            logger.error(f"Error inicializando base de conocimiento: {e}")
            self._create_fallback_mode()
    
    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """Cargar modelo de embeddings"""
        models_to_try = ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2']
        
        for model_name in models_to_try:
            try:
                model = SentenceTransformer(model_name)
                logger.info(f"Modelo de embeddings '{model_name}' cargado exitosamente")
                return model
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo '{model_name}': {e}")
        
        logger.error("No se pudo cargar ningún modelo de embeddings")
        return None
    
    def _initialize_knowledge_base(self):
        """Inicializar y cargar todos los datos"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_embedding_model()
        
        # Cargar datos para todas las categorías
        for category in self.DATA_CATEGORIES:
            self._load_data(category)
        
        # Crear índices si tenemos modelo
        if self.model:
            self._create_vector_indices()
    
    def _create_fallback_mode(self):
        """Crear modo fallback cuando no se pueden cargar embeddings"""
        logger.info("Iniciando modo fallback sin embeddings vectoriales")
        self.model = None
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos básicos
        for category in self.DATA_CATEGORIES:
            self._load_data(category)
    
    def _load_data(self, category: str):
        """Cargar datos para una categoría específica"""
        file_name = self.DATA_CATEGORIES[category]['file']
        data_file = self.data_dir / file_name
        
        try:
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.data[category] = json.load(f)
            else:
                default_data_func = self.DATA_CATEGORIES[category]['default_data']
                self.data[category] = default_data_func()
                self._save_data(category)
            
            logger.info(f"Cargados {len(self.data[category])} {category}")
        except Exception as e:
            logger.error(f"Error cargando {category}: {e}")
            default_data_func = self.DATA_CATEGORIES[category]['default_data']
            self.data[category] = default_data_func()
    
    def _create_vector_indices(self):
        """Crear índices vectoriales FAISS para todas las categorías"""
        if not self.model:
            return
            
        for category in self.DATA_CATEGORIES:
            if not self.data[category]:
                continue
                
            try:
                # Crear textos para embedding
                format_func = getattr(self, f"_format_{category}_text")
                texts = [format_func(item) for item in self.data[category]]
                
                # Crear embeddings
                embeddings = self.model.encode(texts)
                self.DATA_CATEGORIES[category]['embeddings'] = embeddings
                
                # Crear índice FAISS
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                
                # Normalizar embeddings
                embeddings_norm = embeddings.copy()
                faiss.normalize_L2(embeddings_norm)
                index.add(embeddings_norm.astype(np.float32))
                
                self.DATA_CATEGORIES[category]['index'] = index
                logger.info(f"Índice de {category} creado exitosamente")
                
            except Exception as e:
                logger.error(f"Error creando índice para {category}: {e}")
    
    def _format_text(self, item: Dict[str, Any], fields: List[str]) -> str:
        """Formatear datos para embedding de manera genérica"""
        text_parts = []
        for field in fields:
            value = item.get(field, '')
            if isinstance(value, list):
                text_parts.extend(value)
            else:
                text_parts.append(str(value))
        return ' '.join(text_parts)
    
    def _format_species_text(self, species: Dict[str, Any]) -> str:
        """Formatear datos de especies para embedding"""
        fields = ['name', 'scientific_name', 'family', 'characteristics', 'habitat', 'behavior', 'keywords']
        return self._format_text(species, fields)
    
    def _format_concept_text(self, concept: Dict[str, Any]) -> str:
        """Formatear datos de conceptos para embedding"""
        fields = ['name', 'definition', 'description', 'importance', 'keywords', 'related_concepts']
        return self._format_text(concept, fields)
    
    def _format_process_text(self, process: Dict[str, Any]) -> str:
        """Formatear datos de procesos para embedding"""
        fields = ['name', 'description', 'mechanism', 'location', 'steps', 'keywords']
        return self._format_text(process, fields)
    
    def search(self, category: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Búsqueda genérica en una categoría
        
        Args:
            category: Tipo de datos a buscar ('species', 'concepts', 'processes')
            query: Texto de búsqueda
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de resultados
        """
        if not self.data[category]:
            return []
        
        # Intentar búsqueda vectorial primero
        if self.model and self.DATA_CATEGORIES[category]['index']:
            try:
                return self._vector_search(category, query, top_k)
            except Exception as e:
                logger.error(f"Error en búsqueda vectorial de {category}: {e}")
        
        # Fallback a búsqueda básica
        return self._basic_search(category, query, top_k)
    
    def _vector_search(self, category: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial para una categoría"""
        index = self.DATA_CATEGORIES[category]['index']
        data = self.data[category]
        
        # Crear embedding de la consulta
        query_embedding = self.model.encode([query])
        query_embedding_norm = query_embedding.copy()
        faiss.normalize_L2(query_embedding_norm)
        
        # Buscar en el índice
        scores, indices = index.search(
            query_embedding_norm.astype(np.float32), 
            min(top_k, len(data))
        )
        
        # Procesar resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(data):
                result = data[idx].copy()
                result['similarity_score'] = float(score)
                result['confidence'] = min(95, max(60, int(score * 100)))
                result['search_method'] = 'vector'
                results.append(result)
        
        return results
    
    def _basic_search(self, category: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda básica de texto para una categoría"""
        query_lower = query.lower()
        results = []
        format_func = getattr(self, f"_format_{category}_text")
        
        for item in self.data[category]:
            searchable_text = format_func(item).lower()
            score = 0
            
            # Búsqueda exacta
            if query_lower in searchable_text:
                score += 0.8
            
            # Búsqueda por palabras
            query_words = query_lower.split()
            for word in query_words:
                if word in searchable_text:
                    score += 0.2
            
            if score > 0:
                result = item.copy()
                result['similarity_score'] = score
                result['confidence'] = min(90, max(50, int(score * 80)))
                result['search_method'] = 'basic'
                results.append(result)
        
        # Ordenar y limitar resultados
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    # Métodos específicos para compatibilidad con código existente
    def search_species(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search('species', query, top_k)
    
    def search_concepts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search('concepts', query, top_k)
    
    def search_processes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search('processes', query, top_k)
    
    def add_item(self, category: str, item_data: Dict[str, Any]):
        """Agregar nuevo item a la base de conocimiento"""
        self.data[category].append(item_data)
        self._save_data(category)
        
        # Recrear índices si están disponibles
        if self.model:
            self._create_vector_indices()
        
        logger.info(f"Agregado nuevo {category}: {item_data.get('name', 'Sin nombre')}")
    
    # Métodos específicos para compatibilidad
    def add_species(self, species_data: Dict[str, Any]):
        self.add_item('species', species_data)
    
    def add_concept(self, concept_data: Dict[str, Any]):
        self.add_item('concepts', concept_data)
    
    def _save_data(self, category: str):
        """Guardar datos de una categoría en archivo JSON"""
        try:
            file_name = self.DATA_CATEGORIES[category]['file']
            with open(self.data_dir / file_name, 'w', encoding='utf-8') as f:
                json.dump(self.data[category], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de {category}: {e}")
    
    def _get_default_species_data(self) -> List[Dict[str, Any]]:
        """Datos por defecto de especies para GenomiX"""
        return [
            {
                "name": "Ardilla gris",
                "scientific_name": "Sciurus carolinensis",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Mammalia",
                "order": "Rodentia",
                "family": "Sciuridae",
                "characteristics": "Pelaje gris, cola espesa y peluda, ojos grandes, dientes incisivos prominentes",
                "habitat": "Bosques templados, parques urbanos, áreas arboladas",
                "behavior": "Diurno, arborícola, almacena nueces para el invierno, territorial",
                "diet": "Omnívoro: nueces, semillas, frutas, ocasionalmente insectos",
                "size": "20-30 cm cuerpo, cola 15-25 cm",
                "keywords": ["mamífero", "roedor", "arborícola", "peludo", "cola larga", "árbol"]
            },
            # ... (otros datos de especies)
        ]
    
    def _get_default_concepts_data(self) -> List[Dict[str, Any]]:
        """Datos por defecto de conceptos biológicos para GenomiX"""
        return [
            {
                "name": "Fotosíntesis",
                "category": "Bioquímica",
                "definition": "Proceso mediante el cual las plantas convierten luz solar, CO2 y agua en glucosa y oxígeno",
                "description": "La fotosíntesis ocurre en dos fases: reacciones luminosas (fotosistemas) y ciclo de Calvin",
                "importance": "Fundamental para la vida en la Tierra, produce oxígeno y captura carbono",
                "examples": ["Plantas verdes", "Algas", "Cyanobacterias"],
                "related_concepts": ["Respiración celular", "Cadena de electrones", "ATP"],
                "keywords": ["luz", "clorofila", "CO2", "oxígeno", "glucosa", "energía"]
            },
            # ... (otros datos de conceptos)
        ]
    
    def _get_default_processes_data(self) -> List[Dict[str, Any]]:
        """Datos por defecto de procesos biológicos para GenomiX"""
        return [
            {
                "name": "Respiración celular",
                "category": "Metabolismo",
                "description": "Proceso que libera energía de la glucosa para producir ATP",
                "mechanism": "Glucólisis, ciclo de Krebs y cadena de transporte de electrones",
                "location": "Citoplasma y mitocondrias",
                "steps": [
                    "Glucólisis en citoplasma",
                    "Ciclo de Krebs en matriz mitocondrial",
                    "Cadena respiratoria en crestas mitocondriales"
                ],
                "inputs": ["Glucosa", "Oxígeno", "ADP"],
                "outputs": ["ATP", "CO2", "H2O"],
                "keywords": ["ATP", "glucosa", "oxígeno", "mitocondria", "energía"]
            },
            # ... (otros datos de procesos)
        ]
    
    def get_random_items(self, category: str, count: int = 1) -> List[Dict[str, Any]]:
        """Obtener items aleatorios de una categoría"""
        if not self.data[category]:
            return []
        return random.sample(self.data[category], min(count, len(self.data[category])))
    
    # Métodos específicos para compatibilidad
    def get_random_species(self, count: int = 1) -> List[Dict[str, Any]]:
        return self.get_random_items('species', count)
    
    def get_random_concepts(self, count: int = 1) -> List[Dict[str, Any]]:
        return self.get_random_items('concepts', count)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de conocimiento"""
        embedding_dim = None
        if self.model:
            try:
                test_embedding = self.model.encode(["test"])
                embedding_dim = test_embedding.shape[1]
            except:
                embedding_dim = "Desconocida"
        
        stats = {
            "embedding_dimension": embedding_dim,
            "model_available": bool(self.model),
            "search_mode": "vector" if self.model else "basic"
        }
        
        # Agregar estadísticas por categoría
        for category in self.DATA_CATEGORIES:
            stats[f"total_{category}"] = len(self.data[category])
            stats[f"{category}_index_created"] = bool(self.DATA_CATEGORIES[category]['index'])
        
        return stats
    
    def search_combined(self, query: str, top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Búsqueda combinada en todas las categorías
        """
        return {
            "species": self.search_species(query, top_k),
            "concepts": self.search_concepts(query, top_k),
            "processes": self.search_processes(query, top_k)
        }
    
    def update_embeddings(self):
        """Actualizar todos los embeddings y recrear índices"""
        if not self.model:
            logger.warning("Modelo no disponible, no se pueden actualizar embeddings")
            return False
            
        logger.info("Actualizando embeddings de la base de conocimiento...")
        try:
            self._create_vector_indices()
            logger.info("Embeddings actualizados exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error actualizando embeddings: {e}")
            return False
    
    def export_knowledge_base(self, export_path: str):
        """Exportar toda la base de conocimiento a un archivo"""
        try:
            export_data = {
                "species": self.data['species'],
                "concepts": self.data['concepts'], 
                "processes": self.data['processes'],
                "statistics": self.get_statistics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de conocimiento exportada a {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando base de conocimiento: {e}")
            return False
    
    def import_knowledge_base(self, import_path: str):
        """Importar base de conocimiento desde archivo"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Actualizar datos
            for category in ['species', 'concepts', 'processes']:
                if category in import_data:
                    self.data[category] = import_data[category]
                    self._save_data(category)
            
            # Recrear índices si tenemos modelo
            if self.model:
                self._create_vector_indices()
            
            logger.info(f"Base de conocimiento importada desde {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importando base de conocimiento: {e}")
            return False
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validar integridad de los datos"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Validar cada categoría
        for category in self.DATA_CATEGORIES:
            for i, item in enumerate(self.data[category]):
                if not item.get('name'):
                    validation_results["errors"].append(f"{category.capitalize()} {i}: falta nombre")
                    validation_results["valid"] = False
        
        # Información sobre el estado del sistema
        if not self.model:
            validation_results["info"].append("Modelo de embeddings no disponible - usando búsqueda básica")
        
        # Validar índices
        for category in self.DATA_CATEGORIES:
            if self.model and not self.DATA_CATEGORIES[category]['index']:
                validation_results["warnings"].append(f"Índice de {category} no creado")
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        status = {
            "model_loaded": bool(self.model),
            "search_mode": "vector" if self.model else "basic",
            "statistics": self.get_statistics()
        }
        
        # Agregar estado por categoría
        for category in self.DATA_CATEGORIES:
            status[f"{category}_loaded"] = len(self.data[category]) > 0
            status[f"{category}_index_available"] = bool(self.DATA_CATEGORIES[category]['index'])
        
        return status
