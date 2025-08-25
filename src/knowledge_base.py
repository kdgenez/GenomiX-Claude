import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class BiologyKnowledgeBase:
    """
    Base de conocimiento biológico especializada para GenomiX
    Utiliza embeddings vectoriales para búsqueda semántica avanzada
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializar la base de conocimiento GenomiX
        
        Args:
            data_dir: Directorio con archivos de datos biológicos
        """
        self.data_dir = Path(data_dir)
        self.model = self._load_embedding_model()
        
        # Estructuras de datos
        self.species_data = []
        self.concepts_data = []
        self.processes_data = []
        
        # Índices FAISS
        self.species_index = None
        self.concepts_index = None
        self.processes_index = None
        
        # Embeddings cacheados
        self.species_embeddings = None
        self.concepts_embeddings = None  
        self.processes_embeddings = None
        
        self._initialize_knowledge_base()
        
        logger.info("Base de conocimiento GenomiX inicializada")
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Cargar modelo de embeddings optimizado para texto científico"""
        try:
            # Modelo especializado en textos científicos
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Modelo de embeddings cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            # Fallback a modelo más básico
            return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def _initialize_knowledge_base(self):
        """Inicializar y cargar todos los datos de la base de conocimiento"""
        try:
            self._load_species_data()
            self._load_concepts_data()
            self._load_processes_data()
            self._create_vector_indices()
        except Exception as e:
            logger.error(f"Error inicializando base de conocimiento: {e}")
            self._create_default_data()
    
    def _load_species_data(self):
        """Cargar datos de especies biológicas"""
        species_file = self.data_dir / "species_data.json"
        
        if species_file.exists():
            with open(species_file, 'r', encoding='utf-8') as f:
                self.species_data = json.load(f)
        else:
            self.species_data = self._get_default_species_data()
            self._save_species_data()
        
        logger.info(f"Cargadas {len(self.species_data)} especies")
    
    def _load_concepts_data(self):
        """Cargar datos de conceptos biológicos"""
        concepts_file = self.data_dir / "biology_concepts.json"
        
        if concepts_file.exists():
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.concepts_data = json.load(f)
        else:
            self.concepts_data = self._get_default_concepts_data()
            self._save_concepts_data()
        
        logger.info(f"Cargados {len(self.concepts_data)} conceptos biológicos")
    
    def _load_processes_data(self):
        """Cargar datos de procesos biológicos"""
        processes_file = self.data_dir / "biological_processes.json"
        
        if processes_file.exists():
            with open(processes_file, 'r', encoding='utf-8') as f:
                self.processes_data = json.load(f)
        else:
            self.processes_data = self._get_default_processes_data()
            self._save_processes_data()
        
        logger.info(f"Cargados {len(self.processes_data)} procesos biológicos")
    
    def _create_vector_indices(self):
        """Crear índices vectoriales FAISS para búsqueda semántica"""
        try:
            # Crear embeddings para especies
            if self.species_data:
                species_texts = [self._format_species_text(s) for s in self.species_data]
                self.species_embeddings = self.model.encode(species_texts)
                
                # Crear índice FAISS
                dimension = self.species_embeddings.shape[1]
                self.species_index = faiss.IndexFlatIP(dimension)  # Producto interno
                faiss.normalize_L2(self.species_embeddings)
                self.species_index.add(self.species_embeddings.astype(np.float32))
            
            # Crear embeddings para conceptos
            if self.concepts_data:
                concepts_texts = [self._format_concept_text(c) for c in self.concepts_data]
                self.concepts_embeddings = self.model.encode(concepts_texts)
                
                dimension = self.concepts_embeddings.shape[1]
                self.concepts_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(self.concepts_embeddings)
                self.concepts_index.add(self.concepts_embeddings.astype(np.float32))
            
            # Crear embeddings para procesos
            if self.processes_data:
                processes_texts = [self._format_process_text(p) for p in self.processes_data]
                self.processes_embeddings = self.model.encode(processes_texts)
                
                dimension = self.processes_embeddings.shape[1] 
                self.processes_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(self.processes_embeddings)
                self.processes_index.add(self.processes_embeddings.astype(np.float32))
            
            logger.info("Índices vectoriales creados exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando índices vectoriales: {e}")
    
    def _format_species_text(self, species: Dict[str, Any]) -> str:
        """Formatear datos de especies para embedding"""
        text_parts = [
            species.get('name', ''),
            species.get('scientific_name', ''),
            species.get('family', ''),
            species.get('characteristics', ''),
            species.get('habitat', ''),
            species.get('behavior', ''),
            ' '.join(species.get('keywords', []))
        ]
        return ' '.join([part for part in text_parts if part])
    
    def _format_concept_text(self, concept: Dict[str, Any]) -> str:
        """Formatear datos de conceptos para embedding"""
        text_parts = [
            concept.get('name', ''),
            concept.get('definition', ''),
            concept.get('description', ''),
            concept.get('importance', ''),
            ' '.join(concept.get('keywords', [])),
            ' '.join(concept.get('related_concepts', []))
        ]
        return ' '.join([part for part in text_parts if part])
    
    def _format_process_text(self, process: Dict[str, Any]) -> str:
        """Formatear datos de procesos para embedding"""
        text_parts = [
            process.get('name', ''),
            process.get('description', ''),
            process.get('mechanism', ''),
            process.get('location', ''),
            ' '.join(process.get('steps', [])),
            ' '.join(process.get('keywords', []))
        ]
        return ' '.join([part for part in text_parts if part])
    
    def search_species(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar especies usando búsqueda semántica
        
        Args:
            query: Descripción de características de la especie
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de especies más similares
        """
        if not self.species_index or not self.species_data:
            return []
        
        try:
            # Crear embedding de la consulta
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Buscar en el índice
            scores, indices = self.species_index.search(
                query_embedding.astype(np.float32), 
                min(top_k, len(self.species_data))
            )
            
            # Retornar resultados con scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # FAISS retorna -1 para no encontrados
                    species = self.species_data[idx].copy()
                    species['similarity_score'] = float(score)
                    species['confidence'] = min(95, max(60, int(score * 100)))
                    results.append(species)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de especies: {e}")
            return []
    
    def search_concepts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar conceptos biológicos usando búsqueda semántica
        
        Args:
            query: Consulta sobre concepto biológico
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de conceptos más relevantes
        """
        if not self.concepts_index or not self.concepts_data:
            return []
        
        try:
            # Crear embedding de la consulta
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Buscar en el índice
            scores, indices = self.concepts_index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(self.concepts_data))
            )
            
            # Retornar resultados con scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    concept = self.concepts_data[idx].copy()
                    concept['relevance_score'] = float(score)
                    results.append(concept)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de conceptos: {e}")
            return []
    
    def search_processes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar procesos biológicos usando búsqueda semántica
        
        Args:
            query: Consulta sobre proceso biológico
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de procesos más relevantes
        """
        if not self.processes_index or not self.processes_data:
            return []
        
        try:
            # Crear embedding de la consulta
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Buscar en el índice
            scores, indices = self.processes_index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(self.processes_data))
            )
            
            # Retornar resultados con scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    process = self.processes_data[idx].copy()
                    process['relevance_score'] = float(score)
                    results.append(process)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de procesos: {e}")
            return []
    
    def add_species(self, species_data: Dict[str, Any]):
        """Agregar nueva especie a la base de conocimiento"""
        self.species_data.append(species_data)
        self._save_species_data()
        # Recrear índices
        self._create_vector_indices()
        logger.info(f"Agregada nueva especie: {species_data.get('name', 'Sin nombre')}")
    
    def add_concept(self, concept_data: Dict[str, Any]):
        """Agregar nuevo concepto a la base de conocimiento"""
        self.concepts_data.append(concept_data)
        self._save_concepts_data()
        self._create_vector_indices()
        logger.info(f"Agregado nuevo concepto: {concept_data.get('name', 'Sin nombre')}")
    
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
            {
                "name": "Rosa roja",
                "scientific_name": "Rosa rubiginosa",
                "kingdom": "Plantae",
                "phylum": "Tracheophyta",
                "class": "Magnoliopsida",
                "order": "Rosales", 
                "family": "Rosaceae",
                "characteristics": "Flores rojas fragantes, tallos espinosos, hojas compuestas",
                "habitat": "Jardines, zonas templadas, suelos bien drenados",
                "behavior": "Floración estacional, atrae polinizadores",
                "diet": "Autótrofa: fotosíntesis",
                "size": "0.5-2 metros de altura",
                "keywords": ["planta", "flor", "roja", "espinas", "fragante", "jardín"]
            },
            {
                "name": "Colibrí",
                "scientific_name": "Trochilidae",
                "kingdom": "Animalia",
                "phylum": "Chordata", 
                "class": "Aves",
                "order": "Apodiformes",
                "family": "Trochilidae",
                "characteristics": "Muy pequeño, plumaje iridiscente, pico largo y delgado, aleteo rápido",
                "habitat": "Bosques tropicales, jardines con flores, montañas",
                "behavior": "Vuelo estacionario, migración, territorial alrededor de fuentes de néctar",
                "diet": "Nectarívoro: néctar de flores, pequeños insectos",
                "size": "5-20 cm, peso 2-20 gramos",
                "keywords": ["ave", "pequeña", "vuelo", "néctar", "iridiscente", "pico largo"]
            },
            {
                "name": "Tiburón blanco",
                "scientific_name": "Carcharodon carcharias",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Chondrichthyes",
                "order": "Lamniformes",
                "family": "Lamnidae",
                "characteristics": "Grande, cuerpo fusiforme, dientes triangulares afilados, aletas pectorales grandes",
                "habitat": "Océanos templados y tropicales, aguas costeras",
                "behavior": "Depredador apex, migratorio, caza por emboscada",
                "diet": "Carnívoro: peces, mamíferos marinos, focas",
                "size": "4-6 metros, hasta 2000 kg",
                "keywords": ["pez", "cartilaginoso", "grande", "depredador", "océano", "dientes"]
            },
            {
                "name": "Mariposa monarca",
                "scientific_name": "Danaus plexippus",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Lepidoptera",
                "family": "Nymphalidae",
                "characteristics": "Alas naranjas con venas negras, bordes negros con puntos blancos",
                "habitat": "Campos abiertos, jardines, rutas migratorias",
                "behavior": "Migración masiva, metamorfosis completa, vuelo planificado",
                "diet": "Larva: algodoncillo; adulto: néctar de flores",
                "size": "8.5-12.5 cm envergadura",
                "keywords": ["insecto", "mariposa", "naranja", "migración", "metamorfosis", "alas"]
            }
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
                "examples": ["Plantas verdes", "Algas", "Cianobacterias"],
                "related_concepts": ["Respiración celular", "Cadena de electrones", "ATP"],
                "keywords": ["luz", "clorofila", "CO2", "oxígeno", "glucosa", "energía"]
            },
            {
                "name": "Mitosis",
                "category": "Biología Celular",
                "definition": "Proceso de división celular que produce dos células diploides idénticas",
                "description": "Incluye profase, metafase, anafase y telofase, seguido de citocinesis",
                "importance": "Esencial para crecimiento, reparación y reproducción asexual",
                "examples": ["Células somáticas", "Regeneración tisular", "Crecimiento"],
                "related_concepts": ["Meiosis", "Ciclo celular", "Cromosomas"],
                "keywords": ["división", "cromosomas", "husos", "células hijas", "diploide"]
            },
            {
                "name": "Evolución",
                "category": "Biología Evolutiva",
                "definition": "Cambio en las características heredables de poblaciones a través del tiempo",
                "description": "Impulsada por selección natural, deriva genética, flujo génico y mutación",
                "importance": "Explica la diversidad de la vida y las adaptaciones",
                "examples": ["Resistencia a antibióticos", "Picos de pinzones", "Evolución humana"],
                "related_concepts": ["Selección natural", "Especiación", "Filogenia"],
                "keywords": ["adaptación", "selección", "mutación", "especies", "ancestro común"]
            },
            {
                "name": "ADN",
                "category": "Genética Molecular",
                "definition": "Ácido desoxirribonucleico, molécula que almacena información genética",
                "description": "Doble hélice de nucleótidos (A, T, G, C) con enlaces complementarios",
                "importance": "Contiene instrucciones para desarrollo y funcionamiento de organismos",
                "examples": ["Genoma humano", "Código genético", "Replicación"],
                "related_concepts": ["ARN", "Proteínas", "Genes", "Mutaciones"],
                "keywords": ["doble hélice", "nucleótidos", "bases", "genoma", "herencia"]
            },
            {
                "name": "Ecosistema",
                "category": "Ecología",
                "definition": "Comunidad de organismos interactuando con su ambiente físico",
                "description": "Incluye factores bióticos y abióticos en equilibrio dinámico",
                "importance": "Unidad funcional de la ecología, mantiene ciclos biogeoquímicos",
                "examples": ["Bosque tropical", "Arrecife de coral", "Sabana"],
                "related_concepts": ["Cadena alimentaria", "Nicho ecológico", "Biodiversidad"],
                "keywords": ["comunidad", "hábitat", "interacciones", "energía", "ciclos"]
            }
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
            {
                "name": "Transcripción",
                "category": "Expresión génica",
                "description": "Síntesis de ARN usando ADN como molde",
                "mechanism": "ARN polimerasa lee la secuencia de ADN y sintetiza ARN complementario",
                "location": "Núcleo (eucariotas), citoplasma (procariotas)",
                "steps": [
                    "Iniciación: unión de ARN polimerasa al promotor",
                    "Elongación: síntesis de ARN",
                    "Terminación: liberación del transcrito"
                ],
                "inputs": ["ADN molde", "Nucleótidos de ARN", "ARN polimerasa"],
                "outputs": ["ARN primario", "ARNm maduro"],
                "keywords": ["ARN", "polimerasa", "promotor", "gen", "nucleótidos"]
            },
            {
                "name": "Meiosis",
                "category": "Reproducción",
                "description": "División celular que produce gametos haploides",
                "mechanism": "Dos divisiones consecutivas con recombinación genética",
                "location": "Gónadas, órganos reproductivos",
                "steps": [
                    "Meiosis I: separación de cromosomas homólogos",
                    "Crossing over: intercambio genético",
                    "Meiosis II: separación de cromátidas hermanas"
                ],
                "inputs": ["Células diploides", "Energía ATP"],
                "outputs": ["Cuatro gametos haploides"],
                "keywords": ["gametos", "haploide", "recombinación", "crossing over", "reproducción"]
            }
        ]
    
    def _save_species_data(self):
        """Guardar datos de especies en archivo JSON"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.data_dir / "species_data.json", 'w', encoding='utf-8') as f:
                json.dump(self.species_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de especies: {e}")
    
    def _save_concepts_data(self):
        """Guardar datos de conceptos en archivo JSON"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.data_dir / "biology_concepts.json", 'w', encoding='utf-8') as f:
                json.dump(self.concepts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de conceptos: {e}")
    
    def _save_processes_data(self):
        """Guardar datos de procesos en archivo JSON"""
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.data_dir / "biological_processes.json", 'w', encoding='utf-8') as f:
                json.dump(self.processes_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de procesos: {e}")
    
    def _create_default_data(self):
        """Crear datos por defecto si hay errores en la carga"""
        logger.warning("Creando datos por defecto para base de conocimiento")
        self.species_data = self._get_default_species_data()
        self.concepts_data = self._get_default_concepts_data()
        self.processes_data = self._get_default_processes_data()
        
        # Guardar datos por defecto
        self._save_species_data()
        self._save_concepts_data()
        self._save_processes_data()
        
        # Crear índices
        self._create_vector_indices()
    
    def get_random_species(self, count: int = 1) -> List[Dict[str, Any]]:
        """Obtener especies aleatorias para ejemplos"""
        import random
        if not self.species_data:
            return []
        return random.sample(self.species_data, min(count, len(self.species_data)))
    
    def get_random_concepts(self, count: int = 1) -> List[Dict[str, Any]]:
        """Obtener conceptos aleatorios para ejemplos"""
        import random
        if not self.concepts_data:
            return []
        return random.sample(self.concepts_data, min(count, len(self.concepts_data)))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de conocimiento"""
        return {
            "total_species": len(self.species_data),
            "total_concepts": len(self.concepts_data),
            "total_processes": len(self.processes_data),
            "species_families": len(set(s.get('family', 'Unknown') for s in self.species_data)),
            "concept_categories": len(set(c.get('category', 'Unknown') for c in self.concepts_data)),
            "embedding_model": self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 384,
            "indices_created": bool(self.species_index and self.concepts_index and self.processes_index)
        }
    
    def search_combined(self, query: str, top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Búsqueda combinada en todas las categorías
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados por categoría
            
        Returns:
            Diccionario con resultados de especies, conceptos y procesos
        """
        return {
            "species": self.search_species(query, top_k),
            "concepts": self.search_concepts(query, top_k),
            "processes": self.search_processes(query, top_k)
        }
    
    def update_embeddings(self):
        """Actualizar todos los embeddings y recrear índices"""
        logger.info("Actualizando embeddings de la base de conocimiento...")
        self._create_vector_indices()
        logger.info("Embeddings actualizados exitosamente")
    
    def export_knowledge_base(self, export_path: str):
        """Exportar toda la base de conocimiento a un archivo"""
        try:
            export_data = {
                "species": self.species_data,
                "concepts": self.concepts_data, 
                "processes": self.processes_data,
                "statistics": self.get_statistics(),
                "export_timestamp": str(pd.Timestamp.now())
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de conocimiento exportada a {export_path}")
            
        except Exception as e:
            logger.error(f"Error exportando base de conocimiento: {e}")
    
    def import_knowledge_base(self, import_path: str):
        """Importar base de conocimiento desde archivo"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Actualizar datos
            if "species" in import_data:
                self.species_data = import_data["species"]
            if "concepts" in import_data:
                self.concepts_data = import_data["concepts"]
            if "processes" in import_data:
                self.processes_data = import_data["processes"]
            
            # Guardar y recrear índices
            self._save_species_data()
            self._save_concepts_data()
            self._save_processes_data()
            self._create_vector_indices()
            
            logger.info(f"Base de conocimiento importada desde {import_path}")
            
        except Exception as e:
            logger.error(f"Error importando base de conocimiento: {e}")
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validar integridad de los datos"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validar especies
        for i, species in enumerate(self.species_data):
            if not species.get('name'):
                validation_results["errors"].append(f"Especie {i}: falta nombre")
                validation_results["valid"] = False
            if not species.get('scientific_name'):
                validation_results["warnings"].append(f"Especie {i}: falta nombre científico")
        
        # Validar conceptos
        for i, concept in enumerate(self.concepts_data):
            if not concept.get('name'):
                validation_results["errors"].append(f"Concepto {i}: falta nombre")
                validation_results["valid"] = False
            if not concept.get('definition'):
                validation_results["warnings"].append(f"Concepto {i}: falta definición")
        
        # Validar índices
        if not self.species_index:
            validation_results["errors"].append("Índice de especies no creado")
            validation_results["valid"] = False
        
        return validation_results
