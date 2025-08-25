import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
from pathlib import Path
import pickle
import random
from datetime import datetime

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
        self.model = None
        
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
        
        # Estado de inicialización
        self.is_initialized = False
        
        try:
            self._initialize_knowledge_base()
            logger.info("Base de conocimiento GenomiX inicializada exitosamente")
        except Exception as e:
            logger.error(f"Error inicializando base de conocimiento: {e}")
            self._create_fallback_mode()
    
    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """Cargar modelo de embeddings optimizado para texto científico"""
        try:
            # Intentar cargar modelo especializado en textos científicos
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Modelo de embeddings cargado exitosamente")
            return model
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo de embeddings: {e}")
            try:
                # Fallback a modelo más básico
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                logger.info("Modelo de embeddings fallback cargado")
                return model
            except Exception as e2:
                logger.error(f"Error cargando modelo fallback: {e2}")
                return None
    
    def _initialize_knowledge_base(self):
        """Inicializar y cargar todos los datos de la base de conocimiento"""
        try:
            # Crear directorio si no existe
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Cargar modelo de embeddings
            self.model = self._load_embedding_model()
            
            # Cargar datos
            self._load_species_data()
            self._load_concepts_data()
            self._load_processes_data()
            
            # Solo crear índices si tenemos modelo
            if self.model:
                self._create_vector_indices()
                self.is_initialized = True
            else:
                logger.warning("Modelo de embeddings no disponible, usando modo básico")
                self.is_initialized = False
                
        except Exception as e:
            logger.error(f"Error inicializando base de conocimiento: {e}")
            self._create_default_data()
            self.is_initialized = False
    
    def _create_fallback_mode(self):
        """Crear modo fallback cuando no se pueden cargar embeddings"""
        logger.info("Iniciando modo fallback sin embeddings vectoriales")
        self.model = None
        self.is_initialized = False
        
        # Crear directorio si no existe
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos básicos
        self._load_species_data()
        self._load_concepts_data()
        self._load_processes_data()
        
        # Si no hay datos, crear datos por defecto
        if not (self.species_data or self.concepts_data or self.processes_data):
            self._create_default_data()
    
    def _load_species_data(self):
        """Cargar datos de especies biológicas"""
        species_file = self.data_dir / "species_data.json"
        
        try:
            if species_file.exists():
                with open(species_file, 'r', encoding='utf-8') as f:
                    self.species_data = json.load(f)
            else:
                self.species_data = self._get_default_species_data()
                self._save_species_data()
            
            logger.info(f"Cargadas {len(self.species_data)} especies")
        except Exception as e:
            logger.error(f"Error cargando especies: {e}")
            self.species_data = self._get_default_species_data()
    
    def _load_concepts_data(self):
        """Cargar datos de conceptos biológicos"""
        concepts_file = self.data_dir / "biology_concepts.json"
        
        try:
            if concepts_file.exists():
                with open(concepts_file, 'r', encoding='utf-8') as f:
                    self.concepts_data = json.load(f)
            else:
                self.concepts_data = self._get_default_concepts_data()
                self._save_concepts_data()
            
            logger.info(f"Cargados {len(self.concepts_data)} conceptos biológicos")
        except Exception as e:
            logger.error(f"Error cargando conceptos: {e}")
            self.concepts_data = self._get_default_concepts_data()
    
    def _load_processes_data(self):
        """Cargar datos de procesos biológicos"""
        processes_file = self.data_dir / "biological_processes.json"
        
        try:
            if processes_file.exists():
                with open(processes_file, 'r', encoding='utf-8') as f:
                    self.processes_data = json.load(f)
            else:
                self.processes_data = self._get_default_processes_data()
                self._save_processes_data()
            
            logger.info(f"Cargados {len(self.processes_data)} procesos biológicos")
        except Exception as e:
            logger.error(f"Error cargando procesos: {e}")
            self.processes_data = self._get_default_processes_data()
    
    def _create_vector_indices(self):
        """Crear índices vectoriales FAISS para búsqueda semántica"""
        if not self.model:
            logger.warning("Modelo no disponible, no se pueden crear índices vectoriales")
            return
            
        try:
            # Crear embeddings para especies
            if self.species_data:
                species_texts = [self._format_species_text(s) for s in self.species_data]
                self.species_embeddings = self.model.encode(species_texts)
                
                # Crear índice FAISS
                dimension = self.species_embeddings.shape[1]
                self.species_index = faiss.IndexFlatIP(dimension)  # Producto interno
                # Normalizar embeddings
                species_embeddings_norm = self.species_embeddings.copy()
                faiss.normalize_L2(species_embeddings_norm)
                self.species_index.add(species_embeddings_norm.astype(np.float32))
            
            # Crear embeddings para conceptos
            if self.concepts_data:
                concepts_texts = [self._format_concept_text(c) for c in self.concepts_data]
                self.concepts_embeddings = self.model.encode(concepts_texts)
                
                dimension = self.concepts_embeddings.shape[1]
                self.concepts_index = faiss.IndexFlatIP(dimension)
                concepts_embeddings_norm = self.concepts_embeddings.copy()
                faiss.normalize_L2(concepts_embeddings_norm)
                self.concepts_index.add(concepts_embeddings_norm.astype(np.float32))
            
            # Crear embeddings para procesos
            if self.processes_data:
                processes_texts = [self._format_process_text(p) for p in self.processes_data]
                self.processes_embeddings = self.model.encode(processes_texts)
                
                dimension = self.processes_embeddings.shape[1] 
                self.processes_index = faiss.IndexFlatIP(dimension)
                processes_embeddings_norm = self.processes_embeddings.copy()
                faiss.normalize_L2(processes_embeddings_norm)
                self.processes_index.add(processes_embeddings_norm.astype(np.float32))
            
            logger.info("Índices vectoriales creados exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando índices vectoriales: {e}")
            self.is_initialized = False
    
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
        return ' '.join([str(part) for part in text_parts if part])
    
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
        return ' '.join([str(part) for part in text_parts if part])
    
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
        return ' '.join([str(part) for part in text_parts if part])
    
    def search_species(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar especies usando búsqueda semántica o texto básico
        
        Args:
            query: Descripción de características de la especie
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de especies más similares
        """
        if not self.species_data:
            return []
        
        # Búsqueda vectorial si está disponible
        if self.model and self.species_index and self.is_initialized:
            return self._vector_search_species(query, top_k)
        
        # Búsqueda básica de texto
        return self._basic_search_species(query, top_k)
    
    def _vector_search_species(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial de especies"""
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
                    species['search_method'] = 'vector'
                    results.append(species)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial de especies: {e}")
            return self._basic_search_species(query, top_k)
    
    def _basic_search_species(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda básica de texto en especies"""
        query_lower = query.lower()
        results = []
        
        for species in self.species_data:
            score = 0
            # Buscar en campos de texto
            searchable_text = ' '.join([
                str(species.get('name', '')),
                str(species.get('scientific_name', '')),
                str(species.get('characteristics', '')),
                str(species.get('habitat', '')),
                str(species.get('behavior', '')),
                ' '.join(species.get('keywords', []))
            ]).lower()
            
            # Calcular score básico
            if query_lower in searchable_text:
                score += 0.8
            
            # Score por palabras individuales
            query_words = query_lower.split()
            for word in query_words:
                if word in searchable_text:
                    score += 0.2
            
            if score > 0:
                species_copy = species.copy()
                species_copy['similarity_score'] = score
                species_copy['confidence'] = min(90, max(50, int(score * 80)))
                species_copy['search_method'] = 'basic'
                results.append(species_copy)
        
        # Ordenar por score y retornar top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def search_concepts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar conceptos biológicos usando búsqueda semántica o texto básico
        
        Args:
            query: Consulta sobre concepto biológico
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de conceptos más relevantes
        """
        if not self.concepts_data:
            return []
        
        # Búsqueda vectorial si está disponible
        if self.model and self.concepts_index and self.is_initialized:
            return self._vector_search_concepts(query, top_k)
        
        # Búsqueda básica de texto
        return self._basic_search_concepts(query, top_k)
    
    def _vector_search_concepts(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial de conceptos"""
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
                    concept['search_method'] = 'vector'
                    results.append(concept)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial de conceptos: {e}")
            return self._basic_search_concepts(query, top_k)
    
    def _basic_search_concepts(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda básica de texto en conceptos"""
        query_lower = query.lower()
        results = []
        
        for concept in self.concepts_data:
            score = 0
            searchable_text = ' '.join([
                str(concept.get('name', '')),
                str(concept.get('definition', '')),
                str(concept.get('description', '')),
                ' '.join(concept.get('keywords', [])),
                ' '.join(concept.get('related_concepts', []))
            ]).lower()
            
            if query_lower in searchable_text:
                score += 0.8
            
            query_words = query_lower.split()
            for word in query_words:
                if word in searchable_text:
                    score += 0.2
            
            if score > 0:
                concept_copy = concept.copy()
                concept_copy['relevance_score'] = score
                concept_copy['search_method'] = 'basic'
                results.append(concept_copy)
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def search_processes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar procesos biológicos usando búsqueda semántica o texto básico
        
        Args:
            query: Consulta sobre proceso biológico
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de procesos más relevantes
        """
        if not self.processes_data:
            return []
        
        # Búsqueda vectorial si está disponible
        if self.model and self.processes_index and self.is_initialized:
            return self._vector_search_processes(query, top_k)
        
        # Búsqueda básica de texto
        return self._basic_search_processes(query, top_k)
    
    def _vector_search_processes(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial de procesos"""
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
                    process['search_method'] = 'vector'
                    results.append(process)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial de procesos: {e}")
            return self._basic_search_processes(query, top_k)
    
    def _basic_search_processes(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda básica de texto en procesos"""
        query_lower = query.lower()
        results = []
        
        for process in self.processes_data:
            score = 0
            searchable_text = ' '.join([
                str(process.get('name', '')),
                str(process.get('description', '')),
                str(process.get('mechanism', '')),
                ' '.join(process.get('steps', [])),
                ' '.join(process.get('keywords', []))
            ]).lower()
            
            if query_lower in searchable_text:
                score += 0.8
            
            query_words = query_lower.split()
            for word in query_words:
                if word in searchable_text:
                    score += 0.2
            
            if score > 0:
                process_copy = process.copy()
                process_copy['relevance_score'] = score
                process_copy['search_method'] = 'basic'
                results.append(process_copy)
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def add_species(self, species_data: Dict[str, Any]):
        """Agregar nueva especie a la base de conocimiento"""
        self.species_data.append(species_data)
        self._save_species_data()
        # Recrear índices si están disponibles
        if self.model and self.is_initialized:
            self._create_vector_indices()
        logger.info(f"Agregada nueva especie: {species_data.get('name', 'Sin nombre')}")
    
    def add_concept(self, concept_data: Dict[str, Any]):
        """Agregar nuevo concepto a la base de conocimiento"""
        self.concepts_data.append(concept_data)
        self._save_concepts_data()
        if self.model and self.is_initialized:
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
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "species_data.json", 'w', encoding='utf-8') as f:
                json.dump(self.species_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de especies: {e}")
    
    def _save_concepts_data(self):
        """Guardar datos de conceptos en archivo JSON"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "biology_concepts.json", 'w', encoding='utf-8') as f:
                json.dump(self.concepts_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de conceptos: {e}")
    
    def _save_processes_data(self):
        """Guardar datos de procesos en archivo JSON"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
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
        
        # Crear índices solo si tenemos modelo
        if self.model:
            self._create_vector_indices()
    
    def get_random_species(self, count: int = 1) -> List[Dict[str, Any]]:
        """Obtener especies aleatorias para ejemplos"""
        if not self.species_data:
            return []
        return random.sample(self.species_data, min(count, len(self.species_data)))
    
    def get_random_concepts(self, count: int = 1) -> List[Dict[str, Any]]:
        """Obtener conceptos aleatorios para ejemplos"""
        if not self.concepts_data:
            return []
        return random.sample(self.concepts_data, min(count, len(self.concepts_data)))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de conocimiento"""
        embedding_dim = None
        if self.model:
            try:
                # Intentar obtener dimensión del modelo
                test_embedding = self.model.encode(["test"])
                embedding_dim = test_embedding.shape[1]
            except:
                embedding_dim = "Desconocida"
        
        return {
            "total_species": len(self.species_data),
            "total_concepts": len(self.concepts_data),
            "total_processes": len(self.processes_data),
            "species_families": len(set(s.get('family', 'Unknown') for s in self.species_data)),
            "concept_categories": len(set(c.get('category', 'Unknown') for c in self.concepts_data)),
            "embedding_dimension": embedding_dim,
            "indices_created": bool(self.species_index and self.concepts_index and self.processes_index),
            "model_available": bool(self.model),
            "is_initialized": self.is_initialized,
            "search_mode": "vector" if self.is_initialized else "basic"
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
        if not self.model:
            logger.warning("Modelo no disponible, no se pueden actualizar embeddings")
            return False
            
        logger.info("Actualizando embeddings de la base de conocimiento...")
        try:
            self._create_vector_indices()
            self.is_initialized = True
            logger.info("Embeddings actualizados exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error actualizando embeddings: {e}")
            return False
    
    def export_knowledge_base(self, export_path: str):
        """Exportar toda la base de conocimiento a un archivo"""
        try:
            export_data = {
                "species": self.species_data,
                "concepts": self.concepts_data, 
                "processes": self.processes_data,
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
        
        # Validar procesos
        for i, process in enumerate(self.processes_data):
            if not process.get('name'):
                validation_results["errors"].append(f"Proceso {i}: falta nombre")
                validation_results["valid"] = False
        
        # Información sobre el estado del sistema
        if not self.model:
            validation_results["info"].append("Modelo de embeddings no disponible - usando búsqueda básica")
        
        if not self.is_initialized:
            validation_results["info"].append("Índices vectoriales no inicializados - usando búsqueda básica")
        
        # Validar índices solo si deberían existir
        if self.model and not self.species_index:
            validation_results["warnings"].append("Índice de especies no creado")
        
        if self.model and not self.concepts_index:
            validation_results["warnings"].append("Índice de conceptos no creado")
            
        if self.model and not self.processes_index:
            validation_results["warnings"].append("Índice de procesos no creado")
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema"""
        return {
            "model_loaded": bool(self.model),
            "is_initialized": self.is_initialized,
            "search_mode": "vector" if self.is_initialized else "basic",
            "data_loaded": {
                "species": len(self.species_data) > 0,
                "concepts": len(self.concepts_data) > 0,
                "processes": len(self.processes_data) > 0
            },
            "indices_available": {
                "species": bool(self.species_index),
                "concepts": bool(self.concepts_index),
                "processes": bool(self.processes_index)
            },
            "statistics": self.get_statistics()
        }
