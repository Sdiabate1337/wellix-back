"""
Core Workflow Node Strategies.
Implémente les nodes principaux du workflow selon le Strategy Pattern.

Architecture Pattern : Strategy + Template Method + Chain of Responsibility
Inspiration : Apache Airflow, Prefect, Temporal.io
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from abc import abstractmethod
import structlog

from app.workflows.interfaces import (
    IWorkflowNode, WorkflowState, WorkflowStage, WorkflowNodeError,
    InputData, InputType, QualityLevel
)
from app.workflows.container import workflow_container
from app.workflows.factories import IExtractionStrategyFactory, IAnalysisStrategyFactory
from app.models.health import NutritionData, UserHealthContext

logger = structlog.get_logger(__name__)


class BaseWorkflowNode(IWorkflowNode):
    """
    Base abstraite pour tous les nodes workflow.
    
    Implémente le Template Method Pattern avec hooks pour les sous-classes.
    Fournit logging, metrics, error handling communs.
    """
    
    def __init__(self):
        self._logger = logger.bind(node=self.node_name)
        self._performance_stats = {}
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        """
        Template Method : Algorithme de traitement commun avec hooks.
        
        Séquence :
        1. Validation des préconditions
        2. Logging de début
        3. Traitement métier (hook)
        4. Validation du résultat
        5. Logging de fin
        """
        start_time = time.time()
        
        try:
            # Hook 1 : Validation des préconditions
            validation_errors = await self.validate_preconditions(state)
            if validation_errors:
                raise WorkflowNodeError(
                    self.node_name,
                    f"Precondition validation failed: {', '.join(validation_errors)}",
                    recoverable=True
                )
            
            self._logger.info(
                "Node processing started",
                workflow_id=state.workflow_id,
                current_stage=state.current_stage.value
            )
            
            # Hook 2 : Traitement métier spécifique (implémenté par les sous-classes)
            result_state = await self._process_internal(state)
            
            # Hook 3 : Validation du résultat
            await self._validate_result(result_state)
            
            # Avancement du stage
            next_stage = self._get_next_stage()
            if next_stage:
                result_state = result_state.advance_to_stage(next_stage)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            self._logger.info(
                "Node processing completed",
                workflow_id=state.workflow_id,
                processing_time_ms=processing_time,
                next_stage=next_stage.value if next_stage else "COMPLETED"
            )
            
            return result_state
            
        except WorkflowNodeError:
            # Re-raise workflow errors as-is
            raise
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            self._logger.error(
                "Node processing failed",
                workflow_id=state.workflow_id,
                error=str(e),
                processing_time_ms=processing_time
            )
            
            # Encapsulate dans WorkflowNodeError
            raise WorkflowNodeError(
                self.node_name,
                f"Processing failed: {str(e)}",
                original_error=e,
                recoverable=self._is_error_recoverable(e)
            )
    
    @abstractmethod
    async def _process_internal(self, state: WorkflowState) -> WorkflowState:
        """Hook pour le traitement métier spécifique - à implémenter par les sous-classes."""
        pass
    
    @abstractmethod
    def _get_next_stage(self) -> Optional[WorkflowStage]:
        """Retourne le prochain stage après ce node."""
        pass
    
    async def _validate_result(self, state: WorkflowState) -> None:
        """Hook pour valider le résultat - peut être surchargé."""
        if state.has_errors:
            raise WorkflowNodeError(
                self.node_name,
                f"Result validation failed: {state.last_error.message if state.last_error else 'Unknown error'}"
            )
    
    def _is_error_recoverable(self, error: Exception) -> bool:
        """Détermine si une erreur est récupérable - peut être surchargé."""
        # Par défaut, les erreurs de réseau et temporaires sont récupérables
        recoverable_errors = [
            "ConnectionError", "TimeoutError", "HTTPError", 
            "TemporaryFailure", "RateLimitError"
        ]
        return any(err_type in str(type(error)) for err_type in recoverable_errors)


# ================================
# INPUT VALIDATION NODE
# ================================

class InputValidationNode(BaseWorkflowNode):
    """
    Node de validation d'input - Chain of Responsibility Pattern.
    
    Valide les données d'entrée selon une chaîne de validateurs.
    """
    
    @property
    def node_name(self) -> str:
        return "input_validation"
    
    @property
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.INPUT_VALIDATION
    
    async def can_process(self, state: WorkflowState) -> bool:
        """Peut toujours traiter si on est au bon stage."""
        return state.current_stage == WorkflowStage.ENTRY_ROUTING
    
    async def validate_preconditions(self, state: WorkflowState) -> List[str]:
        """Valide que nous avons des données d'input."""
        errors = []
        
        if not state.input_data:
            errors.append("No input data provided")
            return errors
        
        # Validation selon le type d'input
        input_data = state.input_data
        
        if input_data.type == InputType.IMAGE and not input_data.image_data:
            errors.append("Image data required for IMAGE input type")
        elif input_data.type == InputType.BARCODE and not input_data.barcode:
            errors.append("Barcode required for BARCODE input type")
        elif input_data.type == InputType.JSON_DATA and not input_data.json_data:
            errors.append("JSON data required for JSON_DATA input type")
        
        return errors
    
    async def _process_internal(self, state: WorkflowState) -> WorkflowState:
        """Validation de l'input avec chaîne de responsabilité."""
        input_data = state.input_data
        
        # Chaîne de validateurs
        validators = [
            self._validate_input_format,
            self._validate_input_size,
            self._validate_input_security,
            self._validate_input_quality
        ]
        
        validation_results = {}
        
        for validator in validators:
            validator_name = validator.__name__
            try:
                result = await validator(input_data)
                validation_results[validator_name] = result
                
                if not result.get("valid", True):
                    self._logger.warning(
                        "Input validation failed",
                        validator=validator_name,
                        reason=result.get("reason", "Unknown"),
                        workflow_id=state.workflow_id
                    )
                    
            except Exception as e:
                self._logger.error(
                    "Validator execution failed",
                    validator=validator_name,
                    error=str(e),
                    workflow_id=state.workflow_id
                )
                validation_results[validator_name] = {
                    "valid": False,
                    "reason": f"Validator error: {str(e)}"
                }
        
        # Mise à jour du state avec les résultats de validation
        enhanced_input_data = InputData(
            type=input_data.type,
            image_data=input_data.image_data,
            barcode=input_data.barcode,
            json_data=input_data.json_data,
            complexity_hints={
                **input_data.complexity_hints,
                "validation_results": validation_results,
                "quality_score": self._calculate_quality_score(validation_results)
            }
        )
        
        from dataclasses import replace
        return replace(state, input_data=enhanced_input_data)
    
    def _get_next_stage(self) -> Optional[WorkflowStage]:
        return WorkflowStage.TOKEN_VALIDATION
    
    async def _validate_input_format(self, input_data: InputData) -> Dict[str, Any]:
        """Valide le format des données d'input."""
        if input_data.type == InputType.IMAGE:
            # Validation MIME type, headers, etc.
            if input_data.image_data and len(input_data.image_data) > 0:
                # Check magic bytes pour détecter le type réel
                magic_bytes = input_data.image_data[:10]
                if magic_bytes.startswith(b'\xff\xd8') or magic_bytes.startswith(b'\x89PNG'):
                    return {"valid": True, "format": "valid_image"}
                else:
                    return {"valid": False, "reason": "Invalid image format"}
            return {"valid": False, "reason": "Empty image data"}
        
        elif input_data.type == InputType.BARCODE:
            # Validation format barcode
            barcode = input_data.barcode
            if barcode and len(barcode) >= 8 and barcode.isdigit():
                return {"valid": True, "format": "valid_barcode"}
            return {"valid": False, "reason": "Invalid barcode format"}
        
        elif input_data.type == InputType.JSON_DATA:
            # Validation structure JSON
            if isinstance(input_data.json_data, dict):
                return {"valid": True, "format": "valid_json"}
            return {"valid": False, "reason": "Invalid JSON structure"}
        
        return {"valid": True, "format": "unknown"}
    
    async def _validate_input_size(self, input_data: InputData) -> Dict[str, Any]:
        """Valide la taille des données d'input."""
        MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
        MAX_JSON_SIZE = 1 * 1024 * 1024    # 1MB
        
        if input_data.type == InputType.IMAGE and input_data.image_data:
            size = len(input_data.image_data)
            if size > MAX_IMAGE_SIZE:
                return {"valid": False, "reason": f"Image too large: {size} bytes > {MAX_IMAGE_SIZE}"}
            return {"valid": True, "size_bytes": size}
        
        elif input_data.type == InputType.JSON_DATA and input_data.json_data:
            # Estimation de la taille JSON
            import json
            size = len(json.dumps(input_data.json_data))
            if size > MAX_JSON_SIZE:
                return {"valid": False, "reason": f"JSON too large: {size} bytes > {MAX_JSON_SIZE}"}
            return {"valid": True, "size_bytes": size}
        
        return {"valid": True, "size_bytes": 0}
    
    async def _validate_input_security(self, input_data: InputData) -> Dict[str, Any]:
        """Valide la sécurité des données d'input."""
        # Scan basique pour contenus malveillants
        if input_data.type == InputType.IMAGE and input_data.image_data:
            # Check pour scripts cachés, headers malveillants
            suspicious_patterns = [b'<script', b'javascript:', b'<?php']
            for pattern in suspicious_patterns:
                if pattern in input_data.image_data:
                    return {"valid": False, "reason": "Suspicious content detected"}
            return {"valid": True, "security": "clean"}
        
        elif input_data.type == InputType.JSON_DATA and input_data.json_data:
            # Check pour injection, scripts
            import json
            json_str = json.dumps(input_data.json_data)
            suspicious_patterns = ['<script', 'javascript:', 'eval(', 'function(']
            for pattern in suspicious_patterns:
                if pattern in json_str:
                    return {"valid": False, "reason": "Suspicious JSON content detected"}
            return {"valid": True, "security": "clean"}
        
        return {"valid": True, "security": "clean"}
    
    async def _validate_input_quality(self, input_data: InputData) -> Dict[str, Any]:
        """Estime la qualité des données d'input."""
        quality_score = 0.5  # Score par défaut
        
        if input_data.type == InputType.IMAGE and input_data.image_data:
            # Estimation basique de qualité image
            size = len(input_data.image_data)
            if size > 1024 * 1024:  # > 1MB = haute qualité probable
                quality_score = 0.9
            elif size > 100 * 1024:  # > 100KB = qualité moyenne
                quality_score = 0.7
            else:
                quality_score = 0.4  # Petite image = qualité limitée
        
        elif input_data.type == InputType.JSON_DATA and input_data.json_data:
            # Estimation basée sur la complétude des données
            required_fields = ['product_name', 'nutrition_data', 'serving_size']
            present_fields = sum(1 for field in required_fields if field in input_data.json_data)
            quality_score = present_fields / len(required_fields)
        
        elif input_data.type == InputType.BARCODE:
            # Barcode = qualité potentiellement haute si on trouve le produit
            quality_score = 0.8
        
        return {"valid": True, "quality_score": quality_score}
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calcule un score de qualité global."""
        quality_scores = []
        
        for result in validation_results.values():
            if result.get("valid", False):
                # Pondération selon le type de validation
                if "quality_score" in result:
                    quality_scores.append(result["quality_score"])
                else:
                    quality_scores.append(0.8)  # Score par défaut pour validation réussie
            else:
                quality_scores.append(0.0)  # Échec de validation
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0


# ================================
# TOKEN VALIDATION NODE  
# ================================

class TokenValidationNode(BaseWorkflowNode):
    """
    Node de validation de tokens - Strategy + Observer Pattern.
    
    Valide et réserve les tokens nécessaires pour l'analyse.
    """
    
    def __init__(self, token_service=None, cost_calculator=None):
        super().__init__()
        # Dependency injection via container
        self._token_service = token_service  # Will be injected
        self._cost_calculator = cost_calculator  # Will be injected
    
    @property
    def node_name(self) -> str:
        return "token_validation"
    
    @property  
    def required_stage(self) -> WorkflowStage:
        return WorkflowStage.TOKEN_VALIDATION
    
    async def can_process(self, state: WorkflowState) -> bool:
        """Peut traiter si on a des données d'input validées."""
        return (state.current_stage == WorkflowStage.INPUT_VALIDATION and 
                state.input_data is not None)
    
    async def validate_preconditions(self, state: WorkflowState) -> List[str]:
        """Valide les préconditions pour la validation de tokens."""
        errors = []
        
        if not state.user_context or not state.user_context.user_id:
            errors.append("User context with user_id required")
        
        if not state.analysis_config:
            errors.append("Analysis configuration required")
        
        return errors
    
    async def _process_internal(self, state: WorkflowState) -> WorkflowState:
        """Validation et réservation des tokens."""
        # Calcul du coût selon la configuration d'analyse
        cost = await self._calculate_analysis_cost(state)
        
        self._logger.info(
            "Calculating token cost",
            workflow_id=state.workflow_id,
            user_id=str(state.user_context.user_id),
            quality_level=state.analysis_config.quality_level.value,
            estimated_cost=cost
        )
        
        # Validation et réservation via le service de tokens
        # Note: En production, ceci serait injecté via le container
        reservation_result = await self._reserve_tokens(state, cost)
        
        if not reservation_result["success"]:
            raise WorkflowNodeError(
                self.node_name,
                f"Token reservation failed: {reservation_result['reason']}",
                recoverable=False
            )
        
        # Mise à jour du state avec la réservation
        from dataclasses import replace
        return replace(
            state,
            input_data=InputData(
                type=state.input_data.type,
                image_data=state.input_data.image_data,
                barcode=state.input_data.barcode,
                json_data=state.input_data.json_data,
                complexity_hints={
                    **state.input_data.complexity_hints,
                    "token_cost": cost,
                    "token_reservation": reservation_result
                }
            )
        )
    
    def _get_next_stage(self) -> Optional[WorkflowStage]:
        return WorkflowStage.DATA_EXTRACTION
    
    async def _calculate_analysis_cost(self, state: WorkflowState) -> int:
        """Calcule le coût en tokens pour l'analyse."""
        base_cost = 1  # Coût de base
        
        # Facteurs de coût selon la qualité
        quality_multipliers = {
            QualityLevel.BASIC: 1,
            QualityLevel.STANDARD: 2,
            QualityLevel.PREMIUM: 5,
            QualityLevel.EXPERT: 10
        }
        
        cost = base_cost * quality_multipliers.get(state.analysis_config.quality_level, 1)
        
        # Facteurs additionnels
        if state.analysis_config.enable_alternatives:
            cost += 2 * state.analysis_config.max_alternatives_per_type
        
        if state.analysis_config.enable_chat_context:
            cost += 3
        
        # Complexité de l'input
        input_complexity = state.input_data.complexity_hints.get("quality_score", 0.5)
        if input_complexity < 0.5:  # Input de faible qualité = plus de travail
            cost = int(cost * 1.5)
        
        return cost
    
    async def _reserve_tokens(self, state: WorkflowState, cost: int) -> Dict[str, Any]:
        """Réserve les tokens nécessaires."""
        # Simulation de réservation de tokens
        # En production, ceci utiliserait le vrai service de tokens
        
        user_id = str(state.user_context.user_id)
        
        # Simulation : vérification des tokens disponibles
        # TODO: Remplacer par l'appel au vrai service via DI
        available_tokens = 100  # Simulation
        
        if available_tokens >= cost:
            return {
                "success": True,
                "reserved_tokens": cost,
                "remaining_tokens": available_tokens - cost,
                "reservation_id": f"res_{state.workflow_id}_{int(time.time())}"
            }
        else:
            return {
                "success": False,
                "reason": f"Insufficient tokens: need {cost}, have {available_tokens}",
                "available_tokens": available_tokens
            }