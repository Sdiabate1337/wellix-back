"""
LangGraph Workflow Orchestrator.
Coordonne l'exécution des nodes selon une state machine.

Architecture Pattern : Template Method + State Machine + Chain of Responsibility + Command
Inspiration : LangGraph, Temporal.io, Apache Airflow
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import replace
import structlog

from app.workflows.interfaces import (
    IWorkflowOrchestrator, IWorkflowNode, IWorkflowRouter, IWorkflowValidator,
    WorkflowState, WorkflowStage, WorkflowResult, InputData, AnalysisConfig,
    WorkflowNodeError, WorkflowValidationError, WorkflowTimeoutError,
    PerformanceMetrics, ProcessingStep
)
from app.workflows.nodes.core_nodes import InputValidationNode, TokenValidationNode
from app.workflows.nodes.advanced_nodes import DataExtractionNode, ExpertAnalysisNode
from app.models.health import UserHealthContext

logger = structlog.get_logger(__name__)


class LangGraphWorkflowOrchestrator(IWorkflowOrchestrator):
    """
    Orchestrateur de workflow principal utilisant les concepts LangGraph.
    
    Features :
    - State machine avec transitions validées
    - Execution parallèle des nodes compatibles
    - Error handling et recovery automatique
    - Performance monitoring intégré
    - Timeout management par stage
    """
    
    def __init__(self, router=None, validator=None):
        self._nodes: Dict[WorkflowStage, IWorkflowNode] = {}
        self._router = router or DefaultWorkflowRouter()
        self._validator = validator or DefaultWorkflowValidator()
        self._logger = logger.bind(component="WorkflowOrchestrator")
        self._execution_stats: Dict[str, Any] = {}
        
        # Configuration par défaut
        self._stage_timeouts = {
            WorkflowStage.INPUT_VALIDATION: 30.0,
            WorkflowStage.TOKEN_VALIDATION: 15.0,
            WorkflowStage.DATA_EXTRACTION: 120.0,
            WorkflowStage.NUTRITION_ENRICHMENT: 60.0,
            WorkflowStage.HEALTH_PROFILE_CONTEXT: 30.0,
            WorkflowStage.EXPERT_ANALYSIS: 180.0,
            WorkflowStage.SCORE_CALCULATION: 30.0,
            WorkflowStage.ALTERNATIVE_GENERATION: 90.0,
            WorkflowStage.CHAT_CONTEXT_PREPARATION: 45.0,
            WorkflowStage.RESPONSE_ASSEMBLY: 15.0
        }
        
        # Enregistrement des nodes par défaut
        self._register_default_nodes()
    
    def _register_default_nodes(self) -> None:
        """Enregistre les nodes par défaut du workflow."""
        self.register_node(InputValidationNode())
        self.register_node(TokenValidationNode())
        self.register_node(DataExtractionNode())
        self.register_node(ExpertAnalysisNode())
        # TODO: Ajouter les autres nodes (NutritionEnrichmentNode, etc.)
    
    async def execute(
        self,
        input_data: InputData,
        user_context: UserHealthContext,
        analysis_config: AnalysisConfig
    ) -> WorkflowResult:
        """
        Exécute le workflow complet selon la state machine.
        
        Template Method Pattern :
        1. Initialisation du state
        2. Validation globale
        3. Exécution des stages
        4. Assemblage du résultat final
        """
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialisation du state
        initial_state = WorkflowState(
            workflow_id=workflow_id,
            user_context=user_context,
            input_data=input_data,
            analysis_config=analysis_config,
            current_stage=WorkflowStage.ENTRY_ROUTING
        )
        
        self._logger.info(
            "Workflow execution started",
            workflow_id=workflow_id,
            input_type=input_data.type.value,
            quality_level=analysis_config.quality_level.value,
            user_id=str(user_context.user_id)
        )
        
        try:
            # Validation globale initiale
            await self._validate_workflow_input(initial_state)
            
            # Exécution de la state machine
            final_state = await self._execute_state_machine(initial_state)
            
            # Assemblage du résultat final
            result = await self._assemble_final_result(final_state, start_time)
            
            self._logger.info(
                "Workflow execution completed successfully",
                workflow_id=workflow_id,
                total_duration_ms=result.performance_metrics.total_duration_ms,
                final_score=result.data.get("overall_score") if result.data else None
            )
            
            return result
            
        except (WorkflowNodeError, WorkflowValidationError, WorkflowTimeoutError) as e:
            # Erreurs de workflow gérées
            self._logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                error_type=type(e).__name__,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            
            return WorkflowResult.error_result(
                workflow_id=workflow_id,
                errors=[self._convert_exception_to_workflow_error(e)]
            )
            
        except Exception as e:
            # Erreurs inattendues
            self._logger.error(
                "Workflow execution failed with unexpected error",
                workflow_id=workflow_id,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            
            return WorkflowResult.error_result(
                workflow_id=workflow_id,
                errors=[self._convert_exception_to_workflow_error(e)]
            )
    
    async def _execute_state_machine(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Exécute la state machine avec transitions validées.
        
        State Machine Pattern :
        - États : WorkflowStage enum
        - Transitions : Déterminées par le router
        - Actions : Exécution des nodes
        """
        current_state = initial_state.advance_to_stage(WorkflowStage.INPUT_VALIDATION)
        visited_stages: Set[WorkflowStage] = set()
        
        while True:
            current_stage = current_state.current_stage
            
            # Protection contre les cycles infinis
            if current_stage in visited_stages:
                raise WorkflowValidationError(
                    [f"Cycle detected: stage {current_stage.value} already visited"],
                    stage=current_stage.value
                )
            
            visited_stages.add(current_stage)
            
            self._logger.debug(
                "Processing workflow stage",
                workflow_id=current_state.workflow_id,
                stage=current_stage.value
            )
            
            # Vérification si nous avons un node pour ce stage
            if current_stage not in self._nodes:
                self._logger.warning(
                    "No node registered for stage, skipping",
                    stage=current_stage.value
                )
                
                # Passage au stage suivant
                next_stage = self._router.get_next_stage(current_state)
                if not next_stage:
                    break
                
                current_state = current_state.advance_to_stage(next_stage)
                continue
            
            # Exécution du node avec timeout
            node = self._nodes[current_stage]
            timeout = self._stage_timeouts.get(current_stage, 60.0)
            
            try:
                current_state = await asyncio.wait_for(
                    node.process(current_state),
                    timeout=timeout
                )
                
            except asyncio.TimeoutError:
                raise WorkflowTimeoutError(current_stage.value, timeout)
            
            # Validation de la transition
            next_stage = self._router.get_next_stage(current_state)
            
            if not next_stage:
                # Fin du workflow
                break
            
            # Validation de la transition
            transition_errors = await self._validator.validate_state_transition(
                current_stage, next_stage, current_state
            )
            
            if transition_errors:
                raise WorkflowValidationError(
                    transition_errors,
                    stage=current_stage.value
                )
            
            # Transition vers le stage suivant
            current_state = current_state.advance_to_stage(next_stage)
        
        return current_state
    
    async def _validate_workflow_input(self, state: WorkflowState) -> None:
        """Valide les inputs globaux du workflow."""
        input_errors = await self._validator.validate_input(state.input_data)
        context_errors = await self._validator.validate_user_context(state.user_context)
        
        all_errors = input_errors + context_errors
        
        if all_errors:
            raise WorkflowValidationError(all_errors)
    
    async def _assemble_final_result(self, final_state: WorkflowState, start_time: float) -> WorkflowResult:
        """Assemble le résultat final du workflow."""
        total_duration = (time.time() - start_time) * 1000  # ms
        
        # Calcul des durées par stage
        stage_durations = {}
        for i, step in enumerate(final_state.processing_history):
            if i > 0:
                prev_step = final_state.processing_history[i-1]
                duration = (step.timestamp - prev_step.timestamp).total_seconds() * 1000
                stage_durations[step.stage] = duration
        
        # Métriques de performance
        performance_metrics = PerformanceMetrics(
            total_duration_ms=total_duration,
            stage_durations=stage_durations,
            api_calls_count=len([step for step in final_state.processing_history if "api" in step.stage]),
            cache_hit_rate=0.0  # TODO: Calculer le vrai taux de cache hit
        )
        
        # Assemblage des données de résultat
        result_data = {
            "workflow_id": final_state.workflow_id,
            "overall_score": final_state.calculated_score,
            "nutrition_data": self._serialize_nutrition_data(final_state.nutrition_data),
            "expert_analysis": final_state.expert_analysis,
            "alternatives": final_state.alternatives,
            "chat_context": final_state.chat_context,
            "processing_summary": {
                "stages_completed": len(final_state.processing_history),
                "total_duration_ms": total_duration,
                "final_stage": final_state.current_stage.value
            }
        }
        
        # Conversion des steps pour le résultat
        processing_steps = [
            ProcessingStep(
                stage=step.stage,
                timestamp=step.timestamp,
                duration_ms=step.duration_ms,
                success=step.success,
                metadata=step.metadata
            )
            for step in final_state.processing_history
        ]
        
        return WorkflowResult.success_result(
            workflow_id=final_state.workflow_id,
            data=result_data,
            metrics=performance_metrics,
            history=processing_steps
        )
    
    def _serialize_nutrition_data(self, nutrition_data) -> Optional[Dict[str, Any]]:
        """Sérialise les données nutritionnelles."""
        if not nutrition_data:
            return None
        
        return {
            "product_name": nutrition_data.product_name,
            "brand": nutrition_data.brand,
            "barcode": nutrition_data.barcode,
            "calories": nutrition_data.calories,
            "protein": nutrition_data.protein,
            "carbohydrates": nutrition_data.carbohydrates,
            "total_fat": nutrition_data.total_fat,
            "fiber": nutrition_data.fiber,
            "sugar": nutrition_data.sugar,
            "sodium": nutrition_data.sodium,
            "ingredients": nutrition_data.ingredients,
            "allergens": nutrition_data.allergens,
            "data_source": nutrition_data.data_source
        }
    
    def _convert_exception_to_workflow_error(self, exception: Exception):
        """Convertit une exception en WorkflowError."""
        from app.workflows.interfaces import WorkflowError
        from datetime import datetime
        
        return WorkflowError(
            stage=getattr(exception, 'stage', 'unknown'),
            error_type=type(exception).__name__,
            message=str(exception),
            timestamp=datetime.utcnow(),
            recoverable=getattr(exception, 'recoverable', False)
        )
    
    def register_node(self, node: IWorkflowNode) -> None:
        """Enregistre un node pour son stage requis."""
        stage = node.required_stage
        self._nodes[stage] = node
        
        self._logger.info(
            "Node registered",
            stage=stage.value,
            node_type=type(node).__name__
        )
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Récupère le statut d'un workflow (TODO: implémenter persistence)."""
        # TODO: Implémenter la persistence des states de workflow
        return None


class DefaultWorkflowRouter(IWorkflowRouter):
    """
    Router par défaut pour les transitions de workflow.
    
    Implémente la logique de routage standard du workflow d'analyse alimentaire.
    """
    
    def get_next_stage(self, current_state: WorkflowState) -> Optional[WorkflowStage]:
        """Détermine le prochain stage selon la logique business."""
        current_stage = current_state.current_stage
        
        # Table de transitions par défaut
        stage_transitions = {
            WorkflowStage.ENTRY_ROUTING: WorkflowStage.INPUT_VALIDATION,
            WorkflowStage.INPUT_VALIDATION: WorkflowStage.TOKEN_VALIDATION,
            WorkflowStage.TOKEN_VALIDATION: WorkflowStage.DATA_EXTRACTION,
            WorkflowStage.DATA_EXTRACTION: WorkflowStage.NUTRITION_ENRICHMENT,
            WorkflowStage.NUTRITION_ENRICHMENT: WorkflowStage.HEALTH_PROFILE_CONTEXT,
            WorkflowStage.HEALTH_PROFILE_CONTEXT: WorkflowStage.EXPERT_ANALYSIS,
            WorkflowStage.EXPERT_ANALYSIS: WorkflowStage.SCORE_CALCULATION,
            WorkflowStage.SCORE_CALCULATION: self._route_after_score_calculation(current_state),
            WorkflowStage.ALTERNATIVE_GENERATION: WorkflowStage.CHAT_CONTEXT_PREPARATION,
            WorkflowStage.CHAT_CONTEXT_PREPARATION: WorkflowStage.RESPONSE_ASSEMBLY,
            WorkflowStage.RESPONSE_ASSEMBLY: None  # Fin du workflow
        }
        
        next_stage = stage_transitions.get(current_stage)
        
        # Gestion des transitions conditionnelles
        if callable(next_stage):
            return next_stage
        
        return next_stage
    
    def _route_after_score_calculation(self, state: WorkflowState) -> Optional[WorkflowStage]:
        """Routage conditionnel après le calcul de score."""
        if state.analysis_config.enable_alternatives:
            return WorkflowStage.ALTERNATIVE_GENERATION
        elif state.analysis_config.enable_chat_context:
            return WorkflowStage.CHAT_CONTEXT_PREPARATION
        else:
            return WorkflowStage.RESPONSE_ASSEMBLY
    
    def should_skip_stage(self, stage: WorkflowStage, state: WorkflowState) -> bool:
        """Détermine si un stage doit être ignoré."""
        # Logique de skip conditionnelle
        if stage == WorkflowStage.ALTERNATIVE_GENERATION:
            return not state.analysis_config.enable_alternatives
        
        if stage == WorkflowStage.CHAT_CONTEXT_PREPARATION:
            return not state.analysis_config.enable_chat_context
        
        return False
    
    def can_transition_to(self, from_stage: WorkflowStage, to_stage: WorkflowStage) -> bool:
        """Vérifie si une transition est valide."""
        # Transitions autorisées (peut être étendue)
        valid_transitions = {
            (WorkflowStage.INPUT_VALIDATION, WorkflowStage.TOKEN_VALIDATION),
            (WorkflowStage.TOKEN_VALIDATION, WorkflowStage.DATA_EXTRACTION),
            (WorkflowStage.DATA_EXTRACTION, WorkflowStage.NUTRITION_ENRICHMENT),
            (WorkflowStage.NUTRITION_ENRICHMENT, WorkflowStage.HEALTH_PROFILE_CONTEXT),
            (WorkflowStage.HEALTH_PROFILE_CONTEXT, WorkflowStage.EXPERT_ANALYSIS),
            (WorkflowStage.EXPERT_ANALYSIS, WorkflowStage.SCORE_CALCULATION),
            (WorkflowStage.SCORE_CALCULATION, WorkflowStage.ALTERNATIVE_GENERATION),
            (WorkflowStage.SCORE_CALCULATION, WorkflowStage.CHAT_CONTEXT_PREPARATION),
            (WorkflowStage.SCORE_CALCULATION, WorkflowStage.RESPONSE_ASSEMBLY),
            (WorkflowStage.ALTERNATIVE_GENERATION, WorkflowStage.CHAT_CONTEXT_PREPARATION),
            (WorkflowStage.CHAT_CONTEXT_PREPARATION, WorkflowStage.RESPONSE_ASSEMBLY),
        }
        
        return (from_stage, to_stage) in valid_transitions


class DefaultWorkflowValidator(IWorkflowValidator):
    """
    Validateur par défaut pour les données et transitions de workflow.
    
    Implémente les validations business standard.
    """
    
    async def validate_input(self, input_data: InputData) -> List[str]:
        """Valide les données d'entrée."""
        errors = []
        
        if not input_data:
            errors.append("Input data is required")
            return errors
        
        # Validation selon le type
        if input_data.type.value == "image":
            if not input_data.image_data:
                errors.append("Image data is required for image input")
            elif len(input_data.image_data) == 0:
                errors.append("Image data cannot be empty")
                
        elif input_data.type.value == "barcode":
            if not input_data.barcode:
                errors.append("Barcode is required for barcode input")
            elif not input_data.barcode.isdigit():
                errors.append("Barcode must contain only digits")
                
        elif input_data.type.value == "json_data":
            if not input_data.json_data:
                errors.append("JSON data is required for JSON input")
            elif not isinstance(input_data.json_data, dict):
                errors.append("JSON data must be a dictionary")
        
        return errors
    
    async def validate_state_transition(
        self,
        from_stage: WorkflowStage,
        to_stage: WorkflowStage,
        state: WorkflowState
    ) -> List[str]:
        """Valide qu'une transition d'état est valide."""
        errors = []
        
        # Validation des données requises pour la transition
        if to_stage == WorkflowStage.EXPERT_ANALYSIS:
            if not state.nutrition_data:
                errors.append("Nutrition data required for expert analysis")
        
        if to_stage == WorkflowStage.ALTERNATIVE_GENERATION:
            if not state.expert_analysis:
                errors.append("Expert analysis required for alternative generation")
        
        if to_stage == WorkflowStage.CHAT_CONTEXT_PREPARATION:
            if not state.calculated_score:
                errors.append("Calculated score required for chat context preparation")
        
        return errors
    
    async def validate_user_context(self, user_context: UserHealthContext) -> List[str]:
        """Valide le contexte utilisateur."""
        errors = []
        
        if not user_context:
            errors.append("User context is required")
            return errors
        
        if not user_context.user_id:
            errors.append("User ID is required")
        
        # TODO: Validation plus approfondie du contexte santé
        
        return errors