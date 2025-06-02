#!/usr/bin/env python3
"""
Synthesis API Server

This module implements the API server for the Synthesis component,
following the Single Port Architecture pattern.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

import fastapi
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add Tekton root to path if not already present
tekton_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if tekton_root not in sys.path:
    sys.path.insert(0, tekton_root)

# Import shared utilities
from shared.utils.hermes_registration import HermesRegistration, heartbeat_loop
from shared.utils.logging_setup import setup_component_logging
from shared.utils.env_config import get_component_config
from shared.utils.errors import StartupError
from shared.utils.startup import component_startup, StartupMetrics
from shared.utils.shutdown import GracefulShutdown

# Import Tekton utilities
from tekton.utils.tekton_errors import (
    TektonError, 
    ConfigurationError,
    ConnectionError,
    AuthenticationError
)
from tekton.utils.tekton_lifecycle import TektonLifecycle

# Import Synthesis components
from synthesis.core.execution_engine import ExecutionEngine
from synthesis.core.execution_models import (
    ExecutionStage, ExecutionStatus, ExecutionPriority,
    ExecutionResult, ExecutionPlan, ExecutionContext
)
from synthesis.core.events import EventManager, WebSocketManager

# Set up logging
logger = setup_component_logging("synthesis")


# API Data Models
class APIResponse(BaseModel):
    """Generic API response model."""
    status: str = "success"
    message: Optional[str] = None
    data: Optional[Any] = None
    errors: Optional[List[str]] = None


class ExecutionRequest(BaseModel):
    """Execution request model."""
    plan: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    priority: Optional[int] = Field(default=ExecutionPriority.NORMAL)
    wait_for_completion: Optional[bool] = Field(default=False)
    timeout: Optional[int] = Field(default=None)


class ExecutionResponse(BaseModel):
    """Execution response model."""
    execution_id: str
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


class VariableRequest(BaseModel):
    """Variable request model."""
    operation: str  # set, delete, increment, append, merge
    name: str
    value: Optional[Any] = None


class SynthesisComponent(TektonLifecycle):
    """
    Synthesis component lifecycle manager.
    
    This class manages the lifecycle of the Synthesis component,
    including initialization, startup, shutdown, and health checking.
    """
    
    async def initialize(self) -> bool:
        """
        Initialize the Synthesis component.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Create execution engine
            self.execution_engine = ExecutionEngine()
            self.track_resource(self.execution_engine)
            
            # Create event manager
            self.event_manager = EventManager.get_instance()
            
            # Create WebSocket manager
            self.ws_manager = WebSocketManager()
            
            logger.info("Synthesis component initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Synthesis component: {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the Synthesis component.
        
        Returns:
            Health check information
        """
        health_info = await super().health_check()
        
        # Add component-specific health info
        if hasattr(self, "execution_engine"):
            health_info.update({
                "active_executions": len(self.execution_engine.active_executions),
                "execution_capacity": self.execution_engine.max_concurrent_executions,
                "execution_load": len(self.execution_engine.active_executions) / self.execution_engine.max_concurrent_executions
            })
        
        return health_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for Synthesis"""
    # Startup
    logger.info("Starting Synthesis API server...")
    
    # Get configuration
    config = get_component_config()
    port = config.synthesis.port if hasattr(config, 'synthesis') else int(os.environ.get("SYNTHESIS_PORT", 8009))
    
    try:
        # Create and initialize component
        component = SynthesisComponent(
            component_id="synthesis",
            component_name="Synthesis",
            component_type="execution_engine",
            version="1.0.0",
            description="Execution and integration engine for Tekton",
            hermes_registration=False  # We'll handle this manually
        )
        
        # Initialize component
        success = await component.initialize()
        if not success:
            logger.error("Failed to initialize Synthesis component")
            raise StartupError("Failed to initialize Synthesis component")
        
        # Store component in app state
        app.state.component = component
        
        # Register with Hermes
        hermes_registration = HermesRegistration()
        await hermes_registration.register_component(
            component_name="synthesis",
            port=port,
            version="1.0.0",
            capabilities=["code_generation", "integration", "execution"],
            metadata={
                "description": "Execution and integration engine",
                "category": "execution"
            }
        )
        app.state.hermes_registration = hermes_registration
        
        # Start heartbeat task
        if hermes_registration.is_registered:
            heartbeat_task = asyncio.create_task(heartbeat_loop(hermes_registration, "synthesis"))
        
        logger.info(f"Synthesis API server started on port {port}")
        
    except Exception as e:
        logger.error(f"Error starting Synthesis API server: {e}")
        raise StartupError(f"Failed to start Synthesis: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Synthesis API server...")
    
    # Cancel heartbeat task if running
    if hermes_registration.is_registered and 'heartbeat_task' in locals():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    
    try:
        # Shut down component
        if hasattr(app.state, "component") and app.state.component:
            await app.state.component.shutdown()
            logger.info("Synthesis component shut down successfully")
            
    except Exception as e:
        logger.error(f"Error shutting down Synthesis: {e}")
    
    # Deregister from Hermes
    if hasattr(app.state, "hermes_registration") and app.state.hermes_registration:
        await app.state.hermes_registration.deregister("synthesis")
    
    # Give sockets time to close on macOS
    await asyncio.sleep(0.5)
    
    logger.info("Synthesis API server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Synthesis Execution Engine API",
    description="API for the Synthesis execution and integration engine",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create routers for path-based routing
router = APIRouter(prefix="/api")
ws_router = APIRouter()
metrics_router = APIRouter()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint for the Synthesis API."""
    from tekton.utils.port_config import get_synthesis_port
    port = get_synthesis_port()
    
    return {
        "name": "Synthesis Execution Engine API",
        "version": "1.0.0",
        "status": "running",
        "documentation": f"http://localhost:{port}/docs"
    }

# Import and include MCP router
from synthesis.api.fastmcp_endpoints import mcp_router

# Include routers in app
app.include_router(router)
app.include_router(ws_router)
app.include_router(metrics_router)
app.include_router(mcp_router)

# Dependency to get the execution engine
async def get_execution_engine():
    """Get the execution engine or raise an error if not initialized."""
    if not hasattr(app.state, "component") or not app.state.component:
        raise HTTPException(status_code=503, detail="Synthesis component not initialized")
    if not hasattr(app.state.component, "execution_engine") or not app.state.component.execution_engine:
        raise HTTPException(status_code=503, detail="Execution engine not initialized")
    return app.state.component.execution_engine


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health of the Synthesis component following Tekton standards."""
    # Get port from config
    config = get_component_config()
    port = config.synthesis.port if hasattr(config, 'synthesis') else int(os.environ.get("SYNTHESIS_PORT", 8009))
    
    # Try to get component health info
    # Even if the component isn't fully initialized, we'll return a basic health response
    try:
        if hasattr(app.state, "component") and app.state.component:
            # Component exists, get proper health info
            health_info = await app.state.component.health_check()
            
            # Set status code based on health
            status_code = 200
            if health_info.get("status") == "error":
                status_code = 500
                health_status = "error"
            elif health_info.get("status") == "degraded":
                status_code = 429
                health_status = "degraded"
            else:
                health_status = "healthy"
                
            # Message from component health
            message = health_info.get("message", "Synthesis is running")
        else:
            # Component not initialized but app is responding
            status_code = 200
            health_status = "healthy"
            message = "Synthesis API is running (component not fully initialized)"
    except Exception as e:
        # Something went wrong in health check
        status_code = 200
        health_status = "healthy"
        message = f"Synthesis API is running (basic health check only)"
        logger.warning(f"Error in health check, using basic response: {e}")
    
    # Format response according to Tekton standards
    standardized_health = {
        "status": health_status,
        "component": "synthesis",
        "version": "1.0.0",
        "port": port,
        "message": message
    }
    
    return JSONResponse(
        content=standardized_health,
        status_code=status_code
    )


# Metrics endpoint
@metrics_router.get("/metrics")
async def metrics():
    """Get metrics from the Synthesis component."""
    if not hasattr(app.state, "component") or not app.state.component:
        return JSONResponse(
            content={"status": "unavailable", "message": "Synthesis component not initialized"},
            status_code=503
        )
    
    # Get execution engine
    execution_engine = await get_execution_engine()
    
    # Collect metrics
    metrics = {
        "active_executions": len(execution_engine.active_executions),
        "execution_capacity": execution_engine.max_concurrent_executions,
        "execution_load": len(execution_engine.active_executions) / execution_engine.max_concurrent_executions,
        "total_executions": len(execution_engine.execution_history),
        "timestamp": time.time()
    }
    
    return JSONResponse(
        content=metrics,
        status_code=200
    )


# WebSocket endpoints
@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time execution updates.
    """
    # Get WebSocket manager
    if not hasattr(app.state, "component") or not app.state.component or not hasattr(app.state.component, "ws_manager"):
        await websocket.close(code=1011, reason="Synthesis component not initialized")
        return
    
    ws_manager = app.state.component.ws_manager
    
    # Accept connection
    await websocket.accept()
    
    # Register connection
    client_id = await ws_manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "timestamp": time.time()
        })
        
        # Handle incoming messages
        while True:
            message = await websocket.receive_json()
            await ws_manager.process_message(websocket, message)
            
    except WebSocketDisconnect:
        # Handle client disconnect
        ws_manager.disconnect(client_id)
        
    except Exception as e:
        # Handle other errors
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(client_id)


# Execution endpoints
@router.post("/executions", response_model=ExecutionResponse)
async def create_execution(request: ExecutionRequest, execution_engine = Depends(get_execution_engine)):
    """
    Create a new execution.
    """
    try:
        # Create execution plan
        plan = ExecutionPlan.from_dict(request.plan) if isinstance(request.plan, dict) else request.plan
        
        # Create execution context if provided
        context = None
        if request.context:
            context = ExecutionContext.from_dict(request.context)
        
        # Execute plan
        execution_id = await execution_engine.execute_plan(plan, context)
        
        # Get initial status
        status = await execution_engine.get_execution_status(execution_id)
        
        # Wait for completion if requested
        if request.wait_for_completion:
            timeout = request.timeout or 600  # Default: 10 minutes
            start_time = time.time()
            
            while status["status"] in (ExecutionStatus.PENDING, ExecutionStatus.IN_PROGRESS):
                # Check timeout
                if time.time() - start_time > timeout:
                    return ExecutionResponse(
                        execution_id=execution_id,
                        status="timeout",
                        message=f"Execution timed out after {timeout} seconds",
                        data=status
                    )
                
                # Wait a bit
                await asyncio.sleep(0.5)
                
                # Update status
                status = await execution_engine.get_execution_status(execution_id)
        
        return ExecutionResponse(
            execution_id=execution_id,
            status=status["status"],
            message=f"Execution {'completed' if status['status'] == ExecutionStatus.COMPLETED else 'started'}",
            data=status
        )
        
    except Exception as e:
        logger.error(f"Error creating execution: {e}")
        return ExecutionResponse(
            execution_id=str(uuid.uuid4()),
            status="error",
            message="Error creating execution",
            errors=[str(e)]
        )


@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(execution_id: str, execution_engine = Depends(get_execution_engine)):
    """
    Get execution details by ID.
    """
    try:
        # Get execution status
        status = await execution_engine.get_execution_status(execution_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return ExecutionResponse(
            execution_id=execution_id,
            status=status["status"],
            data=status
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error getting execution {execution_id}: {e}")
        return ExecutionResponse(
            execution_id=execution_id,
            status="error",
            message=f"Error getting execution {execution_id}",
            errors=[str(e)]
        )


@router.get("/executions", response_model=APIResponse)
async def list_executions(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of executions to return"),
    execution_engine = Depends(get_execution_engine)
):
    """
    List executions.
    """
    try:
        # Get active executions
        active_executions = execution_engine.active_executions
        
        # Get execution history
        history = execution_engine.execution_history
        
        # Combine and filter results
        all_executions = []
        
        # Add active executions
        for execution_id, execution in active_executions.items():
            if status is None or execution["status"] == status:
                all_executions.append(execution)
        
        # Add historical executions (if not already included in active)
        for execution_id, execution in history.items():
            if execution_id not in active_executions and (status is None or execution.get("status") == status):
                all_executions.append({
                    "context_id": execution_id,
                    "plan_id": execution.get("plan_id"),
                    "status": execution.get("status"),
                    "start_time": execution.get("start_time"),
                    "end_time": execution.get("end_time")
                })
        
        # Sort by start time (descending)
        all_executions.sort(key=lambda x: x.get("start_time", 0), reverse=True)
        
        # Limit results
        all_executions = all_executions[:limit]
        
        return APIResponse(
            status="success",
            data={
                "executions": all_executions,
                "count": len(all_executions)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        return APIResponse(
            status="error",
            message="Error listing executions",
            errors=[str(e)]
        )


@router.post("/executions/{execution_id}/cancel", response_model=APIResponse)
async def cancel_execution(execution_id: str, execution_engine = Depends(get_execution_engine)):
    """
    Cancel an execution.
    """
    try:
        # Cancel execution
        success = await execution_engine.cancel_execution(execution_id)
        
        if not success:
            return APIResponse(
                status="error",
                message=f"Failed to cancel execution {execution_id}",
                errors=[f"Execution {execution_id} not found or already completed"]
            )
        
        return APIResponse(
            status="success",
            message=f"Execution {execution_id} cancelled successfully"
        )
        
    except Exception as e:
        logger.error(f"Error cancelling execution {execution_id}: {e}")
        return APIResponse(
            status="error",
            message=f"Error cancelling execution {execution_id}",
            errors=[str(e)]
        )


@router.get("/executions/{execution_id}/results", response_model=APIResponse)
async def get_execution_results(execution_id: str, execution_engine = Depends(get_execution_engine)):
    """
    Get execution results.
    """
    try:
        # Get execution status
        status = await execution_engine.get_execution_status(execution_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        # Get results from history
        if execution_id in execution_engine.execution_history:
            execution = execution_engine.execution_history[execution_id]
            results = execution.get("results", [])
            
            return APIResponse(
                status="success",
                data={
                    "execution_id": execution_id,
                    "status": status["status"],
                    "results": results
                }
            )
        else:
            return APIResponse(
                status="error",
                message=f"Results not found for execution {execution_id}",
                errors=[f"Results not found for execution {execution_id}"]
            )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error getting execution results {execution_id}: {e}")
        return APIResponse(
            status="error",
            message=f"Error getting execution results {execution_id}",
            errors=[str(e)]
        )


@router.post("/executions/{execution_id}/variables", response_model=APIResponse)
async def update_execution_variables(
    execution_id: str,
    request: VariableRequest,
    execution_engine = Depends(get_execution_engine)
):
    """
    Update execution variables.
    """
    try:
        # Get execution status
        status = await execution_engine.get_execution_status(execution_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        # Get execution context
        if execution_id in execution_engine.active_executions:
            # Get from active executions
            for current_context in execution_engine.active_contexts:
                if current_context.context_id == execution_id:
                    context = current_context
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Context not found for active execution {execution_id}")
        else:
            # Cannot update variables for completed executions
            raise HTTPException(status_code=400, detail=f"Cannot update variables for inactive execution {execution_id}")
        
        # Update variable based on operation
        if request.operation == "set":
            context.variables[request.name] = request.value
            message = f"Variable {request.name} set to {request.value}"
            
        elif request.operation == "delete":
            if request.name in context.variables:
                del context.variables[request.name]
                message = f"Variable {request.name} deleted"
            else:
                message = f"Variable {request.name} not found"
                
        elif request.operation == "increment":
            if request.name not in context.variables:
                context.variables[request.name] = 0
                
            if not isinstance(context.variables[request.name], (int, float)):
                raise HTTPException(status_code=400, detail=f"Variable {request.name} is not a number")
                
            increment = request.value if request.value is not None else 1
            context.variables[request.name] += increment
            message = f"Variable {request.name} incremented to {context.variables[request.name]}"
            
        elif request.operation == "append":
            if request.name not in context.variables:
                context.variables[request.name] = []
                
            if not isinstance(context.variables[request.name], list):
                context.variables[request.name] = [context.variables[request.name]]
                
            context.variables[request.name].append(request.value)
            message = f"Value appended to variable {request.name}"
            
        elif request.operation == "merge":
            if request.name not in context.variables:
                context.variables[request.name] = {}
                
            if not isinstance(context.variables[request.name], dict):
                raise HTTPException(status_code=400, detail=f"Variable {request.name} is not a dictionary")
                
            if not isinstance(request.value, dict):
                raise HTTPException(status_code=400, detail=f"Value must be a dictionary for merge operation")
                
            context.variables[request.name].update(request.value)
            message = f"Dictionary merged into variable {request.name}"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")
        
        return APIResponse(
            status="success",
            message=message,
            data={
                "name": request.name,
                "value": context.variables.get(request.name),
                "operation": request.operation
            }
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error updating execution variables for {execution_id}: {e}")
        return APIResponse(
            status="error",
            message=f"Error updating execution variables for {execution_id}",
            errors=[str(e)]
        )


# Function registry endpoints
@router.post("/functions/{function_name}", response_model=APIResponse)
async def register_function(
    function_name: str,
    function_code: str,
    execution_engine = Depends(get_execution_engine)
):
    """
    Register a function in the execution engine.
    
    Note: This endpoint is disabled in production for security reasons.
    """
    raise HTTPException(status_code=501, detail="Function registration via API is disabled in production")


@router.get("/functions", response_model=APIResponse)
async def list_functions(execution_engine = Depends(get_execution_engine)):
    """
    List registered functions.
    """
    try:
        # Get function names from registry
        function_names = list(execution_engine.function_registry.keys())
        
        return APIResponse(
            status="success",
            data={
                "functions": function_names,
                "count": len(function_names)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing functions: {e}")
        return APIResponse(
            status="error",
            message="Error listing functions",
            errors=[str(e)]
        )


# Events endpoints
@router.get("/events", response_model=APIResponse)
async def list_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, description="Maximum number of events to return")
):
    """
    List recent events.
    """
    try:
        # Get event manager
        if not hasattr(app.state, "component") or not app.state.component:
            raise HTTPException(status_code=503, detail="Synthesis component not initialized")
            
        event_manager = EventManager.get_instance()
        
        # Get recent events
        events = event_manager.get_recent_events(event_type, limit)
        
        return APIResponse(
            status="success",
            data={
                "events": events,
                "count": len(events)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        return APIResponse(
            status="error",
            message="Error listing events",
            errors=[str(e)]
        )


@router.post("/events", response_model=APIResponse)
async def emit_event(
    event_type: str,
    event_data: Dict[str, Any]
):
    """
    Emit a custom event.
    """
    try:
        # Get event manager
        if not hasattr(app.state, "component") or not app.state.component:
            raise HTTPException(status_code=503, detail="Synthesis component not initialized")
            
        event_manager = EventManager.get_instance()
        
        # Emit event
        event_id = await event_manager.emit(event_type, event_data)
        
        return APIResponse(
            status="success",
            message=f"Event {event_type} emitted successfully",
            data={"event_id": event_id}
        )
        
    except Exception as e:
        logger.error(f"Error emitting event: {e}")
        return APIResponse(
            status="error",
            message="Error emitting event",
            errors=[str(e)]
        )


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get port configuration
    config = get_component_config()
    port = config.synthesis.port if hasattr(config, 'synthesis') else int(os.environ.get("SYNTHESIS_PORT", 8009))
    
    uvicorn.run(app, host="0.0.0.0", port=port)