#!/usr/bin/env python3
"""
Register Synthesis with Hermes

This script registers the Synthesis component with Hermes for service discovery
and component integration.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("synthesis.scripts.register_with_hermes")

# Add parent directory to path to allow importing project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
synthesis_dir = os.path.abspath(os.path.join(script_dir, "../.."))
tekton_dir = os.path.abspath(os.path.join(synthesis_dir, ".."))
sys.path.insert(0, synthesis_dir)
sys.path.insert(0, tekton_dir)

# Import Tekton utilities
from tekton.utils.tekton_config import get_component_port
from tekton.utils.tekton_registration import (
    TektonComponent, 
    StandardCapabilities,
    ComponentStatus
)


async def register_synthesis():
    """
    Register Synthesis with Hermes.
    
    Returns:
        True if registration was successful
    """
    try:
        # Get Hermes URL from environment or use default
        hermes_url = os.environ.get("HERMES_URL", f"http://localhost:{get_component_port('hermes')}/api")
        
        # Get Synthesis port
        synthesis_port = get_component_port("synthesis")
        
        # Create component definition
        component = TektonComponent(
            component_id="synthesis",
            component_name="Synthesis",
            component_type="execution_engine",
            version="1.0.0",
            description="Execution and integration engine for Tekton",
            base_url=f"http://localhost:{synthesis_port}",
            capabilities=[
                StandardCapabilities.execution_engine(),
                {
                    "name": "execute_plan",
                    "description": "Execute a plan",
                    "parameters": {
                        "plan": "object",
                        "context": "object (optional)",
                        "wait_for_completion": "boolean (optional)",
                        "timeout": "integer (optional)"
                    },
                    "endpoint": f"http://localhost:{synthesis_port}/api/executions"
                },
                {
                    "name": "get_execution_status",
                    "description": "Get the status of an execution",
                    "parameters": {
                        "execution_id": "string"
                    },
                    "endpoint": f"http://localhost:{synthesis_port}/api/executions/{{execution_id}}"
                },
                {
                    "name": "cancel_execution",
                    "description": "Cancel an execution",
                    "parameters": {
                        "execution_id": "string"
                    },
                    "endpoint": f"http://localhost:{synthesis_port}/api/executions/{{execution_id}}/cancel"
                },
                {
                    "name": "list_executions",
                    "description": "List executions",
                    "parameters": {
                        "status": "string (optional)",
                        "limit": "integer (optional)"
                    },
                    "endpoint": f"http://localhost:{synthesis_port}/api/executions"
                },
                {
                    "name": "update_variables",
                    "description": "Update execution variables",
                    "parameters": {
                        "execution_id": "string",
                        "operation": "string",
                        "name": "string",
                        "value": "any (optional)"
                    },
                    "endpoint": f"http://localhost:{synthesis_port}/api/executions/{{execution_id}}/variables"
                }
            ],
            dependencies=[
                "prometheus.core",
                "engram.memory",
                "rhetor.llm"
            ],
            health_endpoint=f"http://localhost:{synthesis_port}/health"
        )
        
        # Register with Hermes
        await component.register()
        
        # Update status to ready
        await component.update_status(ComponentStatus.READY)
        
        logger.info(f"Synthesis registered with Hermes at {hermes_url}")
        
        # Return the component for further management
        return component
        
    except Exception as e:
        logger.error(f"Error registering Synthesis with Hermes: {e}")
        return None


async def run_with_heartbeat():
    """
    Run the registration process with heartbeat management.
    """
    # Register component
    component = await register_synthesis()
    if not component:
        return False
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    logger.info("Synthesis registration active. Press Ctrl+C to stop.")
    
    try:
        # Run heartbeat until stopped
        while not stop_event.is_set():
            # Send heartbeat every 30 seconds
            await component.send_heartbeat()
            await asyncio.sleep(30)
            
    except asyncio.CancelledError:
        pass
        
    finally:
        # Unregister component on shutdown
        await component.unregister()
        logger.info("Synthesis unregistered from Hermes")
    
    return True


def main():
    """Main entry point."""
    return asyncio.run(run_with_heartbeat())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)