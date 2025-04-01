#!/usr/bin/env python3
"""
Register Synthesis with Hermes

This script registers the Synthesis component with the Hermes service registry,
allowing other components to discover and use its capabilities.
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

# Add parent directory to path to allow importing Tekton core
script_dir = os.path.dirname(os.path.abspath(__file__))
synthesis_dir = os.path.abspath(os.path.join(script_dir, "../.."))
tekton_dir = os.path.abspath(os.path.join(synthesis_dir, ".."))
tekton_core_dir = os.path.join(tekton_dir, "tekton-core")

# Add to Python path
sys.path.insert(0, synthesis_dir)
sys.path.insert(0, tekton_dir)
sys.path.insert(0, tekton_core_dir)

# Determine if we're in a virtual environment
in_venv = sys.prefix != sys.base_prefix
if not in_venv:
    logger.warning("Not running in a virtual environment. Consider activating the Synthesis venv.")

# Try to import startup instructions and heartbeat monitor
try:
    from tekton.core.component_registration import StartUpInstructions, ComponentRegistration
    from tekton.core.heartbeat_monitor import ComponentHeartbeat
    HAS_CORE_MODULES = True
except ImportError:
    logger.error("Failed to import Tekton core modules. Make sure tekton-core is properly installed.")
    # Fallback to direct registration without StartUpInstructions
    HAS_CORE_MODULES = False


async def register_with_hermes(instructions_file: Optional[str] = None, hermes_url: Optional[str] = None):
    """
    Register Synthesis with Hermes.
    
    Args:
        instructions_file: Path to StartUpInstructions JSON file
        hermes_url: URL of Hermes API
    """
    try:
        import aiohttp
        
        # Check for StartUpInstructions
        if HAS_CORE_MODULES and instructions_file and os.path.isfile(instructions_file):
            logger.info(f"Loading startup instructions from {instructions_file}")
            instructions = StartUpInstructions.from_file(instructions_file)
            capabilities = instructions.capabilities
            metadata = instructions.metadata
            hermes_url = instructions.hermes_url
            component_id = instructions.component_id
        else:
            # Use default values
            component_id = "synthesis.execution"
            hermes_url = hermes_url or os.environ.get("HERMES_URL", "http://localhost:5000/api")
            
            # Define Synthesis capabilities
            capabilities = [
                {
                    "name": "execute_plan",
                    "description": "Execute a plan",
                    "parameters": {
                        "plan": "object",
                        "execution_context": "object (optional)"
                    }
                },
                {
                    "name": "get_execution_status",
                    "description": "Get the status of an execution",
                    "parameters": {
                        "execution_id": "string"
                    }
                },
                {
                    "name": "get_execution_result",
                    "description": "Get the result of an execution",
                    "parameters": {
                        "execution_id": "string"
                    }
                },
                {
                    "name": "cancel_execution",
                    "description": "Cancel an execution",
                    "parameters": {
                        "execution_id": "string"
                    }
                }
            ]
            
            # Default metadata
            metadata = {
                "description": "Execution engine for implementing plans",
                "version": "0.1.0",
                "dependencies": ["prometheus.core"]
            }
        
        # Set up heartbeat monitoring if available
        if HAS_CORE_MODULES:
            # Create component heartbeat
            heartbeat = ComponentHeartbeat(
                component_id=component_id,
                component_name="Synthesis",
                hermes_url=hermes_url,
                capabilities=capabilities,
                metadata=metadata
            )
            
            # Start heartbeat (this will also handle registration)
            result = await heartbeat.start()
            
            if result:
                logger.info(f"Successfully registered Synthesis ({component_id}) with Hermes and started heartbeat monitoring")
                
                # Keep the script running to maintain heartbeat
                stop_event = asyncio.Event()
                
                # Set up signal handlers
                loop = asyncio.get_event_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(
                        sig,
                        lambda s=sig: asyncio.create_task(shutdown(s, heartbeat, stop_event))
                    )
                
                logger.info("Press Ctrl+C to stop")
                await stop_event.wait()
                return True
            else:
                logger.error(f"Failed to register Synthesis with Hermes")
                return False
        
        # Fallback to direct registration via API
        else:
            logger.info(f"Registering Synthesis with Hermes at {hermes_url}")
            
            # Define services
            services = [
                {
                    "service_id": component_id,
                    "name": "Synthesis Execution Engine",
                    "version": "0.1.0",
                    "endpoint": "http://localhost:5005/api/execution",
                    "capabilities": capabilities,
                    "metadata": metadata
                },
                {
                    "service_id": "synthesis.phase_management",
                    "name": "Synthesis Phase Manager",
                    "version": "0.1.0",
                    "endpoint": "http://localhost:5005/api/phases",
                    "capabilities": [
                        {
                            "name": "get_phase_status",
                            "description": "Get the status of a phase",
                            "parameters": {
                                "phase_id": "string"
                            }
                        },
                        {
                            "name": "get_execution_phases",
                            "description": "Get all phases for an execution",
                            "parameters": {
                                "execution_id": "string"
                            }
                        }
                    ],
                    "metadata": {
                        "description": "Phase management for execution engine",
                        "version": "0.1.0",
                        "parent_component": "synthesis.execution"
                    }
                }
            ]
            
            # Register each service
            async with aiohttp.ClientSession() as session:
                for service in services:
                    logger.info(f"Registering service: {service['service_id']}")
                    
                    async with session.post(
                        f"{hermes_url}/registration/register",
                        json=service
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully registered service {service['service_id']}: {data}")
                        else:
                            error = await response.text()
                            logger.error(f"Failed to register service {service['service_id']}: {error}")
                
                # Send heartbeat to indicate component is alive
                async with session.post(
                    f"{hermes_url}/registration/heartbeat",
                    json={"component": "synthesis", "status": "active"}
                ) as response:
                    if response.status == 200:
                        logger.info("Sent heartbeat to Hermes")
                    else:
                        logger.warning("Failed to send heartbeat to Hermes")
                
                # Warn that heartbeat monitoring is not available
                logger.warning("Heartbeat monitoring not available. Hermes may not detect if this component becomes unavailable.")
                return True
                    
    except Exception as e:
        logger.exception(f"Error registering services with Hermes: {e}")
        return False


async def shutdown(sig, heartbeat, stop_event):
    """
    Handle shutdown signal.
    
    Args:
        sig: Signal that triggered shutdown
        heartbeat: ComponentHeartbeat instance
        stop_event: Event to signal main loop to stop
    """
    logger.info(f"Received signal {sig.name}, shutting down")
    await heartbeat.stop()
    stop_event.set()


async def main():
    """Main entry point."""
    logger.info("Registering Synthesis with Hermes...")
    
    # Check for StartUpInstructions file from environment
    instructions_file = os.environ.get("STARTUP_INSTRUCTIONS_FILE")
    hermes_url = os.environ.get("HERMES_URL", "http://localhost:5000/api")
    
    await register_with_hermes(instructions_file, hermes_url)
    
    logger.info("Registration complete. Synthesis services are now available to other components.")


if __name__ == "__main__":
    asyncio.run(main())