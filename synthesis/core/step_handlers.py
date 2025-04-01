#!/usr/bin/env python3
"""
Synthesis Step Handlers

This module handles execution of different step types for Synthesis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable

from synthesis.core.execution_models import ExecutionContext, ExecutionResult
from synthesis.core.condition_evaluator import evaluate_condition
from synthesis.core.loop_handlers import handle_loop_step

# Configure logging
logger = logging.getLogger("synthesis.core.step_handlers")


async def handle_command_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """
    Handle a command execution step.
    
    Args:
        parameters: Step parameters
        context: Execution context
        
    Returns:
        ExecutionResult with command output
    """
    import subprocess
    
    # Get command and arguments
    command = parameters.get("command")
    if not command:
        return ExecutionResult(
            success=False,
            message="No command specified",
            errors=["No command specified"]
        )
        
    # Get shell flag
    shell = parameters.get("shell", True)
    
    # Get working directory
    cwd = parameters.get("cwd")
    
    # Get environment variables
    env = parameters.get("env")
    
    # Get timeout
    timeout = parameters.get("timeout", 60)
    
    try:
        # Execute command
        logger.info(f"Executing command: {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=shell,
            cwd=cwd,
            env=env
        )
        
        # Wait for command to complete with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            # Decode output
            stdout_str = stdout.decode("utf-8").strip() if stdout else ""
            stderr_str = stderr.decode("utf-8").strip() if stderr else ""
            
            # Check return code
            if process.returncode == 0:
                return ExecutionResult(
                    success=True,
                    data={
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "return_code": process.returncode
                    },
                    message=f"Command executed successfully"
                )
            else:
                return ExecutionResult(
                    success=False,
                    data={
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "return_code": process.returncode
                    },
                    message=f"Command failed with return code {process.returncode}",
                    errors=[f"Command failed with return code {process.returncode}", stderr_str]
                )
                
        except asyncio.TimeoutError:
            # Kill the process on timeout
            process.kill()
            return ExecutionResult(
                success=False,
                message=f"Command timed out after {timeout}s",
                errors=[f"Command timed out after {timeout}s"]
            )
            
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error executing command: {e}",
            errors=[str(e)]
        )
        

async def handle_function_step(parameters: Dict[str, Any], 
                           context: ExecutionContext, 
                           function_registry: Optional[Dict[str, Callable]] = None) -> ExecutionResult:
    """
    Handle a function execution step.
    
    Args:
        parameters: Step parameters
        context: Execution context
        function_registry: Optional function registry
        
    Returns:
        ExecutionResult with function output
    """
    # Get function reference
    function_name = parameters.get("function")
    if not function_name:
        return ExecutionResult(
            success=False,
            message="No function specified",
            errors=["No function specified"]
        )
        
    # Get function arguments
    args = parameters.get("args", [])
    kwargs = parameters.get("kwargs", {})
    
    # Add context to kwargs if required
    if parameters.get("include_context", False):
        kwargs["context"] = context
        
    try:
        # Get function reference
        if function_registry:
            function = function_registry.get(function_name)
            if not function:
                return ExecutionResult(
                    success=False,
                    message=f"Function {function_name} not found",
                    errors=[f"Function {function_name} not found"]
                )
        else:
            return ExecutionResult(
                success=False,
                message="Function registry not available",
                errors=["Function registry not available"]
            )
            
        # Execute function
        logger.info(f"Executing function: {function_name}")
        result = function(*args, **kwargs)
        
        # Handle async functions
        if asyncio.iscoroutine(result):
            result = await result
            
        return ExecutionResult(
            success=True,
            data={"result": result},
            message=f"Function executed successfully"
        )
        
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error executing function: {e}",
            errors=[str(e)]
        )
        

async def handle_api_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """
    Handle an API request step.
    
    Args:
        parameters: Step parameters
        context: Execution context
        
    Returns:
        ExecutionResult with API response
    """
    try:
        import aiohttp
        
        # Get request parameters
        url = parameters.get("url")
        if not url:
            return ExecutionResult(
                success=False,
                message="No URL specified",
                errors=["No URL specified"]
            )
            
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers", {})
        params = parameters.get("params", {})
        data = parameters.get("data")
        json_data = parameters.get("json")
        timeout = parameters.get("timeout", 30)
        
        # Execute request
        logger.info(f"Executing API request: {method} {url}")
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                # Read response
                status = response.status
                response_text = await response.text()
                
                # Try to parse as JSON
                try:
                    import json
                    response_data = json.loads(response_text)
                except:
                    response_data = response_text
                    
                # Check if request was successful
                success = 200 <= status < 400
                
                return ExecutionResult(
                    success=success,
                    data={
                        "status": status,
                        "headers": dict(response.headers),
                        "data": response_data
                    },
                    message=f"API request {'succeeded' if success else 'failed'} with status {status}",
                    errors=[f"API request failed with status {status}"] if not success else None
                )
                
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error executing API request: {e}",
            errors=[str(e)]
        )
        

async def handle_condition_step(parameters: Dict[str, Any], 
                             context: ExecutionContext, 
                             execute_step_callback) -> ExecutionResult:
    """
    Handle a conditional execution step.
    
    Args:
        parameters: Step parameters
        context: Execution context
        execute_step_callback: Callback to execute a step
        
    Returns:
        ExecutionResult with condition result
    """
    # Get condition
    condition = parameters.get("condition")
    if not condition:
        return ExecutionResult(
            success=False,
            message="No condition specified",
            errors=["No condition specified"]
        )
        
    # Get then/else steps
    then_steps = parameters.get("then", [])
    else_steps = parameters.get("else", [])
    
    try:
        # Evaluate condition
        condition_result = await evaluate_condition(condition, context)
        
        # Execute then or else steps
        if condition_result:
            logger.info(f"Condition {condition} evaluated to True, executing 'then' steps")
            steps_to_execute = then_steps
        else:
            logger.info(f"Condition {condition} evaluated to False, executing 'else' steps")
            steps_to_execute = else_steps
            
        # Execute steps
        results = []
        for step in steps_to_execute:
            result = await execute_step_callback(step, context)
            results.append({
                "step_id": step.get("id", f"step-{len(results)}"),
                "success": result.success,
                "data": result.data
            })
            
            # Stop on failure if specified
            if not result.success and parameters.get("stop_on_failure", True):
                break
                
        return ExecutionResult(
            success=all(result["success"] for result in results),
            data={
                "condition_result": condition_result,
                "results": results
            },
            message=f"Conditional execution completed"
        )
        
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error executing conditional step: {e}",
            errors=[str(e)]
        )


# Placeholder implementations for other step types
async def handle_subprocess_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """Handle subprocess step placeholder."""
    logger.warning("Subprocess step not fully implemented")
    return ExecutionResult(
        success=True,
        message="Subprocess step (placeholder implementation)",
        data={"parameters": parameters}
    )
    

async def handle_notify_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """Handle notification step placeholder."""
    logger.warning("Notify step not fully implemented")
    return ExecutionResult(
        success=True,
        message="Notification step (placeholder implementation)",
        data={"parameters": parameters}
    )
    

async def handle_wait_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """Handle wait step."""
    # Get wait duration
    duration = parameters.get("duration", 1)
    
    logger.info(f"Waiting for {duration} seconds")
    await asyncio.sleep(duration)
    
    return ExecutionResult(
        success=True,
        message=f"Waited for {duration} seconds",
        data={"duration": duration}
    )
    

async def handle_variable_step(parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
    """Handle variable manipulation step."""
    # Get operation type
    operation = parameters.get("operation", "set")
    
    # Get variable name
    name = parameters.get("name")
    if not name:
        return ExecutionResult(
            success=False,
            message="No variable name specified",
            errors=["No variable name specified"]
        )
        
    try:
        if operation == "set":
            # Set variable
            value = parameters.get("value")
            context.variables[name] = value
            return ExecutionResult(
                success=True,
                message=f"Variable {name} set",
                data={"name": name, "value": value}
            )
        elif operation == "delete":
            # Delete variable
            if name in context.variables:
                del context.variables[name]
            return ExecutionResult(
                success=True,
                message=f"Variable {name} deleted",
                data={"name": name}
            )
        elif operation == "increment":
            # Increment variable
            increment = parameters.get("value", 1)
            if name in context.variables:
                context.variables[name] += increment
            else:
                context.variables[name] = increment
            return ExecutionResult(
                success=True,
                message=f"Variable {name} incremented",
                data={"name": name, "value": context.variables[name]}
            )
        else:
            return ExecutionResult(
                success=False,
                message=f"Unsupported variable operation: {operation}",
                errors=[f"Unsupported variable operation: {operation}"]
            )
            
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error manipulating variable: {e}",
            errors=[str(e)]
        )