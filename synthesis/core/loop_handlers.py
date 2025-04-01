#!/usr/bin/env python3
"""
Synthesis Loop Handlers

This module handles loop execution for Synthesis.
"""

import logging
from typing import Dict, List, Any, Optional

from synthesis.core.execution_models import ExecutionContext, ExecutionResult
from synthesis.core.condition_evaluator import evaluate_condition

# Configure logging
logger = logging.getLogger("synthesis.core.loop_handlers")


async def handle_loop_step(parameters: Dict[str, Any], 
                        steps: List[Dict[str, Any]],
                        context: ExecutionContext,
                        execute_step_callback) -> ExecutionResult:
    """
    Handle a loop execution step.
    
    Args:
        parameters: Step parameters
        steps: Steps to execute in the loop
        context: Execution context
        execute_step_callback: Callback to execute a step
        
    Returns:
        ExecutionResult with loop results
    """
    # Get loop type
    loop_type = parameters.get("type", "for")
    
    # Get loop steps
    if not steps:
        return ExecutionResult(
            success=True,
            message="No steps to execute in loop",
            data={"iterations": 0}
        )
        
    try:
        if loop_type == "for":
            # Process for loop
            return await handle_for_loop(parameters, steps, context, execute_step_callback)
        elif loop_type == "while":
            # Process while loop
            return await handle_while_loop(parameters, steps, context, execute_step_callback)
        else:
            return ExecutionResult(
                success=False,
                message=f"Unsupported loop type: {loop_type}",
                errors=[f"Unsupported loop type: {loop_type}"]
            )
            
    except Exception as e:
        return ExecutionResult(
            success=False,
            message=f"Error executing loop: {e}",
            errors=[str(e)]
        )


async def handle_for_loop(parameters: Dict[str, Any], 
                       steps: List[Dict[str, Any]], 
                       context: ExecutionContext, 
                       execute_step_callback) -> ExecutionResult:
    """
    Handle a for loop.
    
    Args:
        parameters: Loop parameters
        steps: Steps to execute in each iteration
        context: Execution context
        execute_step_callback: Callback to execute a step
        
    Returns:
        ExecutionResult with loop results
    """
    # Get iterable
    iterable = parameters.get("iterable")
    if not iterable:
        return ExecutionResult(
            success=False,
            message="No iterable specified for for loop",
            errors=["No iterable specified for for loop"]
        )
        
    # Get item variable name
    item_var = parameters.get("item_var", "item")
    
    # Get max iterations
    max_iterations = parameters.get("max_iterations", 100)
    
    # Process the iterable
    if isinstance(iterable, str) and iterable in context.variables:
        # Get iterable from context variables
        items = context.variables[iterable]
    elif isinstance(iterable, list):
        # Use provided list directly
        items = iterable
    else:
        return ExecutionResult(
            success=False,
            message=f"Invalid iterable: {iterable}",
            errors=[f"Invalid iterable: {iterable}"]
        )
        
    # Limit iterations
    items = items[:max_iterations]
    
    # Execute loop
    results = []
    for index, item in enumerate(items):
        # Add loop variables to context
        context.variables[item_var] = item
        context.variables["loop_index"] = index
        
        # Execute steps
        iteration_results = []
        for step in steps:
            result = await execute_step_callback(step, context)
            iteration_results.append({
                "step_id": step.get("id", f"step-{len(iteration_results)}"),
                "success": result.success,
                "data": result.data
            })
            
            # Stop on failure if specified
            if not result.success and parameters.get("stop_on_failure", True):
                break
                
        # Add iteration results
        results.append({
            "index": index,
            "success": all(result["success"] for result in iteration_results),
            "results": iteration_results
        })
        
        # Stop loop on failure if specified
        if not results[-1]["success"] and parameters.get("break_on_failure", False):
            break
            
    return ExecutionResult(
        success=all(result["success"] for result in results),
        data={
            "iterations": len(results),
            "results": results
        },
        message=f"For loop completed with {len(results)} iterations"
    )


async def handle_while_loop(parameters: Dict[str, Any], 
                         steps: List[Dict[str, Any]], 
                         context: ExecutionContext, 
                         execute_step_callback) -> ExecutionResult:
    """
    Handle a while loop.
    
    Args:
        parameters: Loop parameters
        steps: Steps to execute in each iteration
        context: Execution context
        execute_step_callback: Callback to execute a step
        
    Returns:
        ExecutionResult with loop results
    """
    # Get condition
    condition = parameters.get("condition")
    if not condition:
        return ExecutionResult(
            success=False,
            message="No condition specified for while loop",
            errors=["No condition specified for while loop"]
        )
        
    # Get max iterations
    max_iterations = parameters.get("max_iterations", 100)
    
    # Execute loop
    results = []
    iteration = 0
    
    while iteration < max_iterations:
        # Evaluate condition
        condition_result = await evaluate_condition(condition, context)
        
        # Stop if condition is false
        if not condition_result:
            break
            
        # Add loop variables to context
        context.variables["loop_iteration"] = iteration
        
        # Execute steps
        iteration_results = []
        for step in steps:
            result = await execute_step_callback(step, context)
            iteration_results.append({
                "step_id": step.get("id", f"step-{len(iteration_results)}"),
                "success": result.success,
                "data": result.data
            })
            
            # Stop on failure if specified
            if not result.success and parameters.get("stop_on_failure", True):
                break
                
        # Add iteration results
        results.append({
            "iteration": iteration,
            "success": all(result["success"] for result in iteration_results),
            "results": iteration_results
        })
        
        # Stop loop on failure if specified
        if not results[-1]["success"] and parameters.get("break_on_failure", False):
            break
            
        # Increment iteration counter
        iteration += 1
        
    return ExecutionResult(
        success=all(result["success"] for result in results),
        data={
            "iterations": len(results),
            "results": results
        },
        message=f"While loop completed with {len(results)} iterations"
    )