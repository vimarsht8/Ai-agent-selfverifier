python advanced_agent_v2.py "What is 15% of 247?" --tools calculatory"""
External tool registry for the agent.
Provides various tools for verification and computation.
"""

import re
import json
import math
import os
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import aiohttp
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseTool:
    """Base class for all tools"""
    
    name: str = "base"
    description: str = "Base tool"
    
    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError
    
    def validate_input(self, **kwargs) -> bool:
        return True


class CalculatorTool(BaseTool):
    """Safe mathematical expression evaluator"""
    
    name = "calculator"
    description = "Evaluates mathematical expressions safely"
    
    def __init__(self, max_operations: int = 100):
        self.max_operations = max_operations
        self._allowed_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
        }
    
    def _is_safe_expression(self, expr: str) -> bool:
        """Check if expression contains dangerous operations"""
        # Remove whitespace
        expr = expr.replace(" ", "")
        
        # Check for only allowed characters
        allowed_chars = set("0123456789.+-*/()%^,")
        # Allow function names
        for func in self._allowed_functions:
            allowed_chars.update(func)
        
        if any(c not in allowed_chars and not c.isalpha() for c in expr):
            return False
        
        # Check for import or dangerous modules
        dangerous_keywords = ['__', 'import', 'eval', 'exec', 'open', 'file', 'compile']
        for keyword in dangerous_keywords:
            if keyword in expr:
                return False
        
        return True
    
    async def execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression"""
        try:
            # Clean expression
            expression = expression.replace("×", "*").replace("÷", "/")
            expression = expression.replace("^", "**")
            
            # Handle percentage
            if "% of" in expression:
                parts = expression.split("% of")
                if len(parts) == 2:
                    expr = f"({parts[1].strip()}) * ({parts[0].strip()}) / 100"
                    expression = expr
            
            if not self._is_safe_expression(expression):
                return ToolResult(
                    success=False,
                    result=None,
                    error="Unsafe expression detected"
                )
            
            # Evaluate in restricted environment
            safe_dict = {
                '__builtins__': {},
                **self._allowed_functions
            }
            
            result = eval(expression, safe_dict, {})
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                metadata={"expression": expression}
            )


class WebSearchTool(BaseTool):
    """Web search tool (requires API key)"""
    
    name = "web_search"
    description = "Search the web for information"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = None
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def execute(self, query: str, num_results: int = 3) -> ToolResult:
        """Search the web and return results"""
        try:
            if not self.api_key:
                return ToolResult(
                    success=False,
                    result=None,
                    error="No API key provided for web search"
                )
            
            # Example using Bing Search API (replace with your preferred API)
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, "count": num_results}
            
            session = await self._get_session()
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get("webPages", {}).get("value", []):
                        results.append({
                            "title": item.get("name"),
                            "snippet": item.get("snippet"),
                            "url": item.get("url")
                        })
                    return ToolResult(
                        success=True,
                        result=results,
                        metadata={"query": query, "count": len(results)}
                    )
                else:
                    return ToolResult(
                        success=False,
                        result=None,
                        error=f"Search API error: {response.status}"
                    )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )
    
    async def close(self):
        if self.session:
            await self.session.close()


class KnowledgeBaseTool(BaseTool):
    """Query a local knowledge base or vector database"""
    
    name = "knowledge_base"
    description = "Query internal knowledge base"
    
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path
        self.data = {}
        if kb_path and os.path.exists(kb_path):
            self._load_kb()
    
    def _load_kb(self):
        """Load knowledge base from file"""
        try:
            with open(self.kb_path, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} entries from knowledge base")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    async def execute(self, query: str, threshold: float = 0.7) -> ToolResult:
        """Query the knowledge base (simple text matching)"""
        try:
            results = []
            query_lower = query.lower()
            
            for key, value in self.data.items():
                if query_lower in key.lower():
                    results.append({
                        "key": key,
                        "value": value,
                        "score": 1.0
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return ToolResult(
                success=True,
                result=results[:5],  # Return top 5
                metadata={"query": query, "found": len(results)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class ToolRegistry:
    """Registry for managing and executing tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default tools"""
        self.register(CalculatorTool())
        # Add more default tools as needed
    
    def register(self, tool: BaseTool):
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self._tools.keys())
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        if not tool.validate_input(**kwargs):
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid input for tool '{tool_name}'"
            )
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


# Global tool registry
tools = ToolRegistry()
