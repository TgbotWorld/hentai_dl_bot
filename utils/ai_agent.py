"""
AI Agent Framework for autonomous hentai_dl_bot operation
- Supports any OpenAI-compatible LLM provider
- Sandboxed tool execution (only tools/ directory access)
- Configurable identity and constraints
- Database-persisted AI configuration
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

log = logging.getLogger(__name__)


class AIAgentConfig:
    """AI Agent configuration - stored in MongoDB"""
    
    def __init__(
        self,
        api_base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        enabled: bool = False,
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def _default_system_prompt(self) -> str:
        """Default system prompt defining agent identity and constraints"""
        return """You are HentaiAI, an autonomous download agent for the hentai_dl_bot.

IDENTITY:
- Name: HentaiAI
- Purpose: Autonomous content downloading, management, and user assistance
- Behavior: Professional, helpful, and respectful

CAPABILITIES (What you CAN do):
✓ Search for hentai content on supported sources
✓ Download episodes and series autonomously
✓ Manage downloads (batch operations, scheduling)
✓ Respond to user queries about available content
✓ Organize and catalog downloaded content
✓ Generate reports on download status
✓ Execute scheduled download tasks
✓ Cache and optimize content delivery

CONSTRAINTS (What you MUST NOT do):
✗ Access files OUTSIDE the /tools/ directory
✗ Access MongoDB credentials or user personal data
✗ Execute arbitrary system commands
✗ Modify Telegram bot core configuration
✗ Access other directories or sensitive files
✗ Perform actions without explicit request
✗ Bypass content filtering or restrictions
✗ Share API keys or credentials
✗ Access users' private messages without permission

TOOL ACCESS:
You have access to tools in the /tools/ directory only:
- download_manager.py: Handle content downloads
- video_processor.py: Process video files
- cache_manager.py: Manage local cache
- metadata_extractor.py: Extract content metadata
- scheduler.py: Schedule automated tasks

BEHAVIOR GUIDELINES:
1. Always verify tool_available() before executing
2. Report errors clearly and gracefully
3. Ask for clarification if requests are ambiguous
4. Provide progress updates for long-running tasks
5. Respect rate limits and provider terms of service
6. Maintain a helpful, professional tone"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            "api_base_url": self.api_base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIAgentConfig":
        """Create from dictionary (MongoDB document)"""
        obj = cls(
            api_base_url=data.get("api_base_url", "https://api.openai.com/v1"),
            api_key=data.get("api_key", ""),
            model=data.get("model", "gpt-3.5-turbo"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2000),
            system_prompt=data.get("system_prompt"),
            enabled=data.get("enabled", False),
        )
        obj.created_at = data.get("created_at", datetime.utcnow())
        obj.updated_at = data.get("updated_at", datetime.utcnow())
        return obj


class AIAgent:
    """Main AI Agent class for autonomous operations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.config: Optional[AIAgentConfig] = None
        self.tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Load config from database and setup"""
        try:
            config_doc = await self.db.ai_config.find_one({"_id": "ai_agent"})
            if config_doc:
                self.config = AIAgentConfig.from_dict(config_doc)
                log.info(f"AI Agent loaded: {self.config.model}, enabled={self.config.enabled}")
            else:
                self.config = AIAgentConfig()
                await self.save_config()
                log.info("AI Agent created with default config")
        except Exception as e:
            log.error(f"Error initializing AI Agent: {e}")
            self.config = AIAgentConfig()

    async def save_config(self):
        """Persist configuration to MongoDB"""
        if not self.config:
            return
        
        self.config.updated_at = datetime.utcnow()
        doc = self.config.to_dict()
        doc["_id"] = "ai_agent"
        
        await self.db.ai_config.update_one(
            {"_id": "ai_agent"},
            {"$set": doc},
            upsert=True
        )
        log.info(f"AI Agent config saved: {self.config.model}")

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Send message to AI agent and get response
        
        Args:
            user_message: User's input message
            conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
            
        Returns:
            AI agent's response
        """
        if not self.config or not self.config.enabled:
            return "❌ AI Agent is disabled. Admin must configure it via /ai_config"
        
        if not self.config.api_key or not self.config.api_base_url:
            return "❌ AI Agent not properly configured. Missing API credentials."
        
        try:
            session = await self.get_session()
            
            # Build messages
            messages = conversation_history or []
            messages.append({"role": "user", "content": user_message})
            
            # API endpoint
            endpoint = f"{self.config.api_base_url.rstrip('/')}/chat/completions"
            
            # Request payload (OpenAI compatible)
            payload = {
                "model": self.config.model,
                "messages": [{"role": "system", "content": self.config.system_prompt}] + messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            
            # Make request with timeout
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    log.error(f"AI API error ({resp.status}): {error_text}")
                    return f"❌ AI API Error ({resp.status}). Check configuration."
                
                data = await resp.json()
                
                # Extract response
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    return message.get("content", "No response from AI")
                else:
                    return "❌ Unexpected API response format"
        
        except asyncio.TimeoutError:
            return "⏱️ AI request timed out. Check API base URL."
        except aiohttp.ClientError as e:
            log.error(f"AI HTTP error: {e}")
            return f"❌ Connection error: {str(e)}"
        except json.JSONDecodeError as e:
            log.error(f"AI response JSON error: {e}")
            return "❌ Invalid AI response format"
        except Exception as e:
            log.error(f"Unexpected AI error: {e}")
            return f"❌ Error: {str(e)}"

    async def tool_available(self, tool_name: str) -> bool:
        """Check if a tool exists in the /tools/ directory"""
        tool_path = os.path.join(self.tools_dir, f"{tool_name}.py")
        return os.path.isfile(tool_path)

    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Execute a tool from /tools/ directory (sandboxed)
        
        Args:
            tool_name: Name of tool file (without .py extension)
            *args, **kwargs: Tool arguments
            
        Returns:
            Tool output
        """
        if not await self.tool_available(tool_name):
            raise ValueError(f"Tool '{tool_name}' not found in /tools/ directory")
        
        # Prevent path traversal attacks
        if ".." in tool_name or "/" in tool_name or "\\" in tool_name:
            raise ValueError(f"Invalid tool name: {tool_name}")
        
        try:
            # Import tool dynamically
            tool_path = os.path.join(self.tools_dir, tool_name)
            spec = __import__(f"tools.{tool_name}", fromlist=[tool_name])
            
            # Assume tool has a main() coroutine or function
            if hasattr(spec, "main"):
                if asyncio.iscoroutinefunction(spec.main):
                    return await spec.main(*args, **kwargs)
                else:
                    return spec.main(*args, **kwargs)
            else:
                raise ValueError(f"Tool '{tool_name}' has no main() function")
        except Exception as e:
            log.error(f"Tool execution error ({tool_name}): {e}")
            raise

    async def close(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()


# Global AI Agent instance
_ai_agent: Optional[AIAgent] = None


async def get_ai_agent(db: AsyncIOMotorDatabase) -> AIAgent:
    """Get or create global AI Agent instance"""
    global _ai_agent
    if _ai_agent is None:
        _ai_agent = AIAgent(db)
        await _ai_agent.initialize()
    return _ai_agent
