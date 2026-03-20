# Python Standards for Magi Service

This document outlines the Python coding standards, patterns, and best practices used throughout the Magi codebase.

---

## Table of Contents

1. [Type Hints](#type-hints)
2. [Async Patterns](#async-patterns)
3. [Import Organization](#import-organization)
4. [Error Handling](#error-handling)
5. [Logging](#logging)
6. [Database Patterns](#database-patterns)
7. [Pydantic Models](#pydantic-models)
8. [FastAPI Routers](#fastapi-routers)
9. [LLM Integration](#llm-integration)
10. [Service Orchestration](#service-orchestration)
11. [Naming Conventions](#naming-conventions)
12. [Configuration](#configuration)
13. [Observability](#observability)
14. [Comments and Documentation](#comments-and-documentation)

---

## Type Hints

### Use Python 3.12 Built-in Types

Avoid the `typing` module for standard types. Use built-in generics:

```python
# GOOD
def process_items(items: list[str]) -> dict[str, int]:
    ...

def get_user(user_id: UUID) -> User | None:
    ...

# BAD
from typing import List, Dict, Optional

def process_items(items: List[str]) -> Dict[str, int]:
    ...

def get_user(user_id: UUID) -> Optional[User]:
    ...
```

### Always Annotate Function Signatures

```python
# GOOD
async def extract_feedback(
    transcript: str,
    organization_id: UUID | None = None,
    trackers: list[TrackerInfo] | None = None,
) -> ExtractionResult:
    ...

# BAD
async def extract_feedback(transcript, organization_id=None, trackers=None):
    ...
```

### TypeVar for Generic Functions

Only import from `typing` for advanced patterns:

```python
from typing import TypeVar, Type

T = TypeVar("T", bound=BaseModel)

async def fetch_one(model: Type[T], query: str, args: dict) -> T:
    ...
```

---

## Async Patterns

### Everything is Async

The entire codebase is async. **Sync I/O blocks all of Magi**, not just the current request.

```python
# GOOD - Use async libraries
import httpx
import aiofiles

async with httpx.AsyncClient() as client:
    response = await client.get(url)

async with aiofiles.open(path, "r") as f:
    content = await f.read()

# BAD - Sync I/O blocks the event loop
import requests

response = requests.get(url)  # BLOCKS EVERYTHING

with open(path, "r") as f:  # BLOCKS EVERYTHING
    content = f.read()
```

### Use asyncio.to_thread for Unavoidable Sync Code

```python
# When you must use sync code
result = await asyncio.to_thread(sync_function, arg1, arg2)
```

### Parallel Operations with asyncio.gather()

```python
# GOOD - Run independent operations in parallel
results = await asyncio.gather(
    fetch_user(user_id),
    fetch_organization(org_id),
    fetch_preferences(user_id),
)

# GOOD - Parallel processing of items
grounded_items = await asyncio.gather(
    *[ground_feedback(item, utterances) for item in feedback_items]
)
```

### Background Tasks

```python
# Fire and forget
asyncio.create_task(process_in_background())

# With FastAPI BackgroundTasks
@router.post("/endpoint")
async def endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_async)
```

---

## Import Organization

### Import Order

1. Standard library imports
2. Third-party imports
3. Local imports
4. Feature-specific imports

```python
# Standard library
import asyncio
from uuid import UUID

# Third-party
import structlog
from ddtrace import tracer
from fastapi import APIRouter, Depends

# Local - shared
from godai import database
from godai.settings import settings

# Local - feature-specific
from godai.features.copilot import db
from godai.features.copilot.extract import extract_feedback
```

### Database Import Convention

Always import database modules with clear prefixes:

```python
# Shared database queries
from godai import database

# Feature-specific queries
from godai.features.summary import db

# Usage is immediately clear
metadata = await database.get_conversation_metadata(id)  # shared
prompt = await db.get_custom_prompt(org_id)  # feature-specific
```

---

## Error Handling

### Crash Loudly - No Silent Failures

```python
# GOOD - Let it fail if key must exist
value = config["required_key"]

# BAD - Silent failure with default
value = config.get("required_key", "default")  # hides bugs
```

### Direct Access Over .get()

```python
# GOOD - Direct access when key must exist
user = users[user_id]

# BAD - Unnecessary defensive coding
user = users.get(user_id)
if user is None:
    raise ValueError("User not found")  # redundant
```

### Let Errors Bubble Up

```python
# GOOD - Original context preserved
async def process():
    result = await external_api_call()
    return transform(result)

# BAD - Swallowing and re-raising
async def process():
    try:
        result = await external_api_call()
        return transform(result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise  # loses original traceback context
```

### Custom Exceptions

```python
# Define specific exceptions
class ResourceNotFoundError(Exception):
    pass

class ConversationNotFoundError(ResourceNotFoundError):
    def __init__(self):
        super().__init__("Conversation not found")

# Use from None to suppress chained exceptions in HTTP contexts
try:
    return UUID(user_id_header)
except ValueError:
    raise HTTPException(status_code=400, detail="Invalid format") from None
```

---

## Logging

### Use structlog with Async Methods

```python
import structlog

log = structlog.get_logger()

# GOOD - Always use async logging
await log.ainfo("Processing complete", user_id=user_id, count=len(items))
await log.adebug("Debug info", data=data)
await log.awarn("Warning condition", threshold=threshold)
await log.aexception(exception)

# BAD - Sync logging blocks
log.info("Processing")  # Don't use in async context
```

### Canonical Log Lines

Emit one info-dense log per unit of work:

```python
# GOOD - Single comprehensive log
await log.ainfo(
    "Copilot Feedback Extraction",
    conversation_id=conversation_id,
    organization_id=organization_id,
    used_company_kb=True,
    feedback_count=len(items),
    duration_ms=elapsed,
)

# BAD - Scattered logs
await log.ainfo("Starting extraction")
await log.ainfo(f"User: {user_id}")
await log.ainfo("Extraction complete")
await log.ainfo(f"Found {len(items)} items")
```

### Minimize Logging

- Only log what's relevant for debugging and monitoring
- Combine logs with structured data
- Don't log routine operations

---

## Database Patterns

### Query Helper Functions

Use the typed query helpers from `godai.database.postgres`:

```python
from godai.database.postgres import fetch_one, fetch_many, insert_one, execute

# Single row
user = await fetch_one(User, sql, {"id": user_id})

# Multiple rows
users = await fetch_many(User, sql, {"org_id": org_id})

# Insert with RETURNING
new_user = await insert_one(User, sql, {"name": name})

# No return value
await execute(sql, {"id": id})
```

### SQL Query Conventions

```sql
-- No table aliases
-- Single table: plain column names
SELECT id, name, email FROM magi.users WHERE id = %(id)s

-- Multiple tables: table name prefixes (no schema)
SELECT
    survey_conversation_results.conversation_id,
    surveys.name AS survey_name
FROM magi.survey_conversation_results
JOIN magi.surveys ON survey_conversation_results.survey_id = surveys.id

-- Schema names only in FROM/JOIN clauses
```

### Feature DB Layer Organization

```python
# godai/features/copilot/db/__init__.py
from .coachable_moments import *
from .knowledge_bases import *

# Enables clean imports
from godai.features.copilot import db
moments = await db.get_coachable_moments(id)
```

---

## Pydantic Models

### BaseModel for Internal/DB Models

```python
from pydantic import BaseModel

class ConversationMetadata(BaseModel):
    time_of_recording: datetime.datetime
    rep_name: str
    organization_id: UUID
    organization_name: str
```

### CamelModel for API Responses

```python
from godai.models.camel_model import CamelModel

class CopilotFeedbackResponse(CamelModel):
    version: str
    coaching_moments: list[CoachingMoment]
    # Serializes as: {"version": "...", "coachingMoments": [...]}
```

### Define Types Close to Usage

```python
# In router.py - request/response types
class FeedbackRequest(CamelModel):
    conversation_id: UUID
    return_response: bool = False

@router.post("/generate")
async def generate(request: FeedbackRequest) -> FeedbackResponse:
    ...

# In db/users.py - database types
class User(BaseModel):
    id: UUID
    name: str
    email: str

async def get_user(user_id: UUID) -> User:
    ...
```

### Avoid Separate Type Files

Types should live near where they're used. Only create separate `types.py` files for types shared across multiple modules.

---

## FastAPI Routers

### Basic Router Pattern

```python
from fastapi import APIRouter

router = APIRouter()

class Request(BaseModel):
    conversation_id: UUID

@router.post("/generate")
async def generate(request: Request) -> None:
    await generate_summary(request.conversation_id)
```

### Dependency Injection

```python
from fastapi import Depends
from godai.dependencies.get_user_id import get_user_id

@router.post("/endpoint")
async def endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_user_id),
) -> Response:
    ...
```

### Auth Router Groups

```python
# In app.py
# Service router - Auth token only
service_router = APIRouter(dependencies=[Depends(require_auth_token)])

# Frontend router - Auth token + user ID
frontend_router = APIRouter(
    dependencies=[Depends(require_auth_token), Depends(get_user_id)]
)

# Public router - No auth
public_router = APIRouter()
```

---

## LLM Integration

### Structured Output (Returns Pydantic Model)

```python
from godai.llm.completion import llm_completion
from godai.llm.models import OpenAIModel

result = await llm_completion(
    task="copilot.feedback_extraction",
    model=OpenAIModel.GPT_4O,
    temperature=0.3,
    max_tokens=8192,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=user_prompt,
    response_format=FeedbackExtraction,  # Pydantic model
)
feedback = result.response  # Returns FeedbackExtraction instance
```

### Unstructured Output (Returns String)

```python
from godai.llm.completion import unstructured_completion
from godai.llm.types import SystemMessage, UserMessage

output = await unstructured_completion(
    task="summary",
    model=OpenAIModel.GPT_4O,
    temperature=0.1,
    max_tokens=2048,
    messages=[
        SystemMessage(content=system_prompt),
        UserMessage(content=prompt),
    ],
)
# output is a string
```

### Task Naming Convention

Use dot notation for task names: `feature.operation`

```python
task="copilot.feedback_extraction"
task="summary.generate"
task="atlas.tool_execution"
```

---

## Service Orchestration

### Arrange → Extract → Persist Pattern

```python
@tracer.wrap()
async def generate_summary(conversation_id: UUID) -> None:
    # Arrange: Gather data from DB
    transcript, metadata = await arrange_summary(conversation_id)

    # Extract: Process with LLM
    summary = await extract_summary(transcript, metadata)

    # Persist: Save results to DB
    await persist_summary(conversation_id, summary)
```

### File Organization

```
feature_name/
├── router.py      # FastAPI endpoints
├── service.py     # Main orchestration
├── arrange.py     # Gather data from DB
├── extract.py     # Process with LLM
├── persist.py     # Save results to DB
└── db/
    ├── __init__.py
    └── feature.py
```

---

## Naming Conventions

### Functions and Variables

```python
# Snake case for functions
async def generate_summary(conversation_id: UUID) -> None:
    ...

async def extract_feedback_items(transcript: str) -> list[FeedbackItem]:
    ...

# Snake case for variables
conversation_id = request.conversation_id
feedback_items = await extract_feedback(transcript)
organization_name = metadata.organization_name

# UUID suffix for IDs
user_id: UUID
organization_id: UUID
conversation_id: UUID
```

### Classes and Types

```python
# PascalCase for classes
class CopilotFeedbackResponse(CamelModel):
    ...

class CoachableMoment(BaseModel):
    ...

# Enum naming
class QuestionType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    FREE_TEXT = "free_text"
```

### Constants

```python
# UPPER_SNAKE_CASE
VERSION_BASE = "v2.2.0"
MAX_RETRY_ATTEMPTS = 3
DATE_RANGE_DAYS = 30
```

### Files

```python
# Snake case for files
conversation_processor.py
format_transcript.py
extract_feedback.py

# Standard names
router.py      # FastAPI router
service.py     # Business logic
arrange.py     # Data gathering
extract.py     # LLM processing
persist.py     # Database writes
```

---

## Configuration

### Environment Variables via Pydantic Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: Literal["dev", "prod"]
    PORT: int
    AUTH_TOKEN: str
    OPENAI_API_KEY: str

    # Computed fields for derived values
    @computed_field
    def is_production(self) -> bool:
        return self.ENV == "prod"

settings = Settings()  # Singleton
```

### Usage

```python
from godai.settings import settings

if settings.ENV == "dev":
    model = OpenAIModel.GPT_4O_MINI
else:
    model = OpenAIModel.GPT_4O
```

---

## Observability

### Datadog Tracing

```python
from ddtrace import tracer

# Decorator on service entry points
@tracer.wrap()
async def generate_summary(conversation_id: UUID) -> None:
    ...

# With custom name
@tracer.wrap("atlas.tool_execution")
async def execute_tool(tool_name: str) -> ToolResult:
    ...
```

### Error Tracking with Sentry

```python
from sentry_sdk import capture_exception

# Errors are automatically captured via middleware
# Manual capture for background tasks
try:
    await process()
except Exception as e:
    capture_exception(e)
    raise
```

### Safe Background Tasks

```python
from godai.exceptions import safe_background_task

@safe_background_task
async def process_in_background():
    # Errors are captured to Sentry instead of being swallowed
    ...
```

---

## Comments and Documentation

### Docstrings for Complex Functions

```python
async def generate_copilot_feedback(conversation_id: UUID) -> GenerateFeedbackResult:
    """
    Generate and persist feedback for a conversation.

    Flow:
    1. Fetch utterances
    2. Format transcript with timing
    3. Extract feedback items (LLM - Claude Opus)
    4. Ground each item (LLM - Claude Sonnet)
    5. Persist grounded feedback

    Args:
        conversation_id: UUID of the conversation to analyze

    Returns:
        GenerateFeedbackResult containing coaching moments and version
    """
```

### Comments Explain "Why" Not "What"

```python
# GOOD - Explains reasoning
# Setting min and max to the same value because transcript analysis
# can start many DB requests at once. Can finetune this later.
min_size=MAX_DB_POOL_SIZE,
max_size=MAX_DB_POOL_SIZE,

# BAD - Describes what code does (obvious from reading it)
# Set the pool size
min_size=MAX_DB_POOL_SIZE,
```

### Minimal Comments

- Don't add comments for self-explanatory code
- Don't add docstrings to simple functions
- Don't add type annotations in comments (use type hints)

---

## Summary Checklist

- [ ] Use Python 3.12 built-in types (`list[T]`, `T | None`)
- [ ] All I/O is async (no `requests`, no `open()`)
- [ ] Import database modules with clear prefixes (`database.`, `db.`)
- [ ] Crash loudly - no silent failures or defensive `.get()`
- [ ] Use async logging (`await log.ainfo()`)
- [ ] One canonical log line per operation
- [ ] `BaseModel` for internal, `CamelModel` for API responses
- [ ] Follow arrange → extract → persist pattern
- [ ] Snake case for functions/variables, PascalCase for classes
- [ ] Add `@tracer.wrap()` to service entry points
- [ ] Comments explain "why" not "what"
