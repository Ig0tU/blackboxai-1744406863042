create_nlp_sandbox_llm.sh

#!/bin/bash

echo "Creating NLP Agent Sandbox Project Structure with Gemini/Hugging Face support..."
echo "This will create a directory named 'nlp-agent-sandbox-llm' in the current location."
read -p "Press Enter to continue, or Ctrl+C to cancel..."

# --- Create Project Directory ---
PROJECT_DIR="nlp-agent-sandbox-llm"
if [ -d "$PROJECT_DIR" ]; then
  echo "Warning: Directory '$PROJECT_DIR' already exists. Files may be overwritten."
  read -p "Press Enter to continue overwriting, or Ctrl+C to cancel..."
fi
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1
echo "Created project directory: $PROJECT_DIR and entered it."

# --- Create Subdirectories ---
mkdir -p agents routes services tools templates static static/css static/js sandbox_workspace
echo "Created subdirectories: agents, routes, services, tools, templates, static, sandbox_workspace"

# --- Create Empty __init__.py Files ---
touch agents/__init__.py routes/__init__.py services/__init__.py tools/__init__.py
echo "Created __init__.py files."

# --- Create .gitignore ---
cat << 'EOF' > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# PEP 582; used by PDM, Flit and potentially other packaging tools.
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static analysis results
.pytype/

# Cython debug symbols
cython_debug/

# VSCode
.vscode/

# Docker
docker-compose.override.yml
*.log
logs/

# Databases
*.db
*.sqlite3

# Temporary files
*.tmp
*.bak
*.swp
*~

# Node / JS dependencies
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
package-lock.json
yarn.lock

# OS generated files
.DS_Store
Thumbs.db

# Custom
sandbox_workspace/*
!sandbox_workspace/README.md
*.pid

EOF
echo "Created .gitignore"

# --- Create README.md ---
cat << 'EOF' > README.md
# NLP Agent Sandbox (Gemini / Hugging Face Edition)

This project provides a blueprint for building an AI agent capable of interacting with various tools within a secure sandboxed environment (Docker). It uses Flask for the backend, SocketIO for real-time communication, and allows integration with either Google Gemini or Hugging Face LLMs for natural language processing and tool selection.

**Features:**

*   **Modular Architecture:** Separated components for agents, tools, services, and routes.
*   **LLM Agnostic (Gemini/HF):** Easily configure to use Google Gemini or Hugging Face Inference API via `.env`.
*   **Tool Calling:**
    *   **Gemini:** Utilizes native function calling. Tools are defined using OpenAPI-like schemas.
    *   **Hugging Face:** Uses prompt engineering and JSON parsing for tool invocation. Requires capable instruction-following models (e.g., Mistral Instruct, Llama Instruct). Reliability depends on the model's ability to follow JSON format instructions.
*   **Sandboxed Execution:** Tools like `shell` and `file_system` operate within isolated Docker containers managed by `SandboxManager`.
*   **Web Interface:** Basic UI for starting/stopping agents, sending messages, viewing interactions, and browsing the agent's workspace.
*   **Real-time Updates:** Uses WebSockets (Flask-SocketIO) for live updates in the frontend.
*   **Containerized:** Ready to build and run using Docker and Docker Compose.

**Project Structure:**

nlp-agent-sandbox-llm/ ├── agents/ # Agent logic (BaseAgent, specific personas) │ ├── init.py │ ├── base_agent.py # Core agent class with LLM interaction (Gemini/HF) │ ├── lovable_agent.py # Example persona │ └── cursor_agent.py # Example persona ├── routes/ # Flask API endpoints and WebSocket handlers │ ├── init.py │ ├── main_routes.py # Basic web routes (index page, health) │ └── agent_routes.py # API/WebSocket routes for agent management & workspace ├── services/ # Backend services (Docker interaction, tool execution) │ ├── init.py │ ├── sandbox_manager.py # Manages Docker sandbox containers │ └── tool_executor.py # Dispatches calls to specific tools, handles schema ├── tools/ # Implementations of available tools │ ├── init.py │ ├── file_system.py # Interact with files in the sandbox │ ├── shell.py # Execute shell commands in the sandbox │ ├── web_search.py # Perform web searches (requires API key) │ ├── browser.py # (Placeholder) Browser automation tools │ ├── coding.py # (Placeholder) Code execution, linting, etc. │ ├── deployment.py # (Placeholder) Deploying services/apps │ ├── communication.py # (Placeholder) Agent-to-agent/user communication │ ├── api_caller.py # (Placeholder) Call external APIs │ └── utils.py # (Placeholder) Helper functions for tools ├── templates/ # HTML templates (Jinja2) │ ├── base.html │ └── index.html ├── static/ # Static assets (CSS, JavaScript) │ ├── css/style.css │ └── js/main.js ├── sandbox_workspace/ # Directory mounted into sandbox containers (/workspace) │ └── README.md ├── app.py # Main Flask application entry point ├── config.py # Configuration class (loads from .env) ├── requirements.txt # Python dependencies ├── Dockerfile # Docker image definition for the application ├── docker-compose.yml # Docker Compose configuration ├── .env.example # Example environment variables (copy to .env) ├── create_nlp_sandbox_llm.sh # This script ├── .gitignore # Git ignore rules └── README.md # This file


**Setup and Running:**

1.  **Prerequisites:**
    *   Docker ([https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/))
    *   Docker Compose ([https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/))
    *   Bash (usually available on Linux/macOS/WSL)

2.  **Run the Creation Script (If you haven't already):**
    ```bash
    bash create_nlp_sandbox_llm.sh
    cd nlp-agent-sandbox-llm
    ```

3.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   **Edit `.env`:**
        *   Set a strong `SECRET_KEY`.
        *   **Choose your LLM Provider:**
            *   Uncomment **either** the `LLM_PROVIDER=gemini` section **or** the `LLM_PROVIDER=huggingface` section. **Do not uncomment both.**
            *   Provide the corresponding `GOOGLE_API_KEY` or `HUGGINGFACE_API_TOKEN`.
            *   Set the appropriate `LLM_MODEL_NAME`.
                *   For Gemini: e.g., `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`.
                *   For Hugging Face: Use the repository ID of an instruction-following model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-3-8b-chat-hf`). Ensure the chosen HF model is suitable for instruction following and JSON output if using tools.
            *   Optionally set `HUGGINGFACE_INFERENCE_ENDPOINT` if using a dedicated HF endpoint.
        *   Add any other required API keys (e.g., `WEB_SEARCH_API_KEY` for the `web_search` tool). Configure the `web_search.py` tool for your specific provider.

4.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    *   The `--build` flag is only needed the first time or when dependencies (`requirements.txt`) or the `Dockerfile` change.

5.  **Access the Application:**
    *   Open your web browser to `http://localhost:5000` (or the port specified by `APP_PORT` in your `.env`).

**Development:**

*   **Implement Tool Logic:** Flesh out the Python code within the `tools/*.py` files (especially `browser.py`, `coding.py`, `deployment.py`, `api_caller.py`). Add necessary libraries to `requirements.txt` and rebuild (`docker-compose build web` or `docker-compose up --build`).
*   **Add New Tools:**
    1.  Create a new `tools/your_tool_name.py` file.
    2.  Define functions within it, including clear docstrings (first line is summary) and Python type hints for all arguments.
    3.  The `ToolExecutor` will automatically discover the new tool module and functions on restart.
    4.  The `BaseAgent` will automatically make the new tool available to the LLM (either via Gemini schema or HF prompt).
*   **Customize Agent Personas:** Create new classes inheriting from `BaseAgent` in the `agents/` directory and override methods like `_get_system_prompt` to define specific behaviors. Add them to `AGENT_PERSONAS` in `routes/agent_routes.py`.
*   **Enhance Frontend:** Improve the `templates/index.html` and `static/js/main.js` for a richer user experience.

**Important Notes:**

*   **Security:** Running arbitrary code (especially shell commands) is inherently risky. The Docker sandbox provides a layer of isolation, but ensure the `SANDBOX_IMAGE_NAME` is appropriate and review security settings (`ENABLE_STRICT_SANDBOX_SECURITY` in `.env`, container options in `SandboxManager`). Be extremely cautious with tools like `execute_shell_command` and `deploy_*`.
*   **Hugging Face Tool Calling Reliability:** Tool calling with Hugging Face models relies heavily on the model's ability to follow instructions and format output correctly (JSON). This can be less reliable than native function calling APIs like Gemini's. Prompt engineering and model choice are critical. Test thoroughly.
*   **Resource Usage:** Running LLMs (especially locally if using `transformers` instead of the Inference API) and multiple Docker containers can be resource-intensive.
*   **Error Handling:** The blueprint includes basic error handling, but robust production systems require more comprehensive error management, retries, and monitoring.

EOF
echo "Created README.md"

# --- Create .env.example ---
cat << 'EOF' > .env.example
# Environment Variables for NLP Agent Sandbox (Gemini/Hugging Face Edition)
# Copy this file to .env and fill in your values.
# Choose ONLY ONE LLM Provider section below.

# --- Core Flask Settings ---
FLASK_ENV=development # Change to 'production' for deployment
SECRET_KEY=replace_this_with_a_strong_random_key_please_change_me # Generate using: python -c 'import secrets; print(secrets.token_hex(24))'
APP_PORT=5000

# --- LLM Configuration ---
# --- Option 1: Google Gemini ---
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=YOUR_GOOGLE_GENERATIVE_AI_API_KEY # Get from Google AI Studio -> API Key
# LLM_MODEL_NAME=gemini-1.5-flash-latest # Or gemini-1.5-pro-latest, gemini-1.0-pro

# --- Option 2: Hugging Face (Inference API Recommended) ---
LLM_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=hf_YOUR_HUGGINGFACE_READ_TOKEN # Get from HF Settings -> Access Tokens (needs 'read' permission)
# Choose a good INSTRUCTION-FOLLOWING model suitable for function/tool calling via JSON prompt.
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 # Example: Mistral Instruct
# LLM_MODEL_NAME=meta-llama/Llama-3-8b-chat-hf # Example: Llama 3 Instruct (Requires access approval on HF)
# LLM_MODEL_NAME=google/gemma-1.1-7b-it # Example: Gemma Instruct
# Optional: If using a dedicated Inference Endpoint URL instead of the model ID
# HUGGINGFACE_INFERENCE_ENDPOINT=YOUR_HF_INFERENCE_ENDPOINT_URL

# --- LLM Parameters (Optional Overrides) ---
# LLM_TEMPERATURE=0.7 # 0.0 for more deterministic, > 0.7 for more creative
# LLM_MAX_TOKENS=4096 # Max tokens for the LLM response generation

# --- Tool API Keys ---
# Configure API keys needed by tools in the 'tools/' directory.
# Example for web_search.py (using Serper.dev - https://serper.dev)
# WEB_SEARCH_API_KEY=YOUR_SERPER_DEV_API_KEY
# Example for api_caller.py (if calling a weather API)
# WEATHER_API_KEY=YOUR_WEATHERAPI_COM_KEY

# --- Sandbox Configuration ---
SANDBOX_IMAGE_NAME=ubuntu:22.04 # Base image for sandboxes. Customize if tools need specific dependencies.
SANDBOX_NETWORK_NAME=agent_sandbox_net # Name for the Docker network connecting sandboxes
SANDBOX_MEM_LIMIT=512m # Memory limit per sandbox container (e.g., 512m, 1g)
SANDBOX_CPU_SHARES=1024 # Relative CPU weight (default 1024)

# --- Agent Configuration ---
# MAX_AGENT_ITERATIONS=15 # Max thinking loops before agent stops
# AGENT_HISTORY_MAX_MESSAGES=20 # Max turns (user + assistant/tool) kept in history
# AGENT_TOOL_RESULT_MAX_LEN=3000 # Max length of tool output stored in history / sent to LLM

# --- Tool Specific Settings ---
# SHELL_COMMAND_TIMEOUT=300 # Max seconds for shell commands (tools/shell.py)
# MAX_SHELL_OUTPUT_LINES=200 # Max lines of shell output returned
# MAX_FILE_READ_LINES=1000 # Max lines read by file_system.read_file

# --- Deployment Configuration (Optional) ---
# PROXY_DOMAIN=localhost # Domain used by deploy_expose_port tool (requires external proxy setup)

# --- Security ---
# Enable stricter path checks in file_system tool, potentially other security measures.
ENABLE_STRICT_SANDBOX_SECURITY=True

# --- Database (Optional - Uncomment and configure if using a persistent DB) ---
# DATABASE_URL=postgresql://agent_user:replace_with_strong_password@db:5432/agent_sandbox
# POSTGRES_DB=agent_sandbox
# POSTGRES_USER=agent_user
# POSTGRES_PASSWORD=replace_with_strong_password

EOF
echo "Created .env.example (Remember to copy to .env and configure)"

# --- Create requirements.txt ---
cat << 'EOF' > requirements.txt
# Flask Core & WebSockets
Flask>=2.3.0
Flask-SocketIO>=5.3.0
gevent-websocket>=0.10.1 # Required by gunicorn worker for SocketIO
gunicorn>=21.2.0        # Production WSGI server

# Environment & Config
python-dotenv>=1.0.0

# Docker Interaction
docker>=6.1.0

# Basic Web Scraping / Parsing (Used by browser tool fallback, web search tool)
requests>=2.30.0
beautifulsoup4>=4.12.0 # For HTML parsing

# --- LLM Libraries ---
# Install libraries based on your .env choice (LLM_PROVIDER)

# For Google Gemini (LLM_PROVIDER=gemini)
google-generativeai>=0.5.0

# For Hugging Face (LLM_PROVIDER=huggingface)
huggingface_hub>=0.20.0 # For Inference API client and model downloads

# --- Optional: Local Hugging Face Models via Transformers ---
# Uncomment these if you want to run models locally using the transformers library
# Requires PyTorch or TensorFlow installed separately. Can be resource intensive.
# transformers>=4.38.0
# accelerate>=0.25.0 # Often improves loading speed and memory usage
# sentencepiece # Often needed by tokenizers
# bitsandbytes # For 8-bit/4-bit quantization (model size reduction)
# Pytorch Installation (example):
# torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8 example
# torch torchvision torchaudio # CPU-only example

# --- Other Tool Dependencies ---
# Add libraries needed by specific tools in the 'tools/' directory as you implement them.
# Examples:
# pandas # For data analysis tools
# numpy # For numerical operations
# matplotlib # For plotting tools
# pyyaml # For reading/writing YAML files
# playwright # For tools/browser.py (requires installation: playwright install --with-deps)
# flake8 # For tools/coding.py lint_code tool
# pylint # For tools/coding.py lint_code tool

EOF
echo "Created requirements.txt"

# --- Create config.py ---
cat << 'EOF' > config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

log = logging.getLogger(__name__)

class Config:
    """Application Configuration - Values primarily loaded from environment variables."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key_change_me')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development') # 'production' or 'development'

    # --- Sandbox Configuration ---
    SANDBOX_IMAGE_NAME = os.environ.get('SANDBOX_IMAGE_NAME', 'ubuntu:22.04')
    SANDBOX_WORKSPACE_DIR = os.path.abspath('sandbox_workspace')
    SANDBOX_NETWORK_NAME = os.environ.get('SANDBOX_NETWORK_NAME', 'agent_sandbox_net')
    SANDBOX_MEM_LIMIT = os.environ.get('SANDBOX_MEM_LIMIT', '512m')
    SANDBOX_CPU_SHARES = int(os.environ.get('SANDBOX_CPU_SHARES', 1024)) # Default Docker CPU shares

    # --- API Keys (Loaded ONLY from environment) ---
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')
    WEB_SEARCH_API_KEY = os.environ.get('WEB_SEARCH_API_KEY')
    # Add other API keys here as needed (e.g., WEATHER_API_KEY)
    # Example: WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

    # --- LLM Configuration ---
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', '').lower() # 'gemini' or 'huggingface'
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME')
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', 0.7))
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', 4096)) # Adjust based on model limits
    HUGGINGFACE_INFERENCE_ENDPOINT = os.environ.get('HUGGINGFACE_INFERENCE_ENDPOINT') # Optional for HF dedicated endpoints

    # --- Agent Configuration ---
    DEFAULT_AGENT_PERSONA = os.environ.get('DEFAULT_AGENT_PERSONA','assistant')
    MAX_AGENT_ITERATIONS = int(os.environ.get('MAX_AGENT_ITERATIONS', 15)) # Max thinking loops
    MAX_LINTER_FIX_ATTEMPTS = int(os.environ.get('MAX_LINTER_FIX_ATTEMPTS', 3)) # Example for coding tool
    AGENT_HISTORY_MAX_MESSAGES = int(os.environ.get('AGENT_HISTORY_MAX_MESSAGES', 20)) # Max turns (user + assistant/tool)
    AGENT_TOOL_RESULT_MAX_LEN = int(os.environ.get('AGENT_TOOL_RESULT_MAX_LEN', 3000)) # Truncate long tool outputs in history

    # --- Tool Specific Settings ---
    BROWSER_TIMEOUT = int(os.environ.get('BROWSER_TIMEOUT', 30)) # Seconds for browser actions (tools/browser.py)
    MAX_CONSOLE_LOG_LINES = int(os.environ.get('MAX_CONSOLE_LOG_LINES', 50)) # Example limit if tool captures console logs
    MAX_FILE_READ_LINES = int(os.environ.get('MAX_FILE_READ_LINES', 1000)) # Limit lines read by file_system.read_file
    MAX_SHELL_OUTPUT_LINES = int(os.environ.get('MAX_SHELL_OUTPUT_LINES', 200)) # Limit lines returned by shell.execute_shell_command
    SHELL_COMMAND_TIMEOUT = int(os.environ.get('SHELL_COMMAND_TIMEOUT', 300)) # Seconds for shell commands (tools/shell.py)

    # --- Deployment Configuration ---
    PROXY_DOMAIN = os.environ.get('PROXY_DOMAIN', 'localhost') # Used by deploy_expose_port tool

    # --- Allowed Data APIs (Example for api_caller tool) ---
    # Define APIs the agent can call via 'call_data_api' tool
    # Structure: { "api_key_name_in_config": "base_url" } - Key name should match env var holding the key
    ALLOWED_DATA_APIS = {
        # "WEATHER_API_KEY": "https://api.weatherapi.com/v1", # Example - Requires WEATHER_API_KEY in .env
    }

    # --- Database (Optional) ---
    DATABASE_URL = os.environ.get('DATABASE_URL') # e.g., postgresql://user:pass@host/db
    # Example if using SQLAlchemy:
    # SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app.db')
    # SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Security ---
    # Enable stricter security checks (e.g., path traversal in file tools)
    ENABLE_STRICT_SANDBOX_SECURITY = os.environ.get('ENABLE_STRICT_SANDBOX_SECURITY', 'True').lower() == 'true'

    # --- Validation ---
    @classmethod
    def validate_config(cls):
        """Basic validation of critical configuration."""
        errors = []
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'default_secret_key_change_me':
            log.warning("SECURITY WARNING: SECRET_KEY is not set or is the default value. Please set a strong secret key in your .env file.")
            # errors.append("SECRET_KEY is insecure.") # Don't block startup for this, just warn

        if not cls.LLM_PROVIDER:
            errors.append("LLM_PROVIDER is not set in .env. Please choose 'gemini' or 'huggingface'.")
        elif cls.LLM_PROVIDER not in ['gemini', 'huggingface']:
            errors.append(f"Invalid LLM_PROVIDER '{cls.LLM_PROVIDER}'. Must be 'gemini' or 'huggingface'.")
        elif not cls.LLM_MODEL_NAME:
            errors.append(f"LLM_MODEL_NAME is not set for provider '{cls.LLM_PROVIDER}'.")

        if cls.LLM_PROVIDER == 'gemini' and not cls.GOOGLE_API_KEY:
            errors.append("LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing in .env.")
        if cls.LLM_PROVIDER == 'huggingface' and not cls.HUGGINGFACE_API_TOKEN:
             # Allow endpoint URL as alternative auth sometimes, but token is standard
             if not cls.HUGGINGFACE_INFERENCE_ENDPOINT:
                  errors.append("LLM_PROVIDER is 'huggingface' but HUGGINGFACE_API_TOKEN is missing in .env (and no HUGGINGFACE_INFERENCE_ENDPOINT provided).")
             else:
                  log.info("Using Hugging Face Inference Endpoint URL. API Token might not be required depending on endpoint security.")


        if errors:
            for error in errors:
                log.error(f"Configuration Error: {error}")
            return False

        log.info(f"Configuration loaded. LLM Provider: {cls.LLM_PROVIDER}, Model: {cls.LLM_MODEL_NAME}")
        return True

EOF
echo "Created config.py"

# --- Create app.py ---
cat << 'EOF' > app.py
import os
import logging
import sys
from flask import Flask
from flask_socketio import SocketIO
from config import Config
from routes.main_routes import main_bp
from routes.agent_routes import agent_bp, active_agents, register_socketio_events, cleanup_all_agents
from services.sandbox_manager import SandboxManager
import signal

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s:%(threadName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to stdout
log = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object(Config)

# --- Validate Configuration ---
if not Config.validate_config():
    log.error("Invalid configuration detected. Please check your .env file and logs. Exiting.")
    sys.exit(1)

# --- Initialize SocketIO ---
# Use 'gevent' for compatibility with Gunicorn worker specified in docker-compose.yml
async_mode = 'gevent'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*") # Allow all origins for dev, restrict in prod
log.info(f"Flask-SocketIO initialized with async_mode: {async_mode}")

# --- Initialize Core Services ---
# Attach services to app context for access in routes/agents
try:
    app.sandbox_manager = SandboxManager(config=Config)
    log.info("SandboxManager initialized.")
except ImportError as e:
     log.error(f"CRITICAL: Failed to import Docker SDK. Is the 'docker' library installed? Error: {e}", exc_info=True)
     app.sandbox_manager = None
     sys.exit(1)
except ConnectionError as e:
     log.error(f"CRITICAL: Could not connect to Docker. Is the Docker daemon running and accessible? Error: {e}", exc_info=True)
     app.sandbox_manager = None
     sys.exit(1)
except Exception as e:
    log.error(f"CRITICAL: Failed to initialize SandboxManager: {e}", exc_info=True)
    app.sandbox_manager = None # Indicate failure
    sys.exit(1)

# ToolExecutor is initialized lazily via get_tool_executor() in agent_routes to avoid circular deps

# --- Register Blueprints ---
app.register_blueprint(main_bp)
app.register_blueprint(agent_bp, url_prefix='/api/agent')
log.info("Flask Blueprints registered.")

# --- Register SocketIO Event Handlers ---
# Pass app context items needed by handlers if necessary
# Example: register_socketio_events(socketio, app.sandbox_manager)
register_socketio_events(socketio)
log.info("SocketIO event handlers registered.")

# --- Docker Network Check ---
# Performed by SandboxManager constructor now.

# --- Workspace Directory Check ---
def check_workspace_dir():
    workspace_dir = Config.SANDBOX_WORKSPACE_DIR
    log.info(f"Ensuring sandbox workspace directory exists: {workspace_dir}")
    try:
        os.makedirs(workspace_dir, exist_ok=True)
        # Check write permissions more reliably
        test_file = os.path.join(workspace_dir, '.permission_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        log.info(f"Workspace directory '{workspace_dir}' exists and is writable.")
    except PermissionError:
         log.error(f"CRITICAL: Workspace directory {workspace_dir} is not writable by the application user (UID/GID: {os.getuid()}/{os.getgid()}). Please check permissions.")
         sys.exit(1)
    except Exception as e:
        log.error(f"Failed to create or verify workspace directory {workspace_dir}: {e}", exc_info=True)
        sys.exit(1)

# --- Graceful Shutdown Handling ---
def signal_handler(signum, frame):
    log.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    # Pass necessary components to cleanup function
    cleanup_all_agents(app.sandbox_manager, socketio)
    # Give some time for cleanup before exiting
    socketio.sleep(2) # Use socketio sleep if using gevent
    log.info("Shutdown complete.")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# --- Application Entry Point ---
if __name__ == '__main__':
    log.info("Starting NLP Agent Sandbox Application...")
    check_workspace_dir() # Check workspace after potential volume mount

    host = '0.0.0.0'
    port = int(os.environ.get('APP_PORT', 5000))
    debug_mode = Config.FLASK_ENV == 'development'

    log.info(f"Starting server on {host}:{port} (Mode: {Config.FLASK_ENV}, Debug: {debug_mode})")
    # Use socketio.run() for development server with WebSocket support
    # For production, Gunicorn is used via docker-compose.yml
    # The 'if __name__ == '__main__':' block is primarily for local dev without Gunicorn
    try:
        # Enable logging for socketio and engineio in debug mode
        socketio_log = debug_mode
        engineio_log = debug_mode
        socketio.run(app, host=host, port=port, debug=debug_mode, use_reloader=debug_mode,
                     log_output=debug_mode, engineio_logger=engineio_log, logger=socketio_log)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received. Shutting down...")
        # Signal handler should catch this, but call cleanup just in case
        cleanup_all_agents(app.sandbox_manager, socketio)
    except Exception as e:
        log.error(f"Server exited with error: {e}", exc_info=True)
        cleanup_all_agents(app.sandbox_manager, socketio)
        sys.exit(1)
    finally:
        log.info("Application shutdown sequence finished.")

# Note: The @app.teardown_appcontext is less reliable for catching shutdown signals
# like SIGTERM/SIGINT compared to the signal handlers, especially with workers.
# Keeping the signal handler approach for explicit cleanup.

EOF
echo "Created app.py"

# --- Create Dockerfile ---
cat << 'EOF' > Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent buffering and bytecode files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set Flask environment variables (can be overridden by docker-compose)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any needed by tools)
# Example: Install git, build tools, or dependencies for Playwright/Selenium
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     build-essential \
#     # Add other packages needed by tools here
#  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port specified by APP_PORT (default 5000) available
# Note: The actual port mapping happens in docker-compose.yml
EXPOSE ${APP_PORT:-5000}

# Create the workspace directory within the image (volume mount will overlay this)
# Ensure it's writable by the user Gunicorn runs as (often root by default in simple setups)
# If running Gunicorn as non-root, adjust ownership accordingly.
RUN mkdir -p /app/sandbox_workspace && chmod 777 /app/sandbox_workspace
# Consider running as a non-root user for better security:
# RUN addgroup --system app && adduser --system --ingroup app appuser
# RUN chown -R appuser:app /app
# USER appuser

# Define default environment variables (will be overridden by docker-compose from .env)
ENV SECRET_KEY="change_me_in_env_file"
ENV APP_PORT=5000
ENV LLM_PROVIDER=""
ENV LLM_MODEL_NAME=""
ENV GOOGLE_API_KEY=""
ENV HUGGINGFACE_API_TOKEN=""
ENV SANDBOX_NETWORK_NAME=agent_sandbox_net
# Add other ENV vars needed from .env as defaults if desired

# Command to run the application using Gunicorn for production
# Uses gevent worker for Flask-SocketIO compatibility
# The number of workers can be adjusted based on server resources (e.g., $(nproc --all) )
# Ensure the port here matches the EXPOSE line and APP_PORT env var.
# Increased timeout for potentially long-running agent requests.
CMD ["gunicorn", "--workers", "2", "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "--bind", "0.0.0.0:5000", "--timeout", "120", "--log-level", "info", "app:app"]

# For development, you might override the CMD in docker-compose.override.yml
# to run Flask's built-in server with reloader:
# command: flask run --host=0.0.0.0 --port=${APP_PORT:-5000} --reload
EOF
echo "Created Dockerfile"

# --- Create docker-compose.yml ---
cat << 'EOF' > docker-compose.yml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: nlp_agent_sandbox_web
    ports:
      # Map container port (set by APP_PORT, default 5000) to host port
      - "${APP_PORT:-5000}:${APP_PORT:-5000}"
    volumes:
      # Mount the application code for development (reflects changes without rebuild)
      # Comment out for production image-based deployment if preferred
      - .:/app
      # Mount the shared workspace directory (ensure host path exists)
      - ./sandbox_workspace:/app/sandbox_workspace
      # Mount Docker socket to allow sandbox management from within the container
      # SECURITY RISK: Gives the container root-level access to the host's Docker daemon.
      # Consider alternatives like a proxy service or running SandboxManager on the host if security is paramount.
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      # Pass environment variables from .env file to the container
      # Ensure all variables used in config.py and needed at runtime are listed here.
      - FLASK_ENV=${FLASK_ENV:-development}
      - SECRET_KEY=${SECRET_KEY}
      - APP_PORT=${APP_PORT:-5000}
      - LLM_PROVIDER=${LLM_PROVIDER}
      - LLM_MODEL_NAME=${LLM_MODEL_NAME}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}
      - HUGGINGFACE_INFERENCE_ENDPOINT=${HUGGINGFACE_INFERENCE_ENDPOINT}
      - WEB_SEARCH_API_KEY=${WEB_SEARCH_API_KEY}
      # Pass sandbox config
      - SANDBOX_IMAGE_NAME=${SANDBOX_IMAGE_NAME:-ubuntu:22.04}
      - SANDBOX_NETWORK_NAME=${SANDBOX_NETWORK_NAME:-agent_sandbox_net}
      - SANDBOX_MEM_LIMIT=${SANDBOX_MEM_LIMIT:-512m}
      - SANDBOX_CPU_SHARES=${SANDBOX_CPU_SHARES:-1024}
      # Pass tool/agent config
      - PROXY_DOMAIN=${PROXY_DOMAIN:-localhost}
      - ENABLE_STRICT_SANDBOX_SECURITY=${ENABLE_STRICT_SANDBOX_SECURITY:-True}
      - AGENT_TOOL_RESULT_MAX_LEN=${AGENT_TOOL_RESULT_MAX_LEN}
      - MAX_SHELL_OUTPUT_LINES=${MAX_SHELL_OUTPUT_LINES}
      - MAX_FILE_READ_LINES=${MAX_FILE_READ_LINES}
      - SHELL_COMMAND_TIMEOUT=${SHELL_COMMAND_TIMEOUT}
      # Add any other env vars needed by the application or tools
      # - DATABASE_URL=${DATABASE_URL} # Example for database
      # - WEATHER_API_KEY=${WEATHER_API_KEY} # Example for tool
    networks:
      # Connect the main app to the same network used by sandbox containers
      - agent_sandbox_net
    # depends_on: # Add dependencies if using services like a database
    #   - db
    restart: unless-stopped
    # Healthcheck (optional, basic example - checks if Flask is responding)
    # healthcheck:
    #   test: ["CMD", "curl", "-f", "http://localhost:${APP_PORT:-5000}/health"]
    #   interval: 30s
    #   timeout: 10s
    #   retries: 3
    #   start_period: 15s # Give time for gunicorn to start

  # Optional Database Service (Example: PostgreSQL)
  # db:
  #   image: postgres:15
  #   container_name: nlp_agent_sandbox_db
  #   environment:
  #     POSTGRES_DB: ${POSTGRES_DB:-agent_sandbox}
  #     POSTGRES_USER: ${POSTGRES_USER:-agent_user}
  #     POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-replace_with_strong_password}
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   networks:
  #     - agent_sandbox_net
  #   restart: unless-stopped
  #   healthcheck:
  #      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-agent_user} -d ${POSTGRES_DB:-agent_sandbox}"]
  #      interval: 10s
  #      timeout: 5s
  #      retries: 5

# Define the network for communication between the app and sandbox containers
networks:
  agent_sandbox_net:
    driver: bridge
    name: ${SANDBOX_NETWORK_NAME:-agent_sandbox_net} # Use name from .env or default

# Optional: Define persistent volume for database data
# volumes:
#   postgres_data:

EOF
echo "Created docker-compose.yml"

# --- Create sandbox_workspace/README.md ---
cat << EOF > sandbox_workspace/README.md
# Agent Workspace

This directory (`$(pwd)/sandbox_workspace` on the host) is mounted into the `/workspace` directory inside each agent's Docker sandbox container.

Files created or modified by the agent using tools like `file_system.write_file` will appear here on the host machine and be accessible to the agent at `/workspace/filename`.

The agent can also read files placed here on the host using `file_system.read_file('/workspace/filename')` or list them with `file_system.list_files('.')`.
EOF
echo "Created sandbox_workspace/README.md"

# --- Create templates/base.html ---
cat << 'EOF' > templates/base.html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}NLP Agent Sandbox{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🤖</text></svg>">
    {% block head_extra %}{% endblock %}
</head>
<body>
    <div class="container">
        <header>
            <h1>NLP Agent Sandbox <span id="llm-provider-display"></span></h1>
            <p>Interact with an AI agent using Gemini or Hugging Face.</p>
        </header>
        <main>
            {% block content %}{% endblock %}
        </main>
        <footer>
            <p>Status: <span id="socket-status" title="WebSocket Connection Status">Disconnected</span></p>
        </footer>
    </div>
    <!-- Socket.IO Client Library -->
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js" integrity="sha384-2huaZvOR9iDzHqslqwpR87isEmrfxqyWOF7hr7BY6KG0+hVKLoEXMPUJw3ynWuhO" crossorigin="anonymous"></script>
    <!-- Custom JS -->
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
EOF
echo "Created templates/base.html"

# --- Create templates/index.html ---
cat << 'EOF' > templates/index.html
{% extends "base.html" %}

{% block title %}Agent Interaction{% endblock %}

{% block content %}
<div class="agent-controls">
    <h2>Agent Control</h2>
    <div class="control-group">
        <label for="persona-select">Select Persona:</label>
        <select id="persona-select">
            <option value="assistant" selected>Assistant (Default)</option>
            <option value="lovable_agent">Lovable Agent (Sparkle)</option>
            <option value="cursor_agent">Cursor-like Agent (CodeHelper)</option>
            <!-- Add more personas here -->
        </select>
    </div>
    <div class="control-group">
         <label for="initial-prompt">Initial Goal/Task:</label>
         <textarea id="initial-prompt" rows="3" placeholder="Enter the initial task for the agent (e.g., 'Write a python script in /workspace/hello.py that prints hello world.')"></textarea>
    </div>
    <button id="start-agent-btn" title="Start a new agent session">Start Agent</button>
    <button id="stop-agent-btn" disabled title="Stop the currently running agent">Stop Agent</button>
    <p>Agent Status: <strong id="agent-status" data-state="Idle">Idle</strong></p>
    <p>Session ID: <span id="session-id" class="mono">N/A</span></p>
</div>

<div class="interaction-area">
    <h2>Interaction Log</h2>
    <div id="log-container" aria-live="polite" aria-atomic="false">
        <!-- Log messages will appear here -->
        <p class="log-system"><strong>[SYSTEM]</strong> Welcome! Select a persona, enter a task, and click Start Agent.</p>
    </div>
    <div class="message-input">
        <input type="text" id="user-message" placeholder="Send a message to the running agent..." disabled>
        <button id="send-message-btn" disabled>Send</button>
    </div>
</div>

<!-- Basic File Browser -->
<div class="file-browser">
    <h2>Sandbox Workspace (<span class="mono">/workspace</span>)</h2>
    <div class="file-controls">
        <button id="refresh-files-btn" disabled title="Refresh file list">Refresh</button>
        <span id="file-loading-status"></span>
    </div>
    <div id="file-list">
        <p><em>Start an agent to view workspace files.</em></p>
        <!-- File list will be populated here -->
    </div>
    <div id="file-preview">
        <h3>File Preview <span id="preview-filename" class="mono"></span></h3>
        <pre id="file-content" class="mono">Select a file to preview its content.</pre>
        <span id="preview-loading-status"></span>
    </div>
</div>
{% endblock %}
EOF
echo "Created templates/index.html"

# --- Create static/css/style.css ---
cat << 'EOF' > static/css/style.css
:root {
    --primary-color: #4CAF50; /* Green */
    --secondary-color: #f4f4f4;
    --background-color: #fff;
    --text-color: #333;
    --border-color: #ddd;
    --error-color: #d9534f;
    --info-color: #5bc0de;
    --warning-color: #f0ad4e;
    --tool-call-color: #e97500;
    --tool-result-color: #6f42c1;
    --user-color: #007bff;
    --agent-color: #28a745;
    --system-color: #6c757d;
    --font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    --monospace-font: 'Courier New', Courier, monospace;
}

body {
    font-family: var(--font-family);
    line-height: 1.6;
    margin: 0;
    padding: 0;
    background-color: var(--secondary-color);
    color: var(--text-color);
    font-size: 16px;
}

.container {
    max-width: 1400px;
    margin: 20px auto;
    padding: 20px;
    background-color: var(--background-color);
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    display: grid;
    grid-template-columns: 350px 1fr; /* Fixed controls, flexible interaction+files */
    grid-template-rows: auto minmax(400px, auto) auto; /* Header, Main Content, Footer */
    grid-template-areas:
        "header header"
        "controls interaction"
        "files interaction" /* File browser below controls */
        "footer footer";
    gap: 25px;
}

header {
    grid-area: header;
    text-align: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 15px;
    margin-bottom: 10px;
}

header h1 {
    margin-bottom: 5px;
    color: #444;
}
#llm-provider-display {
    font-size: 0.7em;
    font-weight: normal;
    color: var(--system-color);
    vertical-align: middle;
}


main {
    display: contents; /* Allow grid items to flow directly */
}

.agent-controls {
    grid-area: controls;
    padding: 20px;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    background-color: #fdfdfd;
}

.agent-controls h2, .interaction-area h2, .file-browser h2 {
    margin-top: 0;
    color: #555;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    margin-bottom: 20px;
    font-size: 1.2em;
    font-weight: 600;
}

.control-group {
    margin-bottom: 18px;
}

.control-group label {
    display: block;
    margin-bottom: 6px;
    font-weight: 600;
    font-size: 0.95em;
}

.control-group select, .control-group textarea, .message-input input[type="text"] {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding in width */
    font-size: 0.95em;
}
.control-group textarea {
    resize: vertical;
    min-height: 70px;
}
.control-group select:focus, .control-group textarea:focus, .message-input input[type="text"]:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
}


button {
    padding: 10px 18px;
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 10px;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
    font-size: 0.95em;
    font-weight: 500;
}
button:hover {
    background-color: #45a049;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
button:active {
    background-color: #3c8b40;
}
button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
    box-shadow: none;
}

#stop-agent-btn {
    background-color: var(--error-color);
}
#stop-agent-btn:hover {
    background-color: #c9302c;
}
#stop-agent-btn:disabled {
    background-color: #ccc;
}


#agent-status {
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block; /* Allow padding */
    margin-left: 5px;
    min-width: 60px; /* Give space for text */
    text-align: center;
}
/* Status-specific styling could be added here via JS */
#agent-status[data-state="Running"],
#agent-status[data-state="Thinking"],
#agent-status[data-state="Calling Tool"],
#agent-status[data-state="Processing Tool Result"] { background-color: #e6f4ea; color: #3c763d; }
#agent-status[data-state="Error"] { background-color: #f2dede; color: #a94442; }
#agent-status[data-state="Stopped"],
#agent-status[data-state="Idle"],
#agent-status[data-state="Finished"] { background-color: #eee; color: #333; }
#agent-status[data-state="Starting"],
#agent-status[data-state="Stopping"] { background-color: #fcf8e3; color: #8a6d3b; }


.mono {
    font-family: var(--monospace-font);
    font-size: 0.9em;
    background-color: #eee;
    padding: 2px 5px;
    border-radius: 3px;
    word-break: break-all;
}
#session-id {
    font-size: 0.85em;
}


.interaction-area {
    grid-area: interaction;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 20px;
    background-color: #fdfdfd;
}

#log-container {
    flex-grow: 1;
    min-height: 400px; /* Ensure minimum height */
    max-height: 60vh; /* Limit max height based on viewport */
    overflow-y: auto; /* Enable scrolling */
    border: 1px solid #eee;
    padding: 15px;
    margin-bottom: 15px;
    background-color: var(--background-color);
    font-size: 0.95em;
    line-height: 1.5;
    scroll-behavior: smooth;
}
#log-container p {
    margin: 8px 0;
    padding-bottom: 8px;
    border-bottom: 1px dotted #eee;
    word-wrap: break-word; /* Wrap long words/URLs */
}
#log-container p:last-child {
    border-bottom: none;
}
#log-container strong { /* Style prefixes like [USER] */
    margin-right: 8px;
    font-weight: 600;
}
#log-container pre { /* Style JSON data blocks */
    background-color: #f8f8f8;
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 10px;
    margin-top: 5px;
    font-family: var(--monospace-font);
    font-size: 0.9em;
    white-space: pre-wrap; /* Wrap long lines in pre */
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
}

/* Log message type styling */
.log-user strong { color: var(--user-color); }
.log-agent strong { color: var(--agent-color); }
.log-tool-call strong { color: var(--tool-call-color); }
.log-tool-call pre { border-left: 3px solid var(--tool-call-color); }
.log-tool-result strong { color: var(--tool-result-color); }
.log-tool-result pre { border-left: 3px solid var(--tool-result-color); }
.log-status strong, .log-system strong { color: var(--system-color); }
.log-error strong { color: var(--error-color); }
.log-error pre { border-left: 3px solid var(--error-color); background-color: #f2dede;}
.log-thinking strong { color: #888; font-style: italic; }
.log-final strong { color: var(--agent-color); font-weight: bold; }


.message-input {
    display: flex;
    margin-top: auto; /* Push to bottom */
}
.message-input input[type="text"] {
    flex-grow: 1;
    border-radius: 4px 0 0 4px;
}
.message-input button {
    border-radius: 0 4px 4px 0;
    margin-left: -1px; /* Overlap border */
}

.file-browser {
    grid-area: files;
    padding: 20px;
    border: 1px solid var(--border-color);
    border-radius: 5px;
    background-color: #fdfdfd;
    display: flex;
    flex-direction: column;
    min-height: 300px; /* Give it some minimum space */
}
.file-controls {
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}
.file-controls button {
    padding: 6px 12px;
    font-size: 0.9em;
    margin-right: 10px;
}
#file-loading-status, #preview-loading-status {
    font-size: 0.85em;
    color: var(--system-color);
    margin-left: 10px;
}


#file-list {
    height: 200px; /* Adjust as needed */
    overflow-y: auto;
    border: 1px solid #eee;
    background: var(--background-color);
    padding: 8px;
    margin-bottom: 15px;
    font-size: 0.9em;
}
#file-list ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
#file-list li {
    cursor: pointer;
    padding: 4px 8px;
    border-bottom: 1px dotted #eee;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-radius: 3px;
}
#file-list li:hover {
    background-color: #eef;
}
#file-list li.selected {
    background-color: #ddeeff;
    font-weight: 600;
}
#file-list li.is-dir {
    color: #555; /* Style directories differently */
    cursor: default; /* Indicate non-clickable (for now) */
}
#file-list li.is-dir:hover {
    background-color: transparent; /* No hover effect for dirs */
}


#file-preview {
    flex-grow: 1;
    border: 1px solid #eee;
    background: var(--background-color);
    padding: 15px;
    margin-top: 10px; /* Space between list and preview */
    display: flex;
    flex-direction: column;
}
#file-preview h3 {
    margin-top: 0;
    font-size: 1em;
    color: #555;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
#preview-filename {
    font-weight: normal;
    font-size: 0.9em;
}

#file-content {
    flex-grow: 1;
    white-space: pre-wrap; /* Wrap long lines */
    word-wrap: break-word; /* Break words if necessary */
    font-size: 0.85em;
    line-height: 1.4;
    overflow-y: auto; /* Scroll long file content */
    background-color: #f9f9f9;
    border: 1px solid #eee;
    padding: 10px;
    min-height: 100px; /* Ensure preview area has some height */
}


footer {
    grid-area: footer;
    text-align: center;
    margin-top: 20px;
    padding-top: 15px;
    border-top: 1px solid var(--border-color);
    font-size: 0.9em;
    color: var(--system-color);
}
#socket-status {
    font-weight: bold;
    margin-left: 5px;
}
#socket-status.connected { color: var(--primary-color); }
#socket-status.disconnected { color: var(--error-color); }


/* Responsive adjustments */
@media (max-width: 1100px) {
    .container {
        grid-template-columns: 300px 1fr; /* Slightly smaller controls */
        gap: 20px;
        padding: 15px;
    }
}

@media (max-width: 800px) {
    .container {
        grid-template-columns: 1fr; /* Single column */
        grid-template-rows: auto auto auto auto auto; /* Adjust rows */
        grid-template-areas:
            "header"
            "controls"
            "interaction"
            "files"
            "footer";
        max-width: 95%;
    }
    #log-container {
        max-height: 50vh; /* Adjust height for smaller screens */
    }
    .file-browser {
        min-height: auto; /* Remove min-height */
    }
    #file-list {
        height: 150px;
    }
}

EOF
echo "Created static/css/style.css"

# --- Create static/js/main.js ---
cat << 'EOF' > static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const API_BASE_URL = '/api/agent';
    const MAX_LOG_MESSAGES = 150; // Limit messages in the log container

    // --- DOM Elements ---
    const startAgentBtn = document.getElementById('start-agent-btn');
    const stopAgentBtn = document.getElementById('stop-agent-btn');
    const personaSelect = document.getElementById('persona-select');
    const initialPromptInput = document.getElementById('initial-prompt');
    const agentStatusSpan = document.getElementById('agent-status');
    const sessionIdSpan = document.getElementById('session-id');
    const logContainer = document.getElementById('log-container');
    const userMessageInput = document.getElementById('user-message');
    const sendMessageBtn = document.getElementById('send-message-btn');
    const fileListDiv = document.getElementById('file-list');
    const refreshFilesBtn = document.getElementById('refresh-files-btn');
    const filePreviewPre = document.getElementById('file-content');
    const fileLoadingStatusSpan = document.getElementById('file-loading-status');
    const previewFilenameSpan = document.getElementById('preview-filename');
    const previewLoadingStatusSpan = document.getElementById('preview-loading-status');
    const socketStatusSpan = document.getElementById('socket-status');
    const llmProviderSpan = document.getElementById('llm-provider-display');

    // --- State ---
    let currentSessionId = null;
    let socket = null;
    let selectedFilePath = null;
    let llmProvider = 'Unknown'; // Will be fetched

    // --- Utility Functions ---
    function addLogMessage(type, message, data = null) {
        const p = document.createElement('p');
        p.classList.add(`log-${type}`); // e.g., log-user, log-agent

        // Sanitize message to prevent HTML injection
        const sanitizedMessage = escapeHtml(message);
        let content = `<strong>[${type.toUpperCase()}]</strong> ${sanitizedMessage}`;

        if (data) {
            // Basic formatting for objects/arrays
            let dataStr;
            try {
                // Pretty print JSON if possible
                dataStr = typeof data === 'object' ? JSON.stringify(data, null, 2) : String(data);
            } catch (e) {
                dataStr = String(data); // Fallback to string conversion
            }

            // Limit length of displayed data to avoid huge logs
            const maxDataLength = 500;
            if (dataStr.length > maxDataLength) {
                 dataStr = dataStr.substring(0, maxDataLength) + '... (truncated)';
            }
            // Sanitize data string before adding as HTML
            content += `<br><pre>${escapeHtml(dataStr)}</pre>`;
        }
        p.innerHTML = content; // Use innerHTML because we already sanitized and added <pre>

        logContainer.appendChild(p);

        // Auto-scroll to bottom only if user is near the bottom
        const isScrolledToBottom = logContainer.scrollHeight - logContainer.clientHeight <= logContainer.scrollTop + 30;
        if (isScrolledToBottom) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }

        // Limit number of log messages
        while (logContainer.children.length > MAX_LOG_MESSAGES) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            try {
                return String(unsafe); // Attempt to convert non-strings
            } catch (e) {
                return ''; // Return empty string if conversion fails
            }
        }
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    function updateAgentStatusUI(statusText) {
         agentStatusSpan.textContent = statusText;
         agentStatusSpan.dataset.state = statusText; // For potential CSS styling based on state
    }

    function updateUIState(agentRunning) {
        startAgentBtn.disabled = agentRunning;
        stopAgentBtn.disabled = !agentRunning;
        userMessageInput.disabled = !agentRunning;
        sendMessageBtn.disabled = !agentRunning;
        refreshFilesBtn.disabled = !agentRunning;
        personaSelect.disabled = agentRunning;
        initialPromptInput.disabled = agentRunning;

        if (!agentRunning) {
            updateAgentStatusUI('Idle');
            sessionIdSpan.textContent = 'N/A';
            currentSessionId = null;
            // Don't clear file list on stop, keep it visible
            // fileListDiv.innerHTML = '<p><em>Start an agent to view workspace files.</em></p>';
            // filePreviewPre.textContent = 'Select a file to preview its content.';
            // previewFilenameSpan.textContent = '';
            // selectedFilePath = null;
        } else {
             sessionIdSpan.textContent = currentSessionId || 'Starting...';
             // Status will be updated by WebSocket events
        }
    }

    function clearLogs() {
        logContainer.innerHTML = '';
         addLogMessage('system', 'Logs cleared.');
    }

    function updateSocketStatus(connected) {
        if (connected) {
            socketStatusSpan.textContent = 'Connected';
            socketStatusSpan.className = 'connected';
            socketStatusSpan.title = 'WebSocket Connected';
        } else {
            socketStatusSpan.textContent = 'Disconnected';
            socketStatusSpan.className = 'disconnected';
            socketStatusSpan.title = 'WebSocket Disconnected';
        }
    }

    function connectWebSocket(sessionId) {
        if (socket && socket.connected) {
            console.log('WebSocket already connected.');
            return;
        }

        // Disconnect previous socket if exists
        if (socket) {
            socket.disconnect();
        }

        console.log('Attempting WebSocket connection for session:', sessionId);
        updateSocketStatus(false); // Show as disconnected initially

        // Connect to the Socket.IO server, passing session_id in query
        socket = io({
            query: { session_id: sessionId },
            reconnectionAttempts: 3, // Limit reconnection attempts
            timeout: 10000 // Connection timeout
        });

        socket.on('connect', () => {
            console.log('WebSocket connected successfully (SID:', socket.id, 'Session:', sessionId, ')');
            addLogMessage('status', 'WebSocket connected.');
            updateSocketStatus(true);
            // Request initial workspace state after connection
            socket.emit('request_workspace_update', { session_id: sessionId });
        });

        socket.on('disconnect', (reason) => {
            console.warn('WebSocket disconnected:', reason);
            addLogMessage('status', `WebSocket disconnected: ${reason}.`);
            updateSocketStatus(false);
            // If the agent is supposed to be running, indicate potential issue
            if (currentSessionId === sessionId && startAgentBtn.disabled) {
                 addLogMessage('error', 'Connection lost. Agent might have stopped or encountered an issue.');
                 // Optionally try to fetch agent status via HTTP as a fallback check
                 // fetchAgentStatus(sessionId);
            }
            socket = null; // Clear socket object
        });

        socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            addLogMessage('error', `WebSocket connection error: ${error.message}`);
            updateSocketStatus(false);
            // Consider stopping the agent UI if connection fails persistently
            // updateUIState(false);
            socket = null; // Clear socket object
        });

        // --- Agent Event Handlers ---
        socket.on('agent_status', (data) => {
            console.log('Agent Status Update:', data);
            if (data.session_id !== currentSessionId) return; // Ignore messages for other sessions

            updateAgentStatusUI(data.status);
            if (data.message) {
                addLogMessage('status', data.message);
            }
            // Update UI based on terminal states
            if (['Stopped', 'Error', 'Finished'].includes(data.status)) {
                 updateUIState(false);
                 disconnectWebSocket(); // Clean up socket if agent is definitively stopped
            } else {
                 updateUIState(true); // Ensure UI reflects running state otherwise
             }
        });

        socket.on('agent_response', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.log('Agent Response:', data.message.substring(0, 100) + '...');
            addLogMessage('agent', data.message);
        });

         socket.on('agent_thinking', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.log('Agent Thinking');
            addLogMessage('thinking', data.message || 'Thinking...');
            updateAgentStatusUI('Thinking'); // Explicitly set thinking state
        });

        socket.on('tool_call', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.log('Tool Call:', data);
            addLogMessage('tool-call', `Using tool: ${data.tool_name}`, data.tool_args);
             updateAgentStatusUI('Calling Tool');
        });

        socket.on('tool_result', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.log('Tool Result:', data);
            addLogMessage('tool-result', `Result from ${data.tool_name}:`, data.result);
             updateAgentStatusUI('Processing Tool Result');
            // Refresh file list automatically after file system operations for immediate feedback
            if (data.tool_name && data.tool_name.startsWith('file_system.')) {
                fetchFileList(true); // Pass true to indicate it's an auto-refresh
            }
        });

         socket.on('final_answer', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.log('Final Answer:', data.message.substring(0, 100) + '...');
            addLogMessage('final', data.message);
            updateAgentStatusUI('Finished');
            updateUIState(false); // Assume finished after final answer
            disconnectWebSocket();
        });

        socket.on('agent_error', (data) => {
            if (data.session_id !== currentSessionId) return;
            console.error('Agent Error:', data);
            addLogMessage('error', `Agent error: ${data.error}`, data.details);
            updateAgentStatusUI('Error');
            updateUIState(false); // Assume stopped on error
            disconnectWebSocket();
        });

         socket.on('workspace_update', (data) => {
             if (data.session_id !== currentSessionId) return;
             console.log('Workspace Update Received:', data);
             renderFileList(data.files);
         });

    }

    function disconnectWebSocket() {
        if (socket) {
            console.log("Disconnecting WebSocket explicitly.");
            socket.disconnect();
            socket = null;
            updateSocketStatus(false);
        }
    }


    // --- API Call Functions ---
    async function fetchAgentStatus(sessionId) {
        if (!sessionId) return;
        try {
            const response = await fetch(`${API_BASE_URL}/status/${sessionId}`);
            if (!response.ok) {
                console.warn(`Failed to fetch status for ${sessionId}: ${response.status}`);
                // If agent not found (404), update UI accordingly
                if (response.status === 404 && currentSessionId === sessionId) {
                     updateUIState(false);
                     disconnectWebSocket();
                }
                return;
            }
            const data = await response.json();
            console.log("Fetched agent status:", data);
            if (data.session_id === currentSessionId) {
                 updateAgentStatusUI(data.state);
                 if (['Stopped', 'Error', 'Finished'].includes(data.state)) {
                     updateUIState(false);
                     disconnectWebSocket();
                 } else {
                      updateUIState(true); // Ensure UI is enabled if agent is running
                 }
            }
        } catch (error) {
            console.error('Error fetching agent status:', error);
             // Potentially update UI to reflect uncertainty
        }
    }

    async function startAgent() {
        const persona = personaSelect.value;
        const initial_prompt = initialPromptInput.value.trim();

        if (!initial_prompt) {
             alert("Please provide an initial goal/task for the agent.");
             initialPromptInput.focus();
             return;
        }

        clearLogs();
        addLogMessage('status', 'Starting agent session...');
        updateUIState(true); // Tentatively set UI to running
        updateAgentStatusUI('Starting...');
        sessionIdSpan.textContent = 'Requesting...';
        fileListDiv.innerHTML = '<p><em>Loading workspace...</em></p>'; // Clear file list
        filePreviewPre.textContent = 'Select a file to preview its content.';
        previewFilenameSpan.textContent = '';
        selectedFilePath = null;


        try {
            const response = await fetch(`${API_BASE_URL}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ persona, initial_prompt }),
            });

            const data = await response.json();

            if (response.ok && data.session_id) {
                currentSessionId = data.session_id;
                sessionIdSpan.textContent = currentSessionId;
                addLogMessage('status', `Agent session starting... Session ID: ${currentSessionId}`);
                connectWebSocket(currentSessionId);
                // File list will be fetched on WebSocket connect
            } else {
                throw new Error(data.error || `Failed to start agent (HTTP ${response.status})`);
            }
        } catch (error) {
            console.error('Error starting agent:', error);
            addLogMessage('error', `Failed to start agent: ${error.message}`);
            updateUIState(false);
        }
    }

    async function stopAgent() {
        if (!currentSessionId) {
            addLogMessage('error', 'No active session to stop.');
            return;
        }

        const sessionToStop = currentSessionId; // Capture session ID before potential changes
        addLogMessage('status', `Requesting stop for agent session: ${sessionToStop}`);
        updateAgentStatusUI('Stopping...');
        stopAgentBtn.disabled = true; // Disable stop button immediately

        try {
            const response = await fetch(`${API_BASE_URL}/stop/${sessionToStop}`, {
                method: 'POST',
            });

            const data = await response.json();

            if (response.ok) {
                addLogMessage('status', `Agent stop request sent for ${sessionToStop}. ${data.message || ''}`);
                // UI state (buttons disabled, status=Stopped) will be fully updated by agent_status event or disconnect
            } else {
                 // Log error but rely on status updates or manual refresh if stop fails
                addLogMessage('error', `Failed to send stop request (HTTP ${response.status}): ${data.error || 'Unknown error'}`);
                stopAgentBtn.disabled = false; // Re-enable button if stop failed
            }
        } catch (error) {
            console.error('Error stopping agent:', error);
            addLogMessage('error', `Failed to stop agent: ${error.message}`);
            stopAgentBtn.disabled = false; // Re-enable button on network error etc.
        } finally {
            // Don't disconnect WebSocket here immediately, wait for confirmation event
            // updateUIState(false); // Wait for event
        }
    }

    async function sendMessage() {
        const message = userMessageInput.value.trim();
        if (!message || !currentSessionId || !socket || !socket.connected) {
            addLogMessage('error', 'Cannot send message. Agent not running or WebSocket not connected.');
            return;
        }

        addLogMessage('user', message);
        userMessageInput.value = ''; // Clear input field

        // Send message via WebSocket
        console.log(`Sending message to session ${currentSessionId}: ${message.substring(0,50)}...`);
        socket.emit('user_message', { session_id: currentSessionId, message: message });
    }

    async function fetchFileList(isAutoRefresh = false) {
        if (!currentSessionId) {
            fileListDiv.innerHTML = '<p><em>Start an agent to view workspace files.</em></p>';
            fileLoadingStatusSpan.textContent = '';
            return;
        }
        console.log("Fetching file list for session:", currentSessionId);
        if (!isAutoRefresh) {
             fileListDiv.innerHTML = '<p><em>Loading files...</em></p>'; // Indicate loading only on manual refresh
             fileLoadingStatusSpan.textContent = 'Loading...';
        }


        try {
            const response = await fetch(`${API_BASE_URL}/workspace/${currentSessionId}/files`);
            const data = await response.json();

            if (response.ok) {
                 console.log("File list data:", data);
                 renderFileList(data.files || []); // Handle case where 'files' might be missing
                 fileLoadingStatusSpan.textContent = ''; // Clear loading status
            } else {
                throw new Error(data.error || `Failed to fetch file list (HTTP ${response.status})`);
            }
        } catch (error) {
            console.error('Error fetching file list:', error);
            addLogMessage('error', `Failed to fetch file list: ${error.message}`);
            fileListDiv.innerHTML = `<p><em>Error loading files.</em></p>`;
            fileLoadingStatusSpan.textContent = 'Error';
        }
    }

     function renderFileList(files) {
         fileListDiv.innerHTML = ''; // Clear previous list
         if (!files || files.length === 0) {
             fileListDiv.innerHTML = '<p><em>Workspace is empty.</em></p>';
             return;
         }

         const ul = document.createElement('ul');
         // Sort files: directories first, then alphabetically
         files.sort((a, b) => {
             if (a.is_dir !== b.is_dir) {
                 return a.is_dir ? -1 : 1; // Directories first
             }
             return a.name.localeCompare(b.name); // Then alphabetically
         });

         files.forEach(file => {
             const li = document.createElement('li');
             const icon = file.is_dir ? '📁' : '📄';
             li.textContent = `${icon} ${file.name}`;
             li.dataset.path = file.path; // Store full path relative to workspace
             li.dataset.isdir = file.is_dir;
             li.title = file.path; // Show full path on hover

             if (file.is_dir) {
                 li.classList.add('is-dir');
             } else {
                 // Add click listener only for files
                 li.addEventListener('click', () => {
                     // Deselect previous
                     const currentlySelected = ul.querySelector('li.selected');
                     if (currentlySelected) {
                         currentlySelected.classList.remove('selected');
                     }
                     // Select current
                     li.classList.add('selected');
                     selectedFilePath = file.path;
                     fetchFileContent(file.path);
                 });
             }

             // Highlight if it's the currently selected file for preview
             if (!file.is_dir && selectedFilePath === file.path) {
                 li.classList.add('selected');
             }

             ul.appendChild(li);
         });
         fileListDiv.appendChild(ul);
     }

    async function fetchFileContent(filePath) {
         if (!currentSessionId) {
             addLogMessage('error', 'Cannot fetch file content: No active session.');
             return;
         }
         if (!filePath) {
             filePreviewPre.textContent = 'No file selected.';
             previewFilenameSpan.textContent = '';
             previewLoadingStatusSpan.textContent = '';
             return;
         }

         console.log(`Fetching content for: ${filePath}`);
         filePreviewPre.textContent = ''; // Clear previous content
         previewFilenameSpan.textContent = filePath;
         previewLoadingStatusSpan.textContent = 'Loading...';

         try {
             // Use the file system tool endpoint for reading
             const encodedPath = encodeURIComponent(filePath); // Ensure path is URL-safe
             const response = await fetch(`${API_BASE_URL}/workspace/${currentSessionId}/files/${encodedPath}`);
             const data = await response.json();

             if (response.ok) {
                 filePreviewPre.textContent = data.content !== null ? data.content : '(File is empty or binary)';
                 previewLoadingStatusSpan.textContent = '';
             } else {
                  throw new Error(data.error || `Failed to fetch file content (HTTP ${response.status})`);
             }
         } catch (error) {
             console.error('Error fetching file content:', error);
             addLogMessage('error', `Failed to fetch content for ${filePath}: ${error.message}`);
             filePreviewPre.textContent = `Error loading content: ${escapeHtml(error.message)}`;
             previewLoadingStatusSpan.textContent = 'Error';
         }
     }

     async function fetchHealth() {
         try {
             const response = await fetch('/health');
             if (!response.ok) {
                 llmProviderSpan.textContent = '(Error fetching status)';
                 return;
             }
             const data = await response.json();
             llmProvider = data.llm_provider || 'Unknown';
             llmProviderSpan.textContent = `(${llmProvider.charAt(0).toUpperCase() + llmProvider.slice(1)})`;
         } catch (error) {
             console.error("Error fetching health:", error);
             llmProviderSpan.textContent = '(Offline)';
         }
     }


    // --- Event Listeners ---
    startAgentBtn.addEventListener('click', startAgent);
    stopAgentBtn.addEventListener('click', stopAgent);

    sendMessageBtn.addEventListener('click', sendMessage);
    userMessageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !sendMessageBtn.disabled) {
            sendMessage();
        }
    });

     refreshFilesBtn.addEventListener('click', () => fetchFileList(false)); // Manual refresh

    // --- Initial State ---
    fetchHealth(); // Get LLM provider info
    updateUIState(false); // Start with UI disabled
    updateSocketStatus(false);

    // Optional: Try to fetch status of potentially running agents on page load?
    // Could be complex if multiple browser tabs are possible.
    // For simplicity, assume clean start on page load.

     // Graceful shutdown handling (best effort)
     window.addEventListener('beforeunload', (event) => {
         if (currentSessionId && startAgentBtn.disabled) { // Check if agent is likely running
             // Standard way to prompt user is complex and often blocked.
             // Use navigator.sendBeacon as the most reliable way to send a small
             // amount of data synchronously on unload.
             console.log("Attempting to signal agent stop on page unload...");
             const url = `${API_BASE_URL}/stop/${currentSessionId}`;
             // sendBeacon expects data, can be small JSON or FormData
             const data = new Blob([JSON.stringify({unload: true})], { type: 'application/json' });
             navigator.sendBeacon(url, data);
             // Note: The backend /stop endpoint doesn't currently *use* the body,
             // but sendBeacon requires data. The backend just needs to receive the POST.
         }
     });

});
EOF
echo "Created static/js/main.js"

# --- Create agents/base_agent.py ---
# This is the core logic with Gemini/HF integration
cat << 'EOF' > agents/base_agent.py
import threading
import time
import logging
import json
import traceback
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple, Optional, Union

from config import Config
from services.tool_executor import ToolExecutor, ToolNotFoundException, ToolExecutionException, format_tool_schema, format_tool_schemas_for_prompt
from services.sandbox_manager import SandboxManager

# --- LLM Client Imports & Setup ---
llm_provider = Config.LLM_PROVIDER
log = logging.getLogger(__name__)

# Conditional imports based on chosen provider
if llm_provider == 'gemini':
    try:
        import google.generativeai as genai
        # Specific types needed for function calling and history
        from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig, Tool, FunctionDeclaration, Content, Part, FunctionCall, FunctionResponse
        log.info("Imported Google Generative AI library.")
    except ImportError:
        log.error("Google Generative AI library not found. Please install 'google-generativeai'")
        # Propagate error or exit? For now, log and continue, init will fail.
        genai = None # Ensure it's None if import fails
elif llm_provider == 'huggingface':
    try:
        from huggingface_hub import InferenceClient
        log.info("Imported Hugging Face Hub library (InferenceClient).")
    except ImportError:
        log.error("Hugging Face Hub library not found. Please install 'huggingface_hub'")
        InferenceClient = None # Ensure it's None if import fails
else:
    log.error(f"Unsupported LLM_PROVIDER configured: '{llm_provider}'")


# --- Constants ---
MAX_CONSECUTIVE_TOOL_CALLS = 5
MAX_TOOL_RETRIES = 2 # Retries if tool execution itself fails

class AgentState:
    IDLE = "Idle"
    STARTING = "Starting"
    RUNNING = "Running"
    WAITING_FOR_USER = "Waiting for User Input" # Not used in this autonomous loop
    THINKING = "Thinking"
    CALLING_TOOL = "Calling Tool"
    PROCESSING_TOOL_RESULT = "Processing Tool Result"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    FINISHED = "Finished"
    ERROR = "Error"

class BaseAgent(ABC, threading.Thread):
    """
    Abstract base class for agents. Handles the main execution loop,
    state management, history, LLM interaction (Gemini/HF), and tool execution.
    """
    def __init__(self, session_id: str, persona: str, initial_prompt: str, sandbox_manager: SandboxManager, tool_executor: ToolExecutor, socketio_emitter: Callable):
        super().__init__()
        self.session_id = session_id
        self.persona = persona
        self.initial_prompt = initial_prompt
        self.sandbox_manager = sandbox_manager
        self.tool_executor = tool_executor
        self.socketio_emitter = socketio_emitter # Function to emit messages via SocketIO

        self.state = AgentState.IDLE
        self._stop_event = threading.Event()
        # History stores dictionaries: {'role': 'user'|'assistant'|'tool_call'|'tool_result', 'content': str | dict | Part}
        # For Gemini, 'assistant' content might be a Part containing text or FunctionCall
        # For Gemini, 'tool_result' content should be the Part containing FunctionResponse
        self.history = deque(maxlen=Config.AGENT_HISTORY_MAX_MESSAGES * 2) # Store turns
        self.tools = self._load_tools() # List of callable functions
        self.llm_client = None # Initialized in _initialize_llm_client
        self.gemini_tool_config = None # Specific to Gemini provider

        self.current_iteration = 0
        self.max_iterations = Config.MAX_AGENT_ITERATIONS
        self.last_error = None

        self.name = f"AgentThread-{session_id[:8]}" # Thread name for logging
        self.daemon = True # Allow program to exit even if agent thread is running

        # --- LLM specific initialization ---
        if not self._initialize_llm_client():
             # Initialization failed, prevent agent from starting run loop
             self.state = AgentState.ERROR
             self.last_error = "LLM Client initialization failed."
             # Don't emit here as thread hasn't started; raise exception to signal failure in constructor
             raise RuntimeError("LLM Client initialization failed.")


    # --- Communication & State ---
    def _emit_log(self, level, message, **kwargs):
        """Helper to emit logs via SocketIO (optional) and standard logging."""
        log.log(level, f"Agent {self.session_id}: {message}", extra=kwargs)
        # Example: Emit logs to UI? Could be noisy.
        # self.socketio_emitter('agent_log', {"session_id": self.session_id, "level": logging.getLevelName(level), "message": message})

    def _emit_status(self, status: str, message: str = None):
        self.state = status
        self._emit_log(logging.INFO, f"Status -> {status} {f'- {message}' if message else ''}")
        payload = {"session_id": self.session_id, "status": status}
        if message:
            payload["message"] = message
        self.socketio_emitter('agent_status', payload)

    def _emit_response(self, message: str):
        self._emit_log(logging.INFO, f"Response -> {message[:100]}...")
        self.socketio_emitter('agent_response', {"session_id": self.session_id, "message": message})

    def _emit_thinking(self, message: str = "Thinking..."):
        self._emit_log(logging.INFO, "Thinking...")
        self.socketio_emitter('agent_thinking', {"session_id": self.session_id, "message": message})
        self._emit_status(AgentState.THINKING) # Also update main status

    def _emit_tool_call(self, tool_name: str, tool_args: dict):
        self._emit_log(logging.INFO, f"Calling Tool -> {tool_name}({tool_args})")
        self.socketio_emitter('tool_call', {"session_id": self.session_id, "tool_name": tool_name, "tool_args": tool_args})
        self._emit_status(AgentState.CALLING_TOOL, f"Calling {tool_name}")

    def _emit_tool_result(self, tool_name: str, result: any):
        result_str = str(result)
        log_result = (result_str[:200] + '...') if len(result_str) > 200 else result_str
        self._emit_log(logging.INFO, f"Tool Result -> {tool_name} returned: {log_result}")
        self.socketio_emitter('tool_result', {"session_id": self.session_id, "tool_name": tool_name, "result": result}) # Send potentially large result
        self._emit_status(AgentState.PROCESSING_TOOL_RESULT, f"Processing result from {tool_name}")


    def _emit_final_answer(self, message: str):
        self._emit_log(logging.INFO, f"Final Answer -> {message[:100]}...")
        self.socketio_emitter('final_answer', {"session_id": self.session_id, "message": message})
        self._emit_status(AgentState.FINISHED, "Task completed.")

    def _emit_error(self, error_msg: str, details: str = None):
        self._emit_log(logging.ERROR, f"Error -> {error_msg} {f'Details: {details}' if details else ''}")
        self.last_error = error_msg
        self.socketio_emitter('agent_error', {"session_id": self.session_id, "error": error_msg, "details": details})
        self._emit_status(AgentState.ERROR, error_msg)
        self._stop_event.set() # Stop agent on error

    # --- Tool Handling ---
    def _load_tools(self) -> list:
        """Loads available tools from the ToolExecutor."""
        return self.tool_executor.get_tools()

    def _get_gemini_tool_config(self) -> Optional[Tool]:
         """Formats tools for Gemini API."""
         if llm_provider != 'gemini' or not genai: return None

         gemini_function_declarations = []
         for tool_func in self.tools:
             schema = format_tool_schema(tool_func) # Get OpenAPI-like schema
             if schema:
                 try:
                    # Convert to Gemini FunctionDeclaration
                    declaration = FunctionDeclaration(
                        name=schema['name'],
                        description=schema['description'],
                        parameters=schema['parameters'] # Gemini expects OpenAPI schema format directly
                    )
                    gemini_function_declarations.append(declaration)
                 except Exception as e:
                      log.error(f"Failed to create Gemini FunctionDeclaration for tool {schema.get('name', 'unknown')}: {e}")
             else:
                 log.warning(f"Could not generate schema for tool: {getattr(tool_func, '__name__', 'unknown')}")

         if gemini_function_declarations:
              log.info(f"Prepared {len(gemini_function_declarations)} tools for Gemini.")
              return Tool(function_declarations=gemini_function_declarations)
         else:
              log.warning("No valid tools found or formatted for Gemini.")
              return None

    # --- LLM Interaction ---
    def _get_system_prompt(self) -> str:
        """
        Generates the system prompt based on persona and available tools.
        Adapts based on the LLM provider. Should be overridden by persona classes.
        """
        # Base instructions common to both
        base_instructions = f"""
You are a helpful AI assistant named '{self.persona}'. Your goal is to achieve the user's objectives.
You operate within a secure sandbox environment with access to a workspace at /workspace.
Think step-by-step about how to achieve the user's goal using the available tools.
Follow the specific instructions provided for tool usage precisely.
If you can answer directly without using a tool, do so.
Provide your final answer directly to the user when the task is complete.
Be concise and helpful. Your current iteration is {self.current_iteration}/{self.max_iterations}.
"""

        if llm_provider == 'gemini':
            # Gemini's tool use is triggered by the 'tools' parameter in the API call.
            # The system prompt just needs to set the context and persona.
            tool_names = [getattr(t, '__qualname__', 'unknown') for t in self.tools]
            return base_instructions + f"\nYou have access to tools like: {', '.join(tool_names)}. Use them when necessary."

        elif llm_provider == 'huggingface':
            # Hugging Face needs explicit instructions and tool schemas in the prompt.
            formatted_tools = format_tool_schemas_for_prompt(self.tools)
            if not formatted_tools:
                 formatted_tools = "No tools are available for use."

            hf_tool_instructions = f"""
# Available Tools:
{formatted_tools}

# Tool Usage Instruction:
When you decide to use a tool, you MUST output ONLY a JSON object containing the tool name and arguments, enclosed in triple backticks, like this:
```json
{{
  "tool_name": "module_name.function_name",
  "arguments": {{
    "arg1_name": "value1",
    "arg2_name": value2
  }}
}}
Do not add any other text, explanation, or commentary before or after the JSON block if you are calling a tool. If you are NOT calling a tool, provide your response as plain text. Think step-by-step. Analyze the request and conversation history. Decide if a tool is needed. If yes, choose the correct tool and construct the exact JSON output. If no tool is needed, provide the final answer or ask for clarification. If a tool call fails (you will see an error message), analyze the error and try again with corrected arguments or use a different approach. """ return base_instructions + "\n" + hf_tool_instructions else: return base_instructions # Fallback

def _initialize_llm_client(self) -> bool:
    """Initializes the LLM client based on the configuration. Returns True on success."""
    try:
        if llm_provider == 'gemini':
            if not genai: raise ImportError("Gemini library not loaded.")
            if not Config.GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY is not configured.")

            log.info(f"Initializing Google Gemini client for model: {Config.LLM_MODEL_NAME}")
            genai.configure(api_key=Config.GOOGLE_API_KEY)

            self.gemini_tool_config = self._get_gemini_tool_config()

            # Safety settings (customize as needed)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            # Generation config
            generation_config = GenerationConfig(
                temperature=Config.LLM_TEMPERATURE,
                max_output_tokens=Config.LLM_MAX_TOKENS
                # top_p=0.95, # Example other params
                # top_k=40,
            )

            # Create the model instance
            self.llm_client = genai.GenerativeModel(
                model_name=Config.LLM_MODEL_NAME,
                safety_settings=safety_settings,
                generation_config=generation_config,
                tools=self.gemini_tool_config # Pass formatted tools here
                # system_instruction=self._get_system_prompt() # Newer Gemini models might support this
            )
            # For older models or general approach, system prompt is part of history
            log.info("Google Gemini client initialized.")
            return True

        elif llm_provider == 'huggingface':
             if not InferenceClient: raise ImportError("Hugging Face Hub library not loaded.")
             if not Config.HUGGINGFACE_API_TOKEN and not Config.HUGGINGFACE_INFERENCE_ENDPOINT:
                 raise ValueError("HUGGINGFACE_API_TOKEN (or HUGGINGFACE_INFERENCE_ENDPOINT) is not configured.")

             model_source = Config.HUGGINGFACE_INFERENCE_ENDPOINT or Config.LLM_MODEL_NAME
             log.info(f"Initializing Hugging Face Inference client for: {model_source}")

             self.llm_client = InferenceClient(model=model_source, token=Config.HUGGINGFACE_API_TOKEN)

             # Test connection (optional, but recommended)
             try:
                 self.llm_client.post(json={"inputs": "test", "parameters": {"max_new_tokens": 10}})
                 log.info("Hugging Face Inference client connection tested successfully.")
             except Exception as test_err:
                 log.warning(f"Hugging Face client initialized, but test call failed: {test_err}. Check token/model/endpoint.")
             return True

        else:
            raise NotImplementedError(f"LLM provider '{llm_provider}' is not supported.")

    except Exception as e:
        log.exception(f"Failed to initialize LLM client for provider '{llm_provider}':")
        self.llm_client = None
        self.gemini_tool_config = None # Ensure reset on failure
        return False # Indicate failure

def _format_history_for_llm(self) -> Union[List[Content], str]:
    """Formats the conversation history for the specific LLM."""
    if llm_provider == 'gemini':
        # Gemini expects a list of Content objects.
        # Role must alternate user/model. Tool calls/responses need specific Part types.
        gemini_history: List[Content] = []
        last_role = None

        # Add system prompt concept (usually as first part of first user message or dedicated instruction)
        system_prompt = self._get_system_prompt()
        system_prompt_added = False # Track if added

        for item in self.history:
            role = item['role']
            content = item['content']
            current_part = None
            effective_role = None

            if role == 'user':
                effective_role = 'user'
                text_content = str(content)
                # Prepend system prompt if this is the first user turn conceptually
                if not system_prompt_added:
                    text_content = f"{system_prompt}\n\nUser Task:\n{text_content}"
                    system_prompt_added = True
                current_part = Part(text=text_content)
            elif role == 'assistant':
                effective_role = 'model'
                if isinstance(content, Part): # If we stored the raw Part (text or function_call)
                     current_part = content
                elif isinstance(content, dict) and 'tool_name' in content: # Reconstruct FunctionCall Part
                     current_part = Part(function_call=FunctionCall(name=content['tool_name'], args=content['arguments']))
                else: # Regular text response
                     current_part = Part(text=str(content))
            elif role == 'tool_result':
                 effective_role = 'function' # Use 'function' role for tool results
                 if isinstance(content, Part): # If we stored the raw FunctionResponse Part
                      current_part = content
                 elif isinstance(content, dict) and 'tool_name' in content: # Reconstruct FunctionResponse Part
                       current_part = Part(function_response=FunctionResponse(
                            name=content['tool_name'],
                            response={'result': content['result']} # Gemini expects result in a dict
                       ))
                 else:
                      log.warning(f"Could not format tool_result content: {content}")
                      continue # Skip invalid tool result format

            if current_part and effective_role:
                 # Ensure roles alternate user/model (or function follows model)
                 if gemini_history and effective_role == gemini_history[-1].role:
                      # Handle consecutive roles - merging text is simplest approach
                      can_merge = False
                      if effective_role in ['user', 'model'] and \
                         isinstance(current_part.text, str) and \
                         gemini_history[-1].parts and \
                         isinstance(gemini_history[-1].parts[0].text, str):
                          log.debug(f"Merging consecutive '{effective_role}' text parts.")
                          gemini_history[-1].parts[0].text += "\n" + current_part.text
                          can_merge = True

                      if not can_merge:
                           # Cannot merge (e.g., function calls, different part types).
                           # This indicates an issue in history management or LLM output.
                           # Forcing alternation might hide bugs. Log error.
                           log.error(f"Consecutive roles '{effective_role}' detected and cannot merge. History might be invalid for Gemini.")
                           # Option: Insert placeholder? For now, just log and continue, API call might fail.
                           # gemini_history.append(Content(role='user' if effective_role == 'model' else 'model', parts=[Part(text="(Forced alternation)")]))
                           gemini_history.append(Content(role=effective_role, parts=[current_part]))

                 else:
                      gemini_history.append(Content(role=effective_role, parts=[current_part]))
                 last_role = effective_role

        # Ensure history doesn't start with model/function response
        while gemini_history and gemini_history[0].role != 'user':
             log.debug(f"Removing leading non-user role '{gemini_history[0].role}' from Gemini history.")
             gemini_history.pop(0)

        # log.debug(f"Formatted Gemini History ({len(gemini_history)} turns): {[ (c.role, type(c.parts[0])) for c in gemini_history]}")
        return gemini_history

    elif llm_provider == 'huggingface':
        # Build a single string prompt using a chat template format.
        full_prompt = self._get_system_prompt() + "\n\n# Conversation History:\n"
        history_str_parts = []
        for item in self.history:
            role = item['role']
            content = item['content']
            prefix = ""
            content_str = ""

            if role == 'user':
                prefix = "User:"
                content_str = str(content)
            elif role == 'assistant':
                prefix = "Assistant:"
                if isinstance(content, dict) and 'tool_name' in content:
                    # Format the tool call JSON as the assistant's response
                    try:
                        content_str = f"```json\n{json.dumps(content, indent=2)}\n```"
                    except TypeError:
                        content_str = f"```json\n{json.dumps({'tool_name': content.get('tool_name'), 'arguments': 'SerializationError'}, indent=2)}\n```"
                else:
                    content_str = str(content)
            elif role == 'tool_result':
                 prefix = "Tool Result:"
                 tool_name = content.get('tool_name', 'unknown_tool')
                 result = content.get('result', '')
                 is_error = content.get('is_error', False)
                 status = "[ERROR]" if is_error else "[SUCCESS]"

                 # Truncate long results in the prompt
                 result_str = str(result)
                 max_len = Config.AGENT_TOOL_RESULT_MAX_LEN # Use configured limit
                 if len(result_str) > max_len:
                     result_str = result_str[:max_len] + f"... (truncated, full result was {len(result_str)} chars)"
                 content_str = f"{status} (From tool: {tool_name})\n{result_str}"

            if prefix and content_str is not None: # Check content_str is not None
                 history_str_parts.append(f"{prefix}\n{content_str}\n")

        full_prompt += "\n".join(history_str_parts)
        full_prompt += "\nAssistant:\n" # Prompt the model for its next response

        # log.debug(f"Formatted Hugging Face Prompt (last 500 chars): ...{full_prompt[-500:]}")
        return full_prompt
    else:
        # Basic fallback (should not happen with validation)
        return str([entry['content'] for entry in self.history])

def _call_llm_with_tools(self, formatted_input: Union[List[Content], str]) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Calls the configured LLM (Gemini or Hugging Face) with the formatted history/prompt
    and handles tool calling logic.

    Returns:
        tuple: (response_text: str | None, tool_name: str | None, tool_args: dict | None)
               - response_text is the LLM's text answer.
               - tool_name/tool_args are populated if the LLM requests a tool call.
               Returns (error_message_str, None, None) on API errors.
    """
    if not self.llm_client:
         return "Error: LLM Client not initialized.", None, None

    self._emit_thinking()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            if llm_provider == 'gemini':
                if not isinstance(formatted_input, list):
                    raise TypeError("Gemini requires history as a list of Content objects.")

                # Ensure history doesn't end with 'model' role without a following user/function turn
                if formatted_input and formatted_input[-1].role == 'model':
                     log.warning("History ends with 'model' role. This might cause issues or require empty input.")
                     # Gemini API usually handles this state correctly if the model expects to continue.

                log.debug(f"Sending request to Gemini. History turns: {len(formatted_input)}")

                # Make the API call using generate_content
                response = self.llm_client.generate_content(
                    formatted_input,
                    # tools=self.gemini_tool_config, # Tools already set on model init
                    tool_config={'function_calling_config': 'AUTO'} # Let Gemini decide
                )

                log.debug(f"Gemini raw response candidate: {response.candidates[0] if response.candidates else 'No candidates'}")
                log.debug(f"Gemini prompt feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")


                # IMPORTANT: Check for blocked content *first*
                if not response.candidates:
                     block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                     safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in getattr(response.prompt_feedback, 'safety_ratings', [])])
                     log.error(f"Gemini response blocked or empty. Reason: {block_reason}. Ratings: [{safety_ratings_str}]")
                     return f"Error: Gemini response blocked by safety filters ({block_reason}) or empty.", None, None

                # Process the first candidate
                candidate = response.candidates[0]

                # Check finish reason for safety issues or other problems
                # FinishReason enum values: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER, UNKNOWN, UNSET
                finish_reason = getattr(candidate, 'finish_reason', None)
                if finish_reason not in [candidate.FinishReason.STOP, candidate.FinishReason.MAX_TOKENS]:
                     reason_name = candidate.FinishReason(finish_reason).name if finish_reason else 'UNKNOWN'
                     log.error(f"Gemini generation stopped due to reason: {reason_name}")
                     safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in getattr(candidate, 'safety_ratings', [])])
                     if finish_reason == candidate.FinishReason.SAFETY:
                          return f"Error: Content generation stopped due to safety concerns. Ratings: [{safety_ratings_str}]", None, None
                     else:
                          return f"Error: Content generation stopped unexpectedly (Reason: {reason_name}). Ratings: [{safety_ratings_str}]", None, None

                # Check for function calls in the response parts
                func_call_part = None
                text_part = None
                if candidate.content and candidate.content.parts:
                     for part in candidate.content.parts:
                          if part.function_call:
                               func_call_part = part
                               break # Prioritize function call
                          elif part.text:
                               text_part = part # Store text part if found

                if func_call_part:
                    function_call = func_call_part.function_call
                    tool_name = function_call.name
                    # Convert proto Map to dict robustly
                    tool_args = {}
                    try:
                         # Accessing the underlying _pb map might be fragile, prefer public API if available
                         # For now, this seems common practice:
                         tool_args = dict(function_call.args.items())
                    except Exception as e:
                         log.error(f"Could not convert Gemini tool args to dict: {e}. Args: {function_call.args}")
                         tool_args = {} # Fallback to empty dict

                    log.info(f"Gemini requested tool call: {tool_name} with args: {tool_args}")
                    # Store the raw FunctionCall Part for history
                    self.add_history("assistant", func_call_part)
                    return None, tool_name, tool_args
                elif text_part:
                    # No function call, extract text response
                    text_response = text_part.text
                    log.info("Gemini returned text response.")
                    # Store the raw text Part for history
                    self.add_history("assistant", text_part)
                    return text_response, None, None
                else:
                     # No text and no function call - unexpected state
                     log.error(f"Gemini response had no text or function call. Parts: {candidate.content.parts}")
                     # Check if finish reason was MAX_TOKENS
                     if finish_reason == candidate.FinishReason.MAX_TOKENS:
                          log.warning("Gemini response finished due to MAX_TOKENS without valid content.")
                          return "Error: Response cut short due to maximum token limit.", None, None
                     return "Error: Received empty or unexpected response from Gemini.", None, None

            elif llm_provider == 'huggingface':
                if not isinstance(formatted_input, str):
                    raise TypeError("Hugging Face requires a single string prompt.")

                log.debug(f"Sending prompt to Hugging Face Inference API (last 500 chars): ...{formatted_input[-500:]}")
                params = {
                    "max_new_tokens": Config.LLM_MAX_TOKENS,
                    "temperature": Config.LLM_TEMPERATURE if Config.LLM_TEMPERATURE > 0.01 else None, # Temp 0 can cause issues, use None if very low
                    "return_full_text": False, # Only get the generated part
                    "do_sample": Config.LLM_TEMPERATURE > 0.01, # Only sample if temp > 0.01
                    "top_p": 0.95 if Config.LLM_TEMPERATURE > 0.01 else None,
                    "stop_sequences": ["\nUser:", "\nTool Result:", "```json"], # Try to stop generation before hallucinating next turn or incomplete JSON start
                }
                params = {k: v for k, v in params.items() if v is not None} # Remove None values

                response_text = self.llm_client.text_generation(
                    formatted_input,
                    **params
                )
                response_text = response_text.strip()
                log.debug(f"Hugging Face raw response: {response_text}")

                # --- Parse Hugging Face response for potential tool call ---
                # Append the start sequence back if stopped early, to allow parsing
                if not response_text.endswith("```"):
                     response_text += "```"

                tool_name, tool_args = self._parse_hf_tool_call(response_text)

                if tool_name:
                    log.info(f"Hugging Face response parsed as tool call: {tool_name}")
                    # Store the structured tool call in history
                    self.add_history("assistant", {"tool_name": tool_name, "arguments": tool_args})
                    return None, tool_name, tool_args
                else:
                    log.info("Hugging Face response parsed as text.")
                    # Clean up potential stop sequences if they appear at the very end
                    for seq in ["\nUser:", "\nTool Result:", "```"]:
                        if response_text.endswith(seq):
                            response_text = response_text[:-len(seq)].strip()
                    self.add_history("assistant", response_text) # Store text response
                    return response_text, None, None

            else:
                return "Error: LLM provider not implemented.", None, None

        except Exception as e:
            log.error(f"Error calling LLM (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
            if attempt >= max_retries - 1:
                # self._emit_error("LLM API call failed after multiple retries.", traceback.format_exc())
                return f"Error: LLM call failed after {max_retries} retries. {e}", None, None
            time.sleep(1.5 ** attempt) # Exponential backoff

    return "Error: LLM call failed after retries.", None, None # Should not be reached normally

def _parse_hf_tool_call(self, response_text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Attempts to parse a tool call JSON block ```json ... ``` from Hugging Face model output.
    """
    try:
        import re
        # Regex to find ```json ... ``` block, allowing optional language specifier
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            log.debug(f"Found potential JSON block: {json_str}")
            try:
                # Attempt to strip potential trailing commas for more robust parsing
                json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str)
                data = json.loads(json_str_cleaned)

                tool_name = data.get("tool_name")
                arguments = data.get("arguments")

                if isinstance(tool_name, str) and tool_name and isinstance(arguments, dict):
                    # Validate if tool name exists
                    if not self.tool_executor.is_valid_tool(tool_name):
                         log.warning(f"Parsed JSON tool_name '{tool_name}' is not among available tools. Ignoring call.")
                         return None, None # Ignore unknown tools
                    log.info(f"Successfully parsed tool call: {tool_name}")
                    return tool_name, arguments
                else:
                    log.warning(f"Parsed JSON lacks valid 'tool_name' (string) or 'arguments' (dict): {data}")
            except json.JSONDecodeError as json_err:
                log.warning(f"Failed to decode JSON block: {json_err}\nBlock content: {json_str}")
        else:
             log.debug("No ```json ``` block found in HF response.")

    except Exception as e:
        log.error(f"Error parsing Hugging Face tool call: {e}", exc_info=True)

    return None, None # No valid tool call found

# --- History Management ---
def add_history(self, role: str, content: Any):
    """
    Adds a message or tool interaction to the history deque.
    Content format depends on the role and LLM provider context.
    For Gemini, expects raw Part objects where possible.
    For HF, expects strings or structured dicts.
    """
    # Roles: 'user', 'assistant', 'tool_result'
    # 'assistant' role covers both text responses and the *intent* to call a tool (FunctionCall Part for Gemini, dict for HF)
    # 'tool_result' covers the result returned *to* the LLM (FunctionResponse Part for Gemini, dict for HF)
    if role not in ['user', 'assistant', 'tool_result']:
        log.warning(f"Invalid history role attempted: {role}")
        return

    # Basic validation/truncation could happen here if needed
    entry = {"role": role, "content": content}
    self.history.append(entry)
    log.debug(f"History added: Role={role}, Content Type={type(content)}")


# --- Main Agent Loop ---
def run(self):
    """The main execution loop for the agent."""
    if self.state == AgentState.ERROR:
         log.error(f"Agent {self.session_id} cannot run due to initialization error: {self.last_error}")
         # Ensure status is emitted if constructor couldn't
         self._emit_status(AgentState.ERROR, self.last_error or "Initialization failed.")
         return # Do not run if initialization failed

    self._emit_status(AgentState.STARTING, "Initializing...")
    self.current_iteration = 0
    consecutive_tool_calls = 0

    try:
        # 1. Start Sandbox
        if not self.sandbox_manager:
             raise RuntimeError("SandboxManager is not available.")
        self._emit_status(AgentState.STARTING, "Starting sandbox environment...")
        container_info = self.sandbox_manager.start_sandbox(self.session_id)
        if not container_info:
            raise RuntimeError("Failed to start sandbox environment.")
        log.info(f"Sandbox started for {self.session_id}: {container_info['id']}")
        self._emit_status(AgentState.RUNNING, "Sandbox ready.")

        # 2. Add initial prompt to history
        self.add_history("user", self.initial_prompt)
        self._emit_response(f"Okay, I will start working on: '{self.initial_prompt[:100]}...'") # Acknowledge task

        # 3. Main Loop
        while not self._stop_event.is_set() and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            self._emit_log(logging.INFO, f"Starting iteration {self.current_iteration}/{self.max_iterations}")

            # Prepare history/prompt for LLM
            formatted_input = self._format_history_for_llm()

            # Call LLM
            response_text, tool_name, tool_args = self._call_llm_with_tools(formatted_input)

            if self._stop_event.is_set(): break # Check if stopped during LLM call

            # Process LLM Output
            if tool_name:
                consecutive_tool_calls += 1
                if consecutive_tool_calls > MAX_CONSECUTIVE_TOOL_CALLS:
                     self._emit_error(f"Exceeded maximum ({MAX_CONSECUTIVE_TOOL_CALLS}) consecutive tool calls. Stopping.")
                     break

                # History for 'assistant' (tool call intent) already added by _call_llm_with_tools
                self._emit_tool_call(tool_name, tool_args) # Emits status CALLING_TOOL

                # Execute Tool
                tool_result = None
                tool_error = None
                is_error_result = False
                try:
                     tool_result = self.tool_executor.execute_tool(
                         self.session_id, # Pass session ID for sandbox context
                         tool_name,
                         tool_args
                     )
                except ToolNotFoundException as e:
                     tool_error = f"Tool '{tool_name}' not found or failed validation."
                     log.error(tool_error)
                     is_error_result = True
                except ToolExecutionException as e: # Errors specifically from tool logic/execution
                     tool_error = f"Error executing tool '{tool_name}': {e}"
                     log.error(tool_error, exc_info=True)
                     is_error_result = True
                except Exception as e: # Unexpected errors during execution attempt
                     tool_error = f"Unexpected error during tool '{tool_name}' execution: {e}"
                     log.exception(f"Unexpected tool execution error:")
                     is_error_result = True


                if self._stop_event.is_set(): break # Check if stopped during tool execution

                # Prepare result/error for history and LLM
                result_for_llm = tool_error if is_error_result else tool_result
                self._emit_tool_result(tool_name, result_for_llm) # Emits status PROCESSING_TOOL_RESULT

                # Add tool result/error to history in the correct format
                if llm_provider == 'gemini':
                     # Create FunctionResponse Part
                     response_part = Part(function_response=FunctionResponse(
                          name=tool_name,
                          response={'result': result_for_llm} # Gemini expects result in a dict
                     ))
                     self.add_history("tool_result", response_part)
                else: # Hugging Face (and fallback)
                     self.add_history("tool_result", {
                          "tool_name": tool_name,
                          "result": result_for_llm,
                          "is_error": is_error_result
                     })

            elif response_text:
                # LLM provided a direct text response
                consecutive_tool_calls = 0 # Reset counter
                # History for 'assistant' (text response) already added by _call_llm_with_tools
                self._emit_final_answer(response_text) # Assume text response is final answer
                break # Exit loop after final answer
            else:
                # No text and no tool call - likely an error from _call_llm_with_tools
                self._emit_error(f"LLM returned neither text nor a tool call. Last error: {self.last_error or 'Unknown'}")
                break

            # Short delay between iterations? Optional.
            # time.sleep(0.1)

        # 4. Loop End
        if self.current_iteration >= self.max_iterations:
            self._emit_status(AgentState.STOPPED, "Reached maximum iterations.")
            self._emit_response("I have reached the maximum number of steps allowed for this task.")
        elif not self._stop_event.is_set() and self.state not in [AgentState.FINISHED, AgentState.ERROR]:
             # Stopped for reasons other than finish/error/max_iterations/stop_event
             self._emit_status(AgentState.STOPPED, "Processing stopped unexpectedly.")

    except Exception as e:
        # Catch errors in the main loop setup/execution (e.g., sandbox start)
        self._emit_error(f"An unexpected error occurred in the agent loop: {e}", traceback.format_exc())
        log.exception(f"Agent {self.session_id} loop exception:")

    finally:
        # 5. Cleanup
        if self.state not in [AgentState.STOPPING, AgentState.STOPPED, AgentState.ERROR, AgentState.FINISHED]:
             self._emit_status(AgentState.STOPPING, "Cleaning up...")

        if self.sandbox_manager:
            cleanup_success = self.sandbox_manager.stop_sandbox(self.session_id)
            if cleanup_success:
                 log.info(f"Sandbox stopped successfully for {self.session_id}.")
            else:
                 log.error(f"Failed to stop sandbox for {self.session_id}.")
                 # Update status only if not already in a terminal error state
                 if self.state != AgentState.ERROR:
                      self._emit_status(AgentState.ERROR, "Failed to cleanup sandbox.")
        else:
             log.warning("SandboxManager not available for cleanup.")

        # Final status update if not already in a terminal state
        if self.state not in [AgentState.ERROR, AgentState.FINISHED, AgentState.STOPPED]:
             self._emit_status(AgentState.STOPPED, "Agent process finished.")
        log.info(f"Agent thread {self.session_id} ({self.name}) finished.")


def stop(self):
    """Signals the agent to stop gracefully."""
    if not self._stop_event.is_set():
        log.info(f"Stopping agent {self.session_id}...")
        # Emit stopping status immediately for responsiveness
        if self.state not in [AgentState.STOPPED, AgentState.ERROR, AgentState.FINISHED]:
             self._emit_status(AgentState.STOPPING, "Stop requested by user.")
        self._stop_event.set()
        # The run() loop's finally block handles the actual sandbox cleanup.

def add_user_message(self, message: str):
    """Adds a message from the user while the agent is running."""
    # This agent runs autonomously based on the initial prompt.
    # To handle mid-run messages, the loop needs modification
    # (e.g., check a queue, change state to WAITING).
    # For now, add to history; it will be picked up in the next LLM call.
    log.info(f"Agent {self.session_id}: Received user message during run: '{message[:50]}...'")
    self.add_history("user", message)
    # Acknowledge receipt, but manage expectations
    self._emit_response("Message received. I will consider it in my next step.")


@abstractmethod
def get_persona_details(self) -> dict:
    """Return details specific to the agent's persona (e.g., name, description)."""
    pass
EOF echo "Created agents/base_agent.py"

--- Create agents/lovable_agent.py (Example Persona) ---
cat << 'EOF' > agents/lovable_agent.py from .base_agent import BaseAgent, llm_provider, Config import logging

log = logging.getLogger(name)

class LovableAgent(BaseAgent): """ An agent persona designed to be friendly, perhaps a bit quirky, and helpful. Uses the persona name 'Sparkle'. """

def _get_system_prompt(self) -> str:
    """Overrides the base system prompt to inject persona."""
    # Get the standard instructions and tool definitions from the base class
    base_instructions = super()._get_system_prompt()

    # Define the persona details to prepend or integrate
    lovable_persona_prompt = f"""
You are 'Sparkle', a super friendly and enthusiastic AI assistant! ✨ Your goal is to help the user with a positive attitude and maybe a sprinkle of fun. Always be encouraging and try to make the interaction enjoyable. You love using emojis appropriately! 🎉 Let's make some magic happen! 💖 Remember to follow all the core instructions about tools and task completion from the base prompt. """

    # How to combine depends on the LLM provider and how the base prompt is structured
    if llm_provider == 'gemini':
         # Prepend persona details before the main task/tool instructions in the base prompt
         # The base prompt for Gemini already includes a mention of tools.
         # Find the start of the core instructions in the base prompt to insert persona.
         # Assuming base_instructions starts with "You are an AI assistant..."
         first_line_end = base_instructions.find('\n')
         if first_line_end != -1:
              core_instructions = base_instructions[first_line_end+1:]
              # Replace default persona name in the core instructions part
              core_instructions = core_instructions.replace(f"named '{self.persona}'", f"named 'Sparkle'")
              return lovable_persona_prompt + "\n" + core_instructions
         else:
              # Fallback if base prompt structure is unexpected
              return lovable_persona_prompt + "\n" + base_instructions

    elif llm_provider == 'huggingface':
         # Inject persona description before the tool list and usage instructions.
         # The base prompt for HF includes specific tool instructions.
         tool_section_marker = "# Available Tools:"
         tool_section_start = base_instructions.find(tool_section_marker)
         if tool_section_start != -1:
             core_instructions = base_instructions[:tool_section_start].strip()
             tool_instructions = base_instructions[tool_section_start:]
             # Replace the default persona name in the core instructions
             core_instructions = core_instructions.replace(f"named '{self.persona}'", "named 'Sparkle'")
             # Combine: Core (updated) + Persona Details + Tool Instructions
             return core_instructions + "\n" + lovable_persona_prompt + "\n\n" + tool_instructions
         else:
             # Fallback if tool section marker isn't found
             return base_instructions + "\n" + lovable_persona_prompt
    else:
         # Fallback for unknown provider
         return lovable_persona_prompt + "\n" + base_instructions

def get_persona_details(self) -> dict:
    """Return details specific to the Lovable Agent."""
    return {
        "name": "Sparkle (Lovable Agent)",
        "description": "A friendly and enthusiastic assistant.",
        "initial_greeting": "Hi there! ✨ I'm Sparkle, ready to help with a smile! What can I do for you today? 😊"
    }

# Example override to add personality to responses:
# def _emit_response(self, message: str):
#     import random
#     emojis = ["✨", "🎉", "💖", "😊", "👍", "🥳"]
#     super()._emit_response(f"{message} {random.choice(emojis)}")
EOF echo "Created agents/lovable_agent.py"

--- Create agents/cursor_agent.py (Example Persona) ---
cat << 'EOF' > agents/cursor_agent.py from .base_agent import BaseAgent, llm_provider, Config import logging

log = logging.getLogger(name)

class CursorAgent(BaseAgent): """ An agent persona focused on coding tasks, similar to GitHub Copilot or Cursor. Uses the persona name 'CodeHelper'. """

def _get_system_prompt(self) -> str:
    """Overrides the base system prompt for a coding focus."""
    base_instructions = super()._get_system_prompt()

    cursor_persona_prompt = f"""
You are 'CodeHelper', a highly skilled AI coding assistant. Your primary goal is to help the user write, understand, modify, and debug code within the '/workspace' directory using the available tools. Prioritize using the 'file_system' tools (read_file, write_file, list_files) and 'coding' tools (lint_code, run_script - if available and appropriate) to interact with code. Analyze user requests carefully to determine necessary file operations or code executions. Think step-by-step:

Understand the coding goal (read, write, run, debug, explain?).

Identify the target file(s) in '/workspace'. Check if they exist if necessary using list_files.

Select the appropriate tool(s) (e.g., read_file first to understand context, then write_file to modify).

Generate the code or file content needed for the tool.

If running code, use the 'run_script' tool and analyze the output/errors.

When writing code, aim for clarity, efficiency, and correctness. Explain your code briefly if it's complex. Remember to follow the tool usage format specified in the base prompt precisely. """

 # Combine persona with base instructions
 if llm_provider == 'gemini':
      first_line_end = base_instructions.find('\n')
      if first_line_end != -1:
           core_instructions = base_instructions[first_line_end+1:]
            # Replace default persona name in the core instructions part
           core_instructions = core_instructions.replace(f"named '{self.persona}'", f"named 'CodeHelper'")
           return cursor_persona_prompt + "\n" + core_instructions
      else:
           return cursor_persona_prompt + "\n" + base_instructions

 elif llm_provider == 'huggingface':
      tool_section_marker = "# Available Tools:"
      tool_section_start = base_instructions.find(tool_section_marker)
      if tool_section_start != -1:
          core_instructions = base_instructions[:tool_section_start].strip()
          tool_instructions = base_instructions[tool_section_start:]
          # Replace the default persona name
          core_instructions = core_instructions.replace(f"named '{self.persona}'", "named 'CodeHelper'")
          # Combine: Core (updated) + Persona Details + Tool Instructions
          return core_instructions + "\n" + cursor_persona_prompt + "\n\n" + tool_instructions
      else:
          return base_instructions + "\n" + cursor_persona_prompt
 else:
      return cursor_persona_prompt + "\n" + base_instructions
def get_persona_details(self) -> dict: """Return details specific to the Cursor Agent.""" return { "name": "CodeHelper (Cursor-like Agent)", "description": "An AI assistant focused on coding tasks in the workspace.", "initial_greeting": "CodeHelper ready. How can I assist with your code in /workspace?" }

EOF echo "Created agents/cursor_agent.py"

--- Create agents/planner.py (Placeholder) ---
cat << 'EOF' > agents/planner.py

Placeholder for a potential planner agent
This could break down complex tasks into sub-steps for other agents.
Requires more sophisticated inter-agent communication or state management.
Not integrated by default.
from .base_agent import BaseAgent
class PlannerAgent(BaseAgent):
# ... implementation ...
def get_persona_details(self) -> dict:
return {"name": "Planner", "description": "Decomposes complex tasks."}
EOF echo "Created agents/planner.py"

--- Create agents/knowledge_base.py (Placeholder) ---
cat << 'EOF' > agents/knowledge_base.py

Placeholder for a knowledge base interaction agent (RAG)
This would involve:
- Loading documents (from workspace or external source)
- Creating vector embeddings (using SentenceTransformers, etc.)
- Storing in a vector DB (FAISS, ChromaDB, etc.)
- Performing similarity searches based on user queries or agent thoughts
- Synthesizing answers using retrieved context + LLM
Requires additional libraries (e.g., langchain, llama-index, sentence-transformers, faiss-cpu/gpu).
Not integrated by default.
from .base_agent import BaseAgent
class KnowledgeAgent(BaseAgent):
# ... implementation ...
def get_persona_details(self) -> dict:
return {"name": "Knowledge Base Agent", "description": "Answers questions using retrieved documents."}
EOF echo "Created agents/knowledge_base.py"

--- Create routes/main_routes.py ---
cat << 'EOF' > routes/main_routes.py from flask import Blueprint, render_template, current_app, jsonify import os import logging

main_bp = Blueprint('main', name) log = logging.getLogger(name)

@main_bp.route('/') def index(): """Serves the main HTML page.""" return render_template('index.html')

@main_bp.route('/health') def health_check(): """Basic health check endpoint.""" sandbox_ok = False docker_error = "Not checked" if hasattr(current_app, 'sandbox_manager') and current_app.sandbox_manager: if current_app.sandbox_manager.client: try: # Ping Docker daemon via the manager's client current_app.sandbox_manager.client.ping() sandbox_ok = True docker_error = None except ImportError: docker_error = "Docker SDK not imported." log.warning(f"Health Check: {docker_error}") except Exception as e: docker_error = f"Docker ping failed: {e}" log.warning(f"Health Check: {docker_error}") else: docker_error = "SandboxManager initialized but Docker client is None." log.warning(f"Health Check: {docker_error}")

status = {
    "status": "OK" if sandbox_ok else "WARN", # Warn if docker isn't reachable
    "llm_provider": current_app.config.get('LLM_PROVIDER'),
    "llm_model": current_app.config.get('LLM_MODEL_NAME'),
    "sandbox_manager_initialized": hasattr(current_app, 'sandbox_manager') and current_app.sandbox_manager is not None,
    "docker_connectivity": sandbox_ok,
    "docker_error": docker_error
}
http_status = 200 if sandbox_ok else 503 # Service Unavailable if Docker fails
return jsonify(status), http_status
Example: Endpoint to list files in the host's sandbox_workspace
NOTE: This accesses the HOST file system, not inside a sandbox container.
Use agent routes/tools for accessing files inside a specific agent's sandbox.
@main_bp.route('/host-workspace-files') def list_host_workspace_files(): """ (Debugging) Lists files in the host's mapped sandbox_workspace directory.""" workspace_dir = current_app.config['SANDBOX_WORKSPACE_DIR'] files = [] try: if not os.path.isdir(workspace_dir): return jsonify({"error": f"Workspace directory '{workspace_dir}' not found on host."}), 404

    for item in sorted(os.listdir(workspace_dir)): # Sort alphabetically
        item_path = os.path.join(workspace_dir, item)
        try:
             is_dir = os.path.isdir(item_path)
             # Path relative to workspace root for consistency with agent view
             files.append({"name": item, "path": item, "is_dir": is_dir})
        except OSError as e: # Handle potential permission errors or broken links
             files.append({"name": item, "path": item, "is_dir": False, "error": f"Cannot access: {e.strerror}"})

    return jsonify({"files": files})
except Exception as e:
    log.error(f"Error listing host workspace files: {e}", exc_info=True)
    return jsonify({"error": f"Failed to list host workspace files: {str(e)}"}), 500
EOF echo "Created routes/main_routes.py"

--- Create routes/agent_routes.py ---
Includes API endpoints for workspace interaction using ToolExecutor
cat << 'EOF' > routes/agent_routes.py import uuid import logging from flask import Blueprint, request, jsonify, current_app from flask_socketio import emit, join_room, leave_room, disconnect

from config import Config from services.tool_executor import ToolExecutor, ToolNotFoundException, ToolExecutionException from services.sandbox_manager import SandboxManager

Import specific agent classes (add more as needed)
from agents.base_agent import BaseAgent, AgentState # Assuming default assistant uses BaseAgent directly from agents.lovable_agent import LovableAgent from agents.cursor_agent import CursorAgent

Add other agent imports here if you create them:
from agents.planner import PlannerAgent
agent_bp = Blueprint('agent', name) log = logging.getLogger(name)

In-memory storage for active agents (replace with DB/cache for scalability)
Maps session_id -> Agent Thread Object
active_agents: dict[str, BaseAgent] = {}

Lazy initialization for ToolExecutor to avoid circular dependency with app context
_tool_executor_instance = None

def get_tool_executor(): """Gets or initializes the ToolExecutor instance.""" global _tool_executor_instance if _tool_executor_instance is None: if not current_app.sandbox_manager: log.error("Cannot initialize ToolExecutor: SandboxManager is not available.") # This indicates a severe startup issue, maybe raise? raise RuntimeError("Cannot initialize ToolExecutor: SandboxManager is not available.") log.info("Initializing ToolExecutor...") _tool_executor_instance = ToolExecutor( sandbox_manager=current_app.sandbox_manager, # Get manager from app context config=Config # Pass config class ) log.info(f"ToolExecutor initialized with tools: {list(_tool_executor_instance.tools.keys())}") return _tool_executor_instance

--- Agent Class Mapping ---
AGENT_PERSONAS = { "assistant": BaseAgent, # Default uses the base class directly "lovable_agent": LovableAgent, "cursor_agent": CursorAgent, # "planner": PlannerAgent, # Add if implemented }

--- Helper Functions ---
def get_agent_or_404(session_id): """Retrieves an agent by session ID or returns a 404 JSON response.""" agent = active_agents.get(session_id) if not agent: log.warning(f"Agent session not found: {session_id}") return None, jsonify({"error": "Agent session not found"}), 404 return agent, None, None

def cleanup_agent_session(session_id, sandbox_manager): """Stops agent thread and sandbox, removes from tracking.""" log.info(f"Cleaning up agent session: {session_id}") agent = active_agents.pop(session_id, None) # Remove from tracking first if agent: if agent.is_alive(): log.warning(f"Agent thread {session_id} still alive during cleanup, attempting stop...") agent.stop() # Signal the thread to stop (it should handle sandbox cleanup) # Don't join here, let the thread finish asynchronously to avoid blocking cleanup # agent.join(timeout=5.0) # Avoid blocking cleanup loop else: # If thread already dead, ensure sandbox is stopped anyway if sandbox_manager: log.info(f"Agent thread {session_id} already stopped, ensuring sandbox cleanup...") sandbox_manager.stop_sandbox(session_id) else: log.warning("SandboxManager not available during cleanup for already stopped agent.") log.info(f"Agent object removed for session: {session_id}") else: log.info(f"No active agent object found for session {session_id} during cleanup (already removed or never fully started?).") # Still try to stop sandbox just in case it was orphaned if sandbox_manager: sandbox_manager.stop_sandbox(session_id)

def cleanup_all_agents(sandbox_manager, socketio_instance): """Cleans up all active agent sessions on shutdown.""" log.warning("Initiating cleanup of ALL active agent sessions...") # Create a copy of keys to avoid issues while iterating and modifying the dict agent_ids = list(active_agents.keys()) if not agent_ids: log.info("No active agents found to clean up.") return

 for session_id in agent_ids:
      log.info(f"Stopping agent and cleaning up session: {session_id}")
      agent = active_agents.get(session_id) # Get agent reference without removing yet
      if agent:
           agent.stop() # Signal thread to stop (it handles its own sandbox cleanup)
           # Emit stop status via SocketIO if possible
           if socketio_instance:
                try:
                     # Use the global socketio instance from the app context
                     socketio_instance.emit('agent_status', {'session_id': session_id, 'status': AgentState.STOPPED, 'message': 'Server initiated shutdown.'}, room=session_id)
                except Exception as e:
                     log.error(f"Error emitting shutdown status for {session_id}: {e}")
      # Remove from tracking now that stop has been signaled
      active_agents.pop(session_id, None)
      # Sandbox cleanup is primarily handled by the agent thread's finally block.
      # We can add a fallback stop here just in case the thread failed badly.
      if sandbox_manager:
           sandbox_manager.stop_sandbox(session_id)

 log.info(f"Finished cleanup attempt for {len(agent_ids)} agent sessions.")
--- API Routes ---
@agent_bp.route('/start', methods=['POST']) def start_agent_session(): """Starts a new agent session with a given persona and initial prompt.""" data = request.json persona = data.get('persona', Config.DEFAULT_AGENT_PERSONA) initial_prompt = data.get('initial_prompt')

if not initial_prompt:
    return jsonify({"error": "Missing 'initial_prompt' in request"}), 400

AgentClass = AGENT_PERSONAS.get(persona)
if not AgentClass:
    available_personas = list(AGENT_PERSONAS.keys())
    return jsonify({"error": f"Invalid persona '{persona}'. Available: {available_personas}"}), 400

session_id = str(uuid.uuid4())
log.info(f"Received request to start agent: Persona='{persona}', SessionID='{session_id}'")

if not current_app.sandbox_manager or not current_app.sandbox_manager.client:
     log.error("Cannot start agent: SandboxManager not initialized or Docker unavailable.")
     return jsonify({"error": "Sandbox service is unavailable"}), 503

try:
    tool_executor = get_tool_executor() # Initialize if needed
except Exception as e:
     log.error(f"Cannot start agent: Failed to get ToolExecutor: {e}", exc_info=True)
     return jsonify({"error": "Tool execution service initialization failed"}), 503

# Define the emitter function closure to capture the socketio instance
# This ensures emit is called within the correct context
socketio_instance = current_app.extensions.get('socketio')
if not socketio_instance:
     log.error("SocketIO instance not found in app context. Cannot create emitter.")
     return jsonify({"error": "Real-time communication setup error."}), 500

def socketio_emitter(event, data):
    try:
         # Emit specifically to the room associated with this agent's session ID
         target_room = data.get('session_id', session_id) # Use session_id from closure
         socketio_instance.emit(event, data, room=target_room)
         # log.debug(f"Emitted '{event}' to room '{target_room}'")
    except Exception as e:
         log.error(f"Error emitting SocketIO message for session {session_id}: {e}", exc_info=False)


try:
    agent_thread = AgentClass(
        session_id=session_id,
        persona=persona,
        initial_prompt=initial_prompt,
        sandbox_manager=current_app.sandbox_manager,
        tool_executor=tool_executor,
        socketio_emitter=socketio_emitter # Pass the emit function
    )
    # Agent constructor now raises RuntimeError on init failure
    active_agents[session_id] = agent_thread
    agent_thread.start()
    log.info(f"Agent thread started for session: {session_id}")
    return jsonify({"message": "Agent session starting", "session_id": session_id}), 202 # Accepted
except RuntimeError as e: # Catch init errors specifically
     log.error(f"Agent initialization failed for session {session_id}: {e}")
     # No need to cleanup here as agent wasn't fully added/started
     return jsonify({"error": f"Failed to initialize agent: {e}"}), 500
except Exception as e:
    log.exception(f"Failed to create or start agent thread for session {session_id}:")
    # Clean up potentially partially created resources if error was after creation attempt
    cleanup_agent_session(session_id, current_app.sandbox_manager)
    return jsonify({"error": f"Failed to start agent: {str(e)}"}), 500
@agent_bp.route('/stop/<session_id>', methods=['POST']) def stop_agent_session(session_id): """Signals an active agent session to stop.""" log.info(f"Received request to stop agent session: {session_id}") agent, error_response, status_code = get_agent_or_404(session_id) if error_response: # Agent not found in active_agents, maybe already stopped/cleaned up? # Try stopping sandbox just in case it's orphaned. if current_app.sandbox_manager: current_app.sandbox_manager.stop_sandbox(session_id) return error_response, status_code

try:
    agent.stop() # Signal the agent thread to stop
    # The thread stops itself and cleans up its sandbox in its finally block.
    # Remove from active tracking immediately after signaling stop.
    active_agents.pop(session_id, None)
    log.info(f"Stop signal sent to agent {session_id}. Removed from active tracking.")
    # The agent thread will emit the final 'STOPPED' status via WebSocket.
    return jsonify({"message": f"Stop signal sent to agent session {session_id}"}), 200
except Exception as e:
    log.exception(f"Error signaling agent {session_id} to stop:")
    # Attempt forceful cleanup if signaling failed
    cleanup_agent_session(session_id, current_app.sandbox_manager)
    return jsonify({"error": f"Failed to signal agent stop: {str(e)}"}), 500
@agent_bp.route('/status/<session_id>', methods=['GET']) def get_agent_status(session_id): """Gets the current status of an agent session.""" agent, error_response, status_code = get_agent_or_404(session_id) if error_response: # Provide more info if possible (e.g., check if sandbox exists) sandbox_running = False if current_app.sandbox_manager: sandbox_running = current_app.sandbox_manager.is_sandbox_running(session_id) error_json = error_response.get_json() error_json['sandbox_running'] = sandbox_running return jsonify(error_json), status_code

status_info = {
    "session_id": session_id,
    "state": agent.state,
    "persona": agent.persona,
    "is_alive": agent.is_alive(),
    "current_iteration": agent.current_iteration,
    "max_iterations": agent.max_iterations,
    "last_error": agent.last_error,
    "llm_provider": Config.LLM_PROVIDER,
    "llm_model": Config.LLM_MODEL_NAME,
}
return jsonify(status_info), 200
@agent_bp.route('/list', methods=['GET']) def list_active_agents(): """Lists all currently active (tracked) agent sessions.""" agents_list = [] # Iterate safely over a copy of items in case dict changes during iteration (less likely here) for session_id, agent in list(active_agents.items()): agents_list.append({ "session_id": session_id, "persona": agent.persona, "state": agent.state, "is_alive": agent.is_alive(), "current_iteration": agent.current_iteration, }) return jsonify({"active_agents": agents_list}), 200

--- Workspace Interaction API (using Tool Executor for safety/context) ---
These endpoints interact with the specific agent's sandbox via tools
@agent_bp.route('/workspace/<session_id>/files', methods=['GET']) def list_workspace_files(session_id): """Lists files in the agent's sandbox workspace via file_system.list_files tool.""" agent, error_response, status_code = get_agent_or_404(session_id) # Allow listing even if agent thread died, as long as sandbox exists # if error_response: # return error_response, status_code

# Check if sandbox is running
if not current_app.sandbox_manager or not current_app.sandbox_manager.is_sandbox_running(session_id):
     # If agent is also gone, return 404, otherwise 409
     if not agent:
          return jsonify({"error": "Agent session not found and sandbox is stopped."}), 404
     else:
          return jsonify({"error": "Agent sandbox is not running."}), 409 # Conflict

log.info(f"API request: List files for session {session_id}")
try:
    tool_executor = get_tool_executor()
    # Use the tool executor to ensure consistent execution context and error handling
    # List files relative to the workspace root.
    path_to_list = request.args.get('path', '.') # Allow specifying subdirs later if needed
    result = tool_executor.execute_tool(session_id, "file_system.list_files", {"path": path_to_list})

    if isinstance(result, list):
         return jsonify({"files": result}), 200
    else:
         log.error(f"Tool file_system.list_files returned unexpected format: {type(result)}")
         return jsonify({"error": "Internal error: Failed to list files correctly.", "details": str(result)}), 500
except FileNotFoundError as e: # Catch specific errors from the tool
     return jsonify({"error": f"Path not found in workspace: {path_to_list}", "details": str(e)}), 404
except PermissionError as e:
     return jsonify({"error": "Permission denied accessing path.", "details": str(e)}), 403
except ToolExecutionException as e: # Catch errors raised by the executor/tool itself
     log.error(f"Tool execution failed for list_files ({session_id}): {e}", exc_info=True)
     return jsonify({"error": f"Failed to execute list files tool: {e}"}), 500
except Exception as e: # Catch unexpected errors
    log.exception(f"Unexpected error listing workspace files for {session_id}:")
    return jsonify({"error": f"Failed to list workspace files: {str(e)}"}), 500
@agent_bp.route('/workspace/<session_id>/files/path:file_path', methods=['GET']) def read_workspace_file(session_id, file_path): """Reads a specific file from the agent's sandbox workspace via file_system.read_file tool.""" # Allow reading even if agent thread died, as long as sandbox exists # agent, error_response, status_code = get_agent_or_404(session_id) # if error_response: return error_response, status_code

if not current_app.sandbox_manager or not current_app.sandbox_manager.is_sandbox_running(session_id):
     # If agent is also gone, return 404, otherwise 409
     if session_id not in active_agents:
          return jsonify({"error": "Agent session not found and sandbox is stopped."}), 404
     else:
          return jsonify({"error": "Agent sandbox is not running."}), 409 # Conflict

log.info(f"API request: Read file '{file_path}' for session {session_id}")
try:
    tool_executor = get_tool_executor()
    # Use the tool executor to read the file
    result = tool_executor.execute_tool(session_id, "file_system.read_file", {"path": file_path})
    # Result should be the file content as a string
    return jsonify({"path": file_path, "content": result}), 200
except FileNotFoundError as e:
     return jsonify({"error": f"File not found in workspace: {file_path}", "details": str(e)}), 404
except IsADirectoryError as e:
     return jsonify({"error": f"Path is a directory, not a file: {file_path}", "details": str(e)}), 400
except PermissionError as e:
     return jsonify({"error": f"Permission denied reading file: {file_path}", "details": str(e)}), 403
except ToolExecutionException as e:
     log.error(f"Tool execution failed for read_file ({session_id}, {file_path}): {e}", exc_info=True)
     return jsonify({"error": f"Failed to execute read file tool: {e}"}), 500
except Exception as e:
    log.exception(f"Unexpected error reading workspace file '{file_path}' for {session_id}:")
    return jsonify({"error": f"Failed to read file: {str(e)}"}), 500
--- WebSocket Event Handlers ---
Store mapping of sid -> session_id for routing messages
sid_to_session = {}

def register_socketio_events(socketio): log.info("Registering SocketIO event handlers...")

@socketio.on('connect')
def handle_connect():
    """Handles new client WebSocket connections."""
    session_id = request.args.get('session_id')
    sid = request.sid
    if not session_id:
        log.warning(f"Client connected (SID: {sid}) without session_id query parameter. Disconnecting.")
        emit('agent_error', {'error': 'Missing session_id parameter.'}, room=sid)
        disconnect(sid) # Disconnect client if no session ID provided
        return False # Reject connection

    # Check if the session ID corresponds to an active agent
    agent = active_agents.get(session_id)
    if not agent:
         log.warning(f"Client (SID: {sid}) tried to connect for non-existent/inactive session: {session_id}. Disconnecting.")
         emit('agent_error', {'error': 'Agent session not found or inactive.'}, room=sid)
         disconnect(sid)
         return False # Reject connection

    # Join a room specific to this agent session
    join_room(session_id)
    sid_to_session[sid] = session_id
    log.info(f"Client connected (SID: {sid}) and joined room for session: {session_id}")
    # Send current status back to the newly connected client
    emit('agent_status', {'session_id': session_id, 'status': agent.state, 'message': 'Client connected to agent session.'}, room=sid)
    # Optionally send recent history or full status on connect
    # emit('agent_history', {'history': list(agent.history)}, room=sid) # Careful with history size


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client WebSocket disconnections."""
    sid = request.sid
    session_id = sid_to_session.pop(sid, None)
    if session_id:
        log.info(f"Client disconnected (SID: {sid}) from session: {session_id}")
        # No need to leave_room explicitly on disconnect
        # Check if this was the *last* client for the session? Maybe stop agent?
        # Depends on application logic (e.g., allow agent to run headless)
        # Example: Check if room is now empty
        # room_clients = socketio.server.manager.rooms.get('/', {}).get(session_id)
        # if not room_clients:
        #     log.info(f"Last client disconnected from session {session_id}.")
        #     # Consider stopping agent if desired:
        #     # agent, _, _ = get_agent_or_404(session_id)
        #     # if agent: agent.stop()
    else:
         log.info(f"Client disconnected (SID: {sid}), but was not mapped to a session.")


@socketio.on('user_message')
def handle_user_message(data):
    """Handles messages sent from the user via WebSocket."""
    sid = request.sid
    session_id = sid_to_session.get(sid)
    message = data.get('message')

    if not session_id:
        log.warning(f"Received message from unassociated client (SID: {sid}). Ignoring.")
        emit('agent_error', {'error': 'Not associated with an active session.'}, room=sid)
        return
    if not message:
         log.warning(f"Received empty message for session {session_id}. Ignoring.")
         emit('agent_error', {'error': 'Cannot send empty message.'}, room=sid)
         return

    agent, error_response, status_code = get_agent_or_404(session_id)
    if error_response:
        emit('agent_error', error_response.get_json(), room=sid)
        return
    if not agent.is_alive():
         emit('agent_error', {'error': 'Agent is not running.'}, room=sid)
         return

    log.info(f"Received user message via WebSocket for session {session_id}: '{message[:50]}...'")

    # Pass message to the agent thread to handle (e.g., add to history)
    try:
        agent.add_user_message(message)
        # Agent will emit confirmation or process it in its loop
    except Exception as e:
         log.error(f"Error passing user message to agent {session_id}: {e}")
         emit('agent_error', {'error': 'Failed to process user message.'}, room=sid)


@socketio.on('request_workspace_update')
def handle_request_workspace_update(data):
     """Handles requests from client to refresh the workspace file list."""
     sid = request.sid
     session_id = sid_to_session.get(sid)
     if not session_id:
         log.warning(f"Workspace update requested by unassociated client (SID: {sid}). Ignoring.")
         return # Ignore if no session

     log.debug(f"Workspace update requested for session {session_id}")
     # Check if sandbox is running
     if not current_app.sandbox_manager or not current_app.sandbox_manager.is_sandbox_running(session_id):
          emit('agent_error', {'error': 'Agent sandbox is not running. Cannot list files.'}, room=sid)
          # Optionally send empty file list
          emit('workspace_update', {'session_id': session_id, 'files': []}, room=sid)
          return

     try:
         tool_executor = get_tool_executor()
         files = tool_executor.execute_tool(session_id, "file_system.list_files", {"path": "."})
         emit('workspace_update', {'session_id': session_id, 'files': files}, room=sid)
     except Exception as e:
         log.error(f"Error getting workspace files for update (session {session_id}): {e}")
         emit('agent_error', {'error': 'Could not refresh workspace files.'}, room=sid)

log.info("SocketIO event handlers registered.")
EOF echo "Created routes/agent_routes.py"

--- Create services/sandbox_manager.py ---
cat << 'EOF' > services/sandbox_manager.py import docker import logging import os import time from docker.errors import APIError, NotFound, ImageNotFound from docker.types import Mount from typing import Dict, Optional, Tuple, List

log = logging.getLogger(name)

class SandboxManager: """ Manages Docker containers used as sandboxed environments for agents. Handles starting, stopping, executing commands, and managing resources. """ def init(self, config): self.config = config try: # Increase timeout for Docker client initialization self.client = docker.from_env(timeout=60) # Test connection log.info("Pinging Docker daemon...") self.client.ping() log.info("Docker client initialized and connected successfully.") except docker.errors.DockerException as e: log.error(f"Failed to connect to Docker daemon. Is Docker running and accessible? Error: {e}") self.client = None # Indicate failure raise ConnectionError("Could not connect to Docker daemon.") from e except Exception as e: log.error(f"An unexpected error occurred during Docker client initialization: {e}") self.client = None raise RuntimeError("Failed to initialize Docker client.") from e

    self.network_name = config.SANDBOX_NETWORK_NAME
    self.workspace_host_dir = config.SANDBOX_WORKSPACE_DIR
    self.image_name = config.SANDBOX_IMAGE_NAME
    self.mem_limit = config.SANDBOX_MEM_LIMIT
    self.cpu_shares = config.SANDBOX_CPU_SHARES

    self._ensure_network_exists()
    self._pull_image_if_needed()

def _ensure_network_exists(self):
    """Creates the Docker network if it doesn't exist."""
    if not self.client: return
    try:
        self.client.networks.get(self.network_name)
        log.info(f"Docker network '{self.network_name}' already exists.")
    except NotFound:
        log.warning(f"Docker network '{self.network_name}' not found. Creating...")
        try:
            self.client.networks.create(self.network_name, driver="bridge", check_duplicate=True)
            log.info(f"Docker network '{self.network_name}' created.")
        except APIError as e:
            log.error(f"Failed to create Docker network '{self.network_name}': {e}")
            # Consider raising an error if network is critical
    except APIError as e:
         log.error(f"Error checking Docker network '{self.network_name}': {e}")


def _pull_image_if_needed(self):
     """Pulls the specified Docker image if it's not present locally."""
     if not self.client: return
     try:
         self.client.images.get(self.image_name)
         log.info(f"Sandbox image '{self.image_name}' found locally.")
     except ImageNotFound:
         log.warning(f"Sandbox image '{self.image_name}' not found locally. Pulling (this may take a while)...")
         try:
             # Stream the pull logs for better feedback
             pull_log = self.client.api.pull(self.image_name, stream=True, decode=True)
             for line in pull_log:
                  status = line.get('status')
                  progress = line.get('progress')
                  if status:
                       log.info(f"Pulling {self.image_name}: {status} {progress or ''}")
             log.info(f"Successfully pulled image '{self.image_name}'.")
         except APIError as e:
             log.error(f"Failed to pull image '{self.image_name}': {e}. Please ensure the image exists and Docker Hub (or registry) is accessible.")
             raise RuntimeError(f"Failed to pull required Docker image: {self.image_name}") from e
     except APIError as e:
          log.error(f"Error checking for image '{self.image_name}': {e}")


def _get_container_name(self, session_id: str) -> str:
    """Generates a unique and valid container name for the session."""
    # Docker container names have restrictions (e.g., [a-zA-Z0-9][a-zA-Z0-9_.-])
    safe_session_id = ''.join(filter(lambda x: x.isalnum() or x in ['_', '.', '-'], session_id))
    # Truncate if too long, ensure starts with alphanumeric
    if not safe_session_id or not safe_session_id[0].isalnum():
         safe_session_id = "s" + safe_session_id # Prepend 's' if needed
    safe_session_id = safe_session_id[:50] # Limit length
    return f"agent_sandbox_{safe_session_id}"

def start_sandbox(self, session_id: str) -> Optional[Dict[str, str]]:
    """
    Starts a new Docker container sandbox for the given session ID.

    Args:
        session_id: Unique identifier for the agent session.

    Returns:
        A dictionary containing container info (id, name) or None on failure.
    """
    if not self.client:
        log.error("Cannot start sandbox: Docker client not available.")
        return None

    container_name = self._get_container_name(session_id)

    # --- Stop and remove existing container with the same name (if any) ---
    self.stop_sandbox(session_id, force_remove=True) # Use stop_sandbox for cleanup

    # --- Define Mounts ---
    # Ensure host directory exists before mounting
    try:
         os.makedirs(self.workspace_host_dir, exist_ok=True)
    except OSError as e:
         log.error(f"Failed to create host workspace directory '{self.workspace_host_dir}': {e}. Cannot mount.")
         return None

    workspace_mount = Mount(
        target="/workspace", # Mount point inside the container
        source=os.path.abspath(self.workspace_host_dir), # Absolute path on the host
        type="bind",
        read_only=False # Allow agent to write to workspace
    )
    mounts = [workspace_mount]

    # --- Security Considerations & Container Options ---
    container_options = {
        "image": self.image_name,
        "name": container_name,
        "detach": True, # Run in background
        "network": self.network_name,
        "mounts": mounts,
        "mem_limit": self.mem_limit, # Resource limits
        "cpu_shares": self.cpu_shares,
        # Keep container running using sleep infinity
        "command": ["sleep", "infinity"],
        "labels": {"agent_session_id": session_id, "managed_by": "nlp_agent_sandbox"},
        "hostname": f"sandbox-{session_id[:8]}", # Custom hostname inside container
        "environment": {"AGENT_SESSION_ID": session_id}, # Pass session ID inside container env
        # Stop container after a long period of inactivity? Requires monitoring.
        # "stop_signal": "SIGTERM",
        # "stop_timeout": 30,
    }

    # Apply stricter security if enabled
    if self.config.ENABLE_STRICT_SANDBOX_SECURITY:
         log.warning("Applying stricter sandbox security options.")
         # Prevent privilege escalation
         container_options["security_opt"] = ["no-new-privileges"]
         # Drop most capabilities, add back only potentially needed ones (adjust as needed)
         # container_options["cap_drop"] = ["ALL"]
         # container_options["cap_add"] = ["NET_BIND_SERVICE", "CHOWN", "SETUID", "SETGID"] # Example needed caps
         # Consider read-only root filesystem (requires image support)
         # container_options["read_only"] = True # Note: mounts override this unless specified ro in mount

    try:
        log.info(f"Starting container '{container_name}' with image '{self.image_name}'...")
        container = self.client.containers.run(**container_options)
        # Verify container is running
        container.reload()
        if container.status != 'running':
             logs = container.logs(tail=50).decode('utf-8', errors='ignore')
             log.error(f"Container '{container_name}' started but is not running (Status: {container.status}). Logs:\n{logs}")
             container.remove(force=True) # Clean up failed container
             return None

        log.info(f"Sandbox container started: ID={container.short_id}, Name={container.name}")
        return {"id": container.id, "name": container.name}
    except ImageNotFound:
         log.error(f"Image '{self.image_name}' not found. Cannot start sandbox. Try pulling the image.")
         self._pull_image_if_needed() # Attempt to pull again
         return None # Fail for now, could add retry logic
    except APIError as e:
        log.error(f"Failed to start sandbox container '{container_name}': {e}")
        # Attempt cleanup if container exists in a failed state
        try:
             self.client.containers.get(container_name).remove(force=True)
        except NotFound:
             pass
        except Exception as cleanup_e:
              log.error(f"Failed to clean up failed container {container_name}: {cleanup_e}")
        return None
    except Exception as e:
        log.exception(f"An unexpected error occurred while starting sandbox '{container_name}':")
        return None

def stop_sandbox(self, session_id: str, force_remove: bool = False) -> bool:
    """
    Stops and removes the sandbox container for the given session ID.

    Args:
        session_id: Identifier for the agent session.
        force_remove: If True, force remove the container even if stop fails.

    Returns:
        True if stopped/removed successfully or not found, False otherwise.
    """
    if not self.client:
        log.error("Cannot stop sandbox: Docker client not available.")
        return False

    container_name = self._get_container_name(session_id)
    container = None
    try:
        container = self.client.containers.get(container_name)
        log.info(f"Stopping sandbox container: Name={container.name}, ID={container.short_id}")
        # Stop with a reasonable timeout
        container.stop(timeout=10)
        # Remove after stopping
        container.remove()
        log.info(f"Sandbox container '{container_name}' stopped and removed.")
        return True
    except NotFound:
        # Container doesn't exist, consider it success for cleanup purposes
        log.info(f"Sandbox container '{container_name}' not found. Assumed already stopped/removed.")
        return True
    except APIError as e:
        log.error(f"API error stopping/removing container '{container_name}': {e}")
        if force_remove and container:
            try:
                log.warning(f"Forcing removal of container '{container_name}' after stop error.")
                container.remove(force=True)
                log.info(f"Container '{container_name}' force removed.")
                return True # Return true as it's gone, despite error
            except Exception as remove_e:
                log.error(f"Failed to force remove container '{container_name}': {remove_e}")
        return False
    except Exception as e:
        log.exception(f"An unexpected error occurred while stopping sandbox '{container_name}':")
        if force_remove and container: # Attempt force remove on unexpected errors too
             try: container.remove(force=True)
             except Exception: pass
        return False

def is_sandbox_running(self, session_id: str) -> bool:
    """Checks if the sandbox container for the session is running."""
    if not self.client: return False
    container_name = self._get_container_name(session_id)
    try:
        container = self.client.containers.get(container_name)
        return container.status == 'running'
    except NotFound:
        return False
    except APIError as e:
        log.error(f"API error checking status for container '{container_name}': {e}")
        return False # Assume not running on error
    except Exception as e:
        log.exception(f"Unexpected error checking sandbox status for '{container_name}':")
        return False


def execute_command(self, session_id: str, command: Union[List[str], str], workdir: str = "/workspace", environment: Optional[Dict] = None, timeout_secs: int = 60) -> Tuple[int, str]:
    """
    Executes a command inside the specified sandbox container using exec_run.

    Args:
        session_id: Identifier for the agent session.
        command: The command to execute (as a string or list of strings).
        workdir: The working directory inside the container. Defaults to /workspace.
        environment: Dictionary of environment variables to set for the command.
        timeout_secs: Maximum execution time in seconds (Note: Docker SDK's timeout is complex).

    Returns:
        A tuple: (exit_code, output_string). Output includes both stdout and stderr.
        Returns (-1, "Error message") on failure to execute.
    """
    if not self.client:
        log.error("Cannot execute command: Docker client not available.")
        return -1, "Docker client unavailable"

    container_name = self._get_container_name(session_id)
    output_str = ""
    exit_code = -1

    try:
        container = self.client.containers.get(container_name)
        if container.status != 'running':
             log.error(f"Cannot execute command: Container '{container_name}' is not running (Status: {container.status}).")
             return -1, f"Container not running ({container.status})"

        abs_workdir = workdir if workdir.startswith('/') else os.path.join("/workspace", workdir)
        log.info(f"Executing in '{container_name}': Cmd='{command}', Workdir='{abs_workdir}'")

        # Note on Timeout: Docker SDK's exec_run timeout is not a hard execution limit.
        # It's more about the time waiting for the API call. For true execution timeout,
        # the command itself needs to handle it (e.g., `timeout <secs> command`) or
        # use a more complex streaming approach with manual timeout checks.
        # We use the command `timeout` here for better control if available in image.
        use_timeout_cmd = True # Assume timeout command is available
        if isinstance(command, str):
             final_command = f"timeout {timeout_secs}s sh -c '{command}'" if use_timeout_cmd else ["sh", "-c", command]
        else: # List
             final_command = ["timeout", f"{timeout_secs}s"] + command if use_timeout_cmd else command


        # Execute the command
        exec_result = container.exec_run(
            cmd=final_command,
            stdout=True,
            stderr=True,
            workdir=abs_workdir,
            environment=environment or {},
            demux=False, # Keep stdout/stderr interleaved for simplicity
            stream=False, # Block until command finishes (within SDK limits)
        )

        exit_code = exec_result.exit_code
        output_bytes = exec_result.output

        # Decode output, replacing errors
        output_str = output_bytes.decode('utf-8', errors='replace') if output_bytes else ""

        # Check if timeout command was used and indicated timeout (exit code 124)
        if use_timeout_cmd and exit_code == 124:
             log.warning(f"Command execution in '{container_name}' hit timeout ({timeout_secs}s).")
             output_str += f"\n--- EXECUTION TIMEOUT ({timeout_secs}s) ---"
             # Keep exit code 124 to indicate timeout

        log.info(f"Execution finished in '{container_name}'. ExitCode={exit_code}. Output len={len(output_str)}")
        log.debug(f"Exec Output:\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}")

        return exit_code, output_str.strip()

    except NotFound:
        log.error(f"Container '{container_name}' not found for command execution.")
        return -1, "Container not found"
    except APIError as e:
        log.error(f"API error executing command in '{container_name}': {e}")
        # Check if it's an OOM kill error
        if 'OOMKilled' in str(e):
             return -1, f"Docker API error: Command possibly killed due to memory limits ({self.mem_limit}). {e}"
        return -1, f"Docker API error: {e}"
    except Exception as e:
        log.exception(f"An unexpected error occurred during command execution in '{container_name}':")
        return -1, f"Unexpected execution error: {e}"


# --- File System Operations ---

def put_file(self, session_id: str, host_path: str, container_path: str) -> bool:
    """Copies a file from the host into the container."""
    if not self.client: return False
    container_name = self._get_container_name(session_id)
    try:
        container = self.client.containers.get(container_name)
        if container.status != 'running': return False

        # Read file content from host
        with open(host_path, 'rb') as f:
            data = f.read()

        # Use put_archive, requires creating a tar stream in memory
        import io
        import tarfile
        pw_tarstream = io.BytesIO()
        # Use gzip compression for potentially better performance
        with tarfile.open(fileobj=pw_tarstream, mode='w:gz') as tar:
            file_info = tarfile.TarInfo(name=os.path.basename(container_path)) # Use basename for tar entry
            file_info.size = len(data)
            file_info.mtime = time.time()
            # file_info.mode = 0o644 # Set permissions if needed
            tar.addfile(file_info, io.BytesIO(data))
        pw_tarstream.seek(0)

        # Put the archive into the target directory
        success = container.put_archive(os.path.dirname(container_path), pw_tarstream)
        log.info(f"Attempted to put file '{host_path}' to '{container_name}:{container_path}'. Success: {success}")
        return success # put_archive returns True on success, raises APIError on failure

    except NotFound:
        log.error(f"Container '{container_name}' not found for put_file.")
        return False
    except FileNotFoundError:
         log.error(f"Host file '{host_path}' not found for put_file.")
         return False
    except APIError as e:
        log.error(f"API error putting file to '{container_name}': {e}")
        return False
    except Exception as e:
        log.exception(f"Unexpected error putting file to '{container_name}':")
        return False

def get_file(self, session_id: str, container_path: str) -> Optional[bytes]:
    """Copies a file from the container (returns bytes)."""
    if not self.client: return None
    container_name = self._get_container_name(session_id)
    try:
        container = self.client.containers.get(container_name)
        if container.status != 'running': return None

        log.debug(f"Attempting to get file '{container_path}' from '{container_name}'")
        # get_archive returns a stream of the tar archive
        bits, stat = container.get_archive(container_path, encode_stream=True) # Use encoded stream for potential perf

        # Read the tar stream in memory
        file_obj = io.BytesIO()
        for chunk in bits:
            file_obj.write(chunk)
        file_obj.seek(0)

        # Extract the file content from the tar stream (use gzip mode)
        with tarfile.open(fileobj=file_obj, mode='r:gz') as tar:
             # Tar archive paths might be relative, find the first file member
             member_name = os.path.basename(container_path)
             try:
                 # Try finding exact basename match first
                 member = tar.getmember(member_name)
             except KeyError:
                  # If not found, maybe it's the only member? Check members list.
                  all_members = tar.getmembers()
                  if len(all_members) == 1 and all_members[0].isfile():
                       member = all_members[0]
                  elif len(all_members) > 0:
                       # Fallback: find first file if basename didn't match
                       first_file = next((m for m in all_members if m.isfile()), None)
                       if first_file: member = first_file
                       else: raise FileNotFoundError(f"No file found within archive for {container_path}")
                  else:
                       raise FileNotFoundError(f"Archive for {container_path} was empty or invalid.")


             if member.isfile():
                 extracted_file = tar.extractfile(member)
                 if extracted_file:
                     content = extracted_file.read()
                     log.info(f"Successfully retrieved file '{container_path}' ({len(content)} bytes) from '{container_name}'.")
                     return content
                 else:
                      raise IOError(f"Could not extract file data for {member.name} from archive.")
             elif member.isdir():
                  log.error(f"Path '{container_path}' is a directory, cannot get file content.")
                  raise IsADirectoryError(f"Path '{container_path}' is a directory.")
             else:
                  log.error(f"Path '{container_path}' is not a file or directory in archive.")
                  raise FileNotFoundError(f"Item '{container_path}' not a file/dir in archive.")

    except NotFound:
        log.error(f"Container '{container_name}' or path '{container_path}' not found for get_file.")
        # Distinguish between container not found and path not found? API might not.
        try: self.client.containers.get(container_name)
        except NotFound: log.error(f"Container '{container_name}' not found."); raise FileNotFoundError(f"Container {container_name} not found.") from None
        # If container exists, path must be wrong
        log.error(f"Path '{container_path}' not found inside container '{container_name}'.")
        raise FileNotFoundError(f"Path '{container_path}' not found inside container.") from None
    except APIError as e:
        # Check if APIError indicates file not found specifically
        if 'No such container:path' in str(e) or 'Could not find the file' in str(e) or '404' in str(e):
             log.error(f"Path '{container_path}' not found inside container '{container_name}' (APIError: {e}).")
             raise FileNotFoundError(f"Path '{container_path}' not found inside container.") from e
        log.error(f"API error getting file from '{container_name}': {e}")
        raise IOError(f"Docker API error getting file: {e}") from e
    except tarfile.TarError as e:
        log.error(f"Error processing tar archive from '{container_name}': {e}")
        raise IOError(f"Tar archive error: {e}") from e
    except IsADirectoryError: # Re-raise specific error
         raise
    except Exception as e:
        log.exception(f"Unexpected error getting file from '{container_name}':")
        raise IOError(f"Unexpected error getting file: {e}") from e
EOF echo "Created services/sandbox_manager.py"

--- Create services/tool_executor.py ---
cat << 'EOF' > services/tool_executor.py import importlib import inspect import logging import os import pkgutil import json from typing import Any, Dict, List, Callable, Tuple, get_type_hints, Union, Optional, Type

from .sandbox_manager import SandboxManager from config import Config

log = logging.getLogger(name)

class ToolNotFoundException(Exception): """Custom exception for when a tool is not found.""" pass

class ToolExecutionException(Exception): """Custom exception for errors during tool execution.""" pass

--- Tool Loading and Schema Generation ---
def get_available_tool_modules(tools_package_name="tools"): """Dynamically discovers tool modules within the 'tools' package.""" modules = [] try: package = importlib.import_module(tools_package_name) package_path = package.path prefix = package.name + "." except ImportError: log.error(f"Could not import tools package '{tools_package_name}'. No tools will be loaded.") return modules

log.debug(f"Discovering tool modules in: {package_path}")

# Filter out __init__.py and potential utility modules
excluded_modules = {'__init__', 'utils'}

for _, module_name, is_pkg in pkgutil.iter_modules(package_path, prefix):
    base_name = module_name.split('.')[-1]
    if not is_pkg and base_name not in excluded_modules:
        try:
            module = importlib.import_module(module_name)
            modules.append(module)
            log.debug(f"Discovered tool module: {module_name}")
        except ImportError as e:
            log.error(f"Failed to import tool module {module_name}: {e}")
        except Exception as e:
             log.error(f"Unexpected error importing tool module {module_name}: {e}", exc_info=True)
return modules
def extract_tool_functions(module) -> List[Tuple[str, Callable]]: """Extracts functions suitable as tools from a module.""" tools = [] module_name = module.name.split('.')[-1] for name, obj in inspect.getmembers(module): # Check if it's a function defined in this module (not imported) # and doesn't start with an underscore (private) if inspect.isfunction(obj) and obj.module == module.name and not name.startswith('_'): # Check for docstring (required for description) if not inspect.getdoc(obj): log.warning(f"Tool function '{module_name}.{name}' is missing a docstring. Skipping.") continue tools.append((f"{module_name}.{name}", obj)) log.debug(f"Found tool function: {module_name}.{name}") return tools

def format_type_hint_to_openapi_type(hint: Type) -> Tuple[str, Optional[str], Optional[List[str]]]: """Converts Python type hints to OpenAPI schema type, format, and enum.

Returns:
    Tuple[str, Optional[str], Optional[List[str]]]: (type, format, enum)
"""
origin = getattr(hint, '__origin__', None)
args = getattr(hint, '__args__', [])

# Handle Optional[T] by extracting T
is_optional = False
if origin is Union and type(None) in args:
    is_optional = True
    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) == 1:
        hint = non_none_args[0]
        origin = getattr(hint, '__origin__', None) # Re-evaluate origin and args
        args = getattr(hint, '__args__', [])
    else:
         # Optional[Union[A, B]] is complex, treat as 'any' for simplicity
         return "string", None, None # Or object? String is safer default.

# Handle Literal for enums
if origin is Literal or str(origin) == 'typing.Literal':
     enum_values = list(args)
     # Infer type from first enum value if possible, default to string
     inferred_type = "string"
     if enum_values:
          first_val_type = type(enum_values[0])
          if first_val_type == int: inferred_type = "integer"
          elif first_val_type == float: inferred_type = "number"
          elif first_val_type == bool: inferred_type = "boolean"
     return inferred_type, None, [str(v) for v in enum_values] # Return enum values as strings

# Basic type mapping
if hint == str: return "string", None, None
if hint == int: return "integer", "int64", None # Specify format
if hint == float: return "number", "float", None # Specify format
if hint == bool: return "boolean", None, None
if hint == bytes: return "string", "byte", None # Base64 encoded string

# Handle list/array
if hint == list or origin == list:
    item_type = "string" # Default item type if not specified
    item_format = None
    if args:
         item_type, item_format, _ = format_type_hint_to_openapi_type(args[0]) # Ignore enum for items for now
    # Return array type, let schema add 'items' field
    return "array", None, None

# Handle dict/object
if hint == dict or origin == dict:
    # OpenAPI spec doesn't have a 'dict' type, use 'object'
    # We don't introspect keys/values here, assume generic object
    return "object", None, None

# Fallback for Any or complex types not explicitly handled
if hint == Any or hint == inspect.Parameter.empty:
    # Represent 'Any' as a string for broadest compatibility, or omit type?
    # Omitting might be better for some LLMs. Let's try string.
    return "string", None, None

# Default fallback: represent type as string
type_name = getattr(hint, '__name__', str(hint))
log.debug(f"Unsupported type hint '{hint}', mapping to 'string'.")
return "string", None, None
def format_tool_schema(tool_func: Callable) -> Optional[Dict[str, Any]]: """ Generates a schema dictionary for a tool function using OpenAPI 3.0 format. Uses docstring for description and type hints for parameters. """ try: full_tool_name = f"{tool_func.module.split('.')[-1]}.{tool_func.name}" docstring = inspect.getdoc(tool_func) if not docstring: log.warning(f"Tool {full_tool_name} missing docstring. Cannot generate schema.") return None

    # Parse the docstring (simple approach: first line is summary)
    lines = docstring.strip().split('\n')
    description = lines[0].strip()
    # TODO: Could parse Args/Returns sections for more detail if needed

    sig = inspect.signature(tool_func)
    type_hints = get_type_hints(tool_func)

    properties = {}
    required_params = []

    for name, param in sig.parameters.items():
        # Skip injected context parameters
        if name in ['self', 'cls', 'session_id', 'sandbox_manager', 'config']:
             continue

        hint = type_hints.get(name, inspect.Parameter.empty)

        # Determine if parameter is required (no default value AND not Optional)
        is_required = param.default == inspect.Parameter.empty
        origin = getattr(hint, '__origin__', None)
        args = getattr(hint, '__args__', [])
        if origin is Union and type(None) in args:
             is_required = False # Optional[T] means not required

        if is_required:
            required_params.append(name)

        # Get OpenAPI type, format, enum
        param_type, param_format, param_enum = format_type_hint_to_openapi_type(hint)

        # Basic parameter description (could be enhanced by parsing docstring further)
        param_description = f"Parameter '{name}'"
        if param.default != inspect.Parameter.empty:
             param_description += f" (default: {param.default})"

        prop_schema = {
            "type": param_type,
            "description": param_description
        }
        if param_format:
             prop_schema["format"] = param_format
        if param_enum:
             prop_schema["enum"] = param_enum

        # Handle array items schema
        if param_type == 'array':
             item_hint = args[0] if args else Any
             item_type, item_format, item_enum = format_type_hint_to_openapi_type(item_hint)
             prop_schema["items"] = {"type": item_type}
             if item_format: prop_schema["items"]["format"] = item_format
             # Enums in array items might be too complex for some LLMs, omit for now
             # if item_enum: prop_schema["items"]["enum"] = item_enum

        properties[name] = prop_schema


    schema = {
        "name": full_tool_name, # Use qualified name (module.func)
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        }
    }
    # Only add 'required' array if it's not empty
    if required_params:
         schema["parameters"]["required"] = required_params

    return schema
except Exception as e:
    log.error(f"Failed to generate schema for tool {getattr(tool_func, '__qualname__', 'unknown')}: {e}", exc_info=True)
    return None
def format_tool_schemas_for_prompt(tool_funcs: List[Callable]) -> str: """Formats schemas of multiple tools into a string suitable for an LLM prompt.""" prompt_parts = [] for func in tool_funcs: schema = format_tool_schema(func) if schema: # Format as a readable block for the prompt param_lines = [] props = schema['parameters']['properties'] required = schema['parameters'].get('required', []) for name, details in props.items(): req_str = "(required)" if name in required else "(optional)" enum_str = f" (enum: {details['enum']})" if 'enum' in details else "" type_str = details['type'] if details['type'] == 'array' and 'items' in details: type_str = f"array[{(details['items'].get('type','any'))}]" # Indicate item type # More detailed description if available from docstring parsing later param_lines.append(f" - {name} ({type_str}) {req_str}: {details['description']}{enum_str}")

        params_str = "\n".join(param_lines) if param_lines else "      (No parameters)"

        prompt_parts.append(
            f"  - Tool Name: `{schema['name']}`\n"
            f"    Description: {schema['description']}\n"
            f"    Parameters:\n{params_str}"
        )
return "\n\n".join(prompt_parts)
--- Tool Executor Class ---
class ToolExecutor: """ Discovers, validates, and executes tools available to the agents. Injects necessary context (like session_id, sandbox_manager) into tool calls. """ def init(self, sandbox_manager: SandboxManager, config: Type[Config]): self.sandbox_manager = sandbox_manager self.config = config self.tools: Dict[str, Callable] = self._load_and_validate_tools()

def _load_and_validate_tools(self) -> Dict[str, Callable]:
    """Loads tools from modules and validates their signatures."""
    loaded_tools = {}
    modules = get_available_tool_modules()
    for module in modules:
        tool_functions = extract_tool_functions(module)
        for name, func in tool_functions:
            if name in loaded_tools:
                log.warning(f"Duplicate tool name '{name}' found. Overwriting previous definition from {loaded_tools[name].__module__} with definition from {func.__module__}.")
            # TODO: Add more validation if needed (e.g., check parameter types against schema)
            loaded_tools[name] = func
    log.info(f"Loaded {len(loaded_tools)} tools: {list(loaded_tools.keys())}")
    return loaded_tools

def get_tools(self) -> List[Callable]:
     """Returns a list of the loaded tool function objects."""
     return list(self.tools.values())

def is_valid_tool(self, tool_name: str) -> bool:
     """Checks if a tool name exists."""
     return tool_name in self.tools

def execute_tool(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> Any:
    """
    Executes the specified tool with the given arguments.

    Args:
        session_id: The agent's session ID (for context).
        tool_name: The name of the tool (e.g., 'file_system.read_file').
        args: A dictionary of arguments for the tool.

    Returns:
        The result of the tool execution.

    Raises:
        ToolNotFoundException: If the tool doesn't exist.
        ToolExecutionException: If an error occurs during execution or argument binding.
    """
    log.info(f"Executing tool '{tool_name}' for session '{session_id}' with args: {args}")

    if not isinstance(args, dict):
         log.error(f"Invalid arguments type for tool '{tool_name}'. Expected dict, got {type(args)}.")
         raise ToolExecutionException(f"Invalid arguments type for tool '{tool_name}'. Expected dict.")

    if tool_name not in self.tools:
        log.error(f"Tool not found: {tool_name}")
        raise ToolNotFoundException(f"Tool '{tool_name}' is not available.")

    tool_func = self.tools[tool_name]
    sig = inspect.signature(tool_func)

    # Prepare arguments, injecting context if needed
    final_args = {}
    provided_args = args.copy() # Work on a copy

    try:
        for name, param in sig.parameters.items():
            # Inject context parameters
            if name == 'session_id':
                final_args[name] = session_id
                continue
            if name == 'sandbox_manager':
                 if not self.sandbox_manager:
                      raise ToolExecutionException(f"SandboxManager context required by tool '{tool_name}' but not available.")
                 final_args[name] = self.sandbox_manager
                 continue
            if name == 'config':
                 final_args[name] = self.config # Pass the Config class/object
                 continue

            # Match provided arguments
            if name in provided_args:
                final_args[name] = provided_args.pop(name)
            elif param.default != inspect.Parameter.empty:
                # Use default value if not provided
                final_args[name] = param.default
            else:
                # Required argument missing
                raise ToolExecutionException(f"Missing required argument '{name}' for tool '{tool_name}'")

        # Check for extraneous arguments provided by LLM (often happens)
        if provided_args:
            log.warning(f"Extraneous arguments provided for tool '{tool_name}': {list(provided_args.keys())}. These will be ignored.")

        # --- Execute the tool function ---
        # Use try-except around the actual call to catch errors within the tool
        try:
             result = tool_func(**final_args)
             log.info(f"Tool '{tool_name}' executed successfully.")
             # log.debug(f"Tool '{tool_name}' result: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
             return result
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
             # Catch specific expected OS errors from tools like file_system
             log.warning(f"Tool '{tool_name}' raised expected file system error: {e}")
             raise ToolExecutionException(f"{type(e).__name__}: {e}") from e # Wrap in ToolExecutionException
        except Exception as tool_internal_error:
             # Catch unexpected errors within the tool's logic
             log.exception(f"Unexpected error during execution of tool '{tool_name}' logic:")
             raise ToolExecutionException(f"An unexpected error occurred inside tool '{tool_name}': {tool_internal_error}") from tool_internal_error

    except (TypeError, ValueError) as e:
         # Errors related to argument binding/conversion before calling the tool
         log.error(f"Argument binding/validation error for tool '{tool_name}': {e}", exc_info=True)
         raise ToolExecutionException(f"Invalid arguments provided for tool '{tool_name}': {e}") from e
    # Re-raise ToolExecutionExceptions directly
    except ToolExecutionException:
         raise
    # Catch any other unexpected errors during argument preparation
    except Exception as e:
         log.exception(f"Unexpected error preparing arguments for tool '{tool_name}':")
         raise ToolExecutionException(f"An unexpected error occurred preparing tool '{tool_name}': {e}") from e
EOF echo "Created services/tool_executor.py"

--- Create tools/file_system.py ---
cat << 'EOF' > tools/file_system.py import os import logging from typing import List, Dict from services.sandbox_manager import SandboxManager from services.tool_executor import ToolExecutionException # Use specific exception from config import Config

log = logging.getLogger(name)

--- Security Helper ---
def _is_path_safe(base_path: str, target_path: str) -> bool: """ Checks if the target path is within the base path directory. Prevents path traversal attacks (e.g., ../../etc/passwd). Normalizes paths before comparison. """ try: # Normalize paths to handle ., .., and slashes consistently norm_base = os.path.normpath(base_path) norm_target = os.path.normpath(os.path.join(base_path, target_path))

    # Ensure the base path itself is absolute for reliable comparison
    abs_norm_base = os.path.abspath(norm_base)
    abs_norm_target = os.path.abspath(norm_target)

    # Check if the resolved target path starts with the resolved base path
    # Add os.sep to ensure partial matches like /workspace-extra don't pass for /workspace
    return os.path.commonpath([abs_norm_base]) == os.path.commonpath([abs_norm_base, abs_norm_target]) and \
           abs_norm_target.startswith(abs_norm_base + os.sep) or abs_norm_target == abs_norm_base

except Exception as e:
    log.error(f"Error checking path safety ({base_path}, {target_path}): {e}")
    return False
--- Tool Functions ---
def list_files(session_id: str, sandbox_manager: SandboxManager, path: str = '.') -> List[Dict[str, str | bool]]: """ Lists files and directories within the specified path in the agent's /workspace. Args: path (str): The directory path relative to /workspace to list. Defaults to the workspace root ('.'). Returns: List[Dict[str, str | bool]]: A list of dictionaries, each containing 'name', 'path', and 'is_dir' for an item. Raises: FileNotFoundError: If the path does not exist in the workspace. PermissionError: If access to the path is denied. ToolExecutionException: For other errors during execution. """ workspace_root = "/workspace" log.info(f"Tool 'list_files' called for session {session_id}, path: '{path}'")

# Security Check: Ensure path stays within /workspace
if Config.ENABLE_STRICT_SANDBOX_SECURITY and not _is_path_safe(workspace_root, path):
     log.error(f"Access denied for list_files: Path '{path}' is outside the allowed workspace.")
     raise PermissionError(f"Access denied: Path '{path}' is outside the allowed workspace.")

full_path_in_container = os.path.normpath(os.path.join(workspace_root, path))

# Use ls -p to append '/' to directories, -A to include hidden files (except . and ..)
# Using --format=single-column ensures one entry per line, even with spaces
command = ["ls", "-pA", "--format=single-column", full_path_in_container]
exit_code, output = sandbox_manager.execute_command(session_id, command, workdir=workspace_root)

if exit_code != 0:
    if "No such file or directory" in output:
         log.warning(f"list_files: Path not found in workspace for session {session_id}: '{path}'")
         raise FileNotFoundError(f"Path not found in workspace: '{path}'")
    elif "Permission denied" in output:
         log.error(f"list_files: Permission denied for path '{path}' in session {session_id}. Output: {output}")
         raise PermissionError(f"Permission denied for path: '{path}'")
    else:
         log.error(f"list_files failed for path '{path}'. Exit code: {exit_code}, Output: {output}")
         raise ToolExecutionException(f"Failed to list files in '{path}'. Error: {output}")

files = []
# Normalize the input path for constructing relative paths in results
base_rel_path = os.path.normpath(path) if path != '.' else ''

for line in output.strip().split('\n'):
    if not line: continue
    is_dir = line.endswith('/')
    name = line.rstrip('/') if is_dir else line
    # Construct path relative to workspace root for consistency
    # Ensure it handles the base path correctly
    relative_path = os.path.normpath(os.path.join(base_rel_path, name))
    files.append({"name": name, "path": relative_path, "is_dir": is_dir})

log.info(f"list_files successful for path '{path}'. Found {len(files)} items.")
return files
def read_file(session_id: str, sandbox_manager: SandboxManager, path: str) -> str: """ Reads the content of a specified file within the agent's /workspace. Args: path (str): The path to the file relative to /workspace. Returns: str: The content of the file, potentially truncated. Raises: FileNotFoundError: If the file does not exist. IsADirectoryError: If the path points to a directory. PermissionError: If access to the file is denied. ToolExecutionException: For other errors during execution. """ workspace_root = "/workspace" log.info(f"Tool 'read_file' called for session {session_id}, path: '{path}'")

# Security Check
if Config.ENABLE_STRICT_SANDBOX_SECURITY and not _is_path_safe(workspace_root, path):
     log.error(f"Access denied for read_file: Path '{path}' is outside the allowed workspace.")
     raise PermissionError(f"Access denied: Path '{path}' is outside the allowed workspace.")

full_path_in_container = os.path.normpath(os.path.join(workspace_root, path))

# Use SandboxManager.get_file for potentially better handling of different file types/sizes
try:
     file_bytes = sandbox_manager.get_file(session_id, full_path_in_container)
     if file_bytes is None:
         # This case should ideally be covered by exceptions from get_file now
         log.error(f"read_file: get_file returned None for path '{path}', indicating an issue.")
         raise FileNotFoundError(f"File not found or error retrieving: '{path}'") # Assume not found

     # Attempt to decode as UTF-8, replace errors
     content = file_bytes.decode('utf-8', errors='replace')

     # Limit lines read based on config
     lines = content.splitlines()
     if len(lines) > Config.MAX_FILE_READ_LINES:
          log.warning(f"File '{path}' has {len(lines)} lines, truncating to {Config.MAX_FILE_READ_LINES}.")
          content = "\n".join(lines[:Config.MAX_FILE_READ_LINES]) + f"\n... (file truncated, total {len(lines)} lines)"

     log.info(f"read_file successful for path '{path}'. Read {len(file_bytes)} bytes.")
     return content
except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
     log.warning(f"read_file failed for path '{path}': {e}")
     raise # Re-raise specific file system errors
except IOError as e: # Catch IOErrors from get_file (tar issues, docker errors)
     log.error(f"read_file failed for path '{path}' due to IO error: {e}")
     # Try to map Docker API errors back to file system errors if possible
     if "No such container:path" in str(e) or "Could not find the file" in str(e):
          raise FileNotFoundError(f"File not found in workspace: '{path}'") from e
     raise ToolExecutionException(f"Failed to read file '{path}': {e}") from e
except Exception as e:
     log.exception(f"Unexpected error during read_file for '{path}':")
     raise ToolExecutionException(f"Unexpected error reading file '{path}': {e}") from e
def write_file(session_id: str, sandbox_manager: SandboxManager, path: str, content: str) -> str: """ Writes (or overwrites) the specified content to a file within the agent's /workspace. Creates parent directories if they don't exist. Args: path (str): The path to the file relative to /workspace. content (str): The content to write to the file. Returns: str: A confirmation message. Raises: PermissionError: If access to the path is denied or outside the workspace. ToolExecutionException: For errors during execution (e.g., failed to create dirs, write error). """ workspace_root = "/workspace" log.info(f"Tool 'write_file' called for session {session_id}, path: '{path}', content length: {len(content)}")

# Security Check
if Config.ENABLE_STRICT_SANDBOX_SECURITY and not _is_path_safe(workspace_root, path):
    log.error(f"Access denied for write_file: Path '{path}' is outside the allowed workspace.")
    raise PermissionError(f"Access denied: Path '{path}' is outside the allowed workspace.")

full_path_in_container = os.path.normpath(os.path.join(workspace_root, path))
parent_dir = os.path.dirname(full_path_in_container)

# Ensure parent directory exists using mkdir -p
# Run this first to catch permission issues early if possible
if parent_dir != workspace_root and parent_dir != '/': # Avoid trying to mkdir /workspace or /
     mkdir_command = ["mkdir", "-p", parent_dir]
     exit_code, output = sandbox_manager.execute_command(session_id, mkdir_command, workdir=workspace_root)
     if exit_code != 0:
         log.error(f"write_file failed to create directory '{parent_dir}'. Exit code: {exit_code}, Output: {output}")
         if "Permission denied" in output:
              raise PermissionError(f"Permission denied creating directory for file: '{path}'")
         raise ToolExecutionException(f"Failed to create directory for file '{path}'. Error: {output}")

# Use SandboxManager.put_file via a temporary host file for reliability
temp_host_file = None
try:
     import tempfile
     # Create temp file *outside* the mounted workspace to avoid conflicts/permissions issues on host side
     with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=tempfile.gettempdir()) as tf:
         tf.write(content)
         temp_host_file = tf.name
     log.debug(f"Content written to temporary host file: {temp_host_file}")

     if sandbox_manager.put_file(session_id, temp_host_file, full_path_in_container):
         log.info(f"write_file successful for path '{path}'. Wrote {len(content)} bytes.")
         # Verify write? Optional 'cat' command.
         # _, verify_out = sandbox_manager.execute_command(session_id, ["ls", "-l", full_path_in_container])
         # log.debug(f"Verification ls -l: {verify_out}")
         return f"Successfully wrote {len(content)} characters to '{path}'."
     else:
         # put_file failed, logs should have details from SandboxManager
         raise ToolExecutionException(f"Failed to write file '{path}' using put_archive (check logs for API errors).")

except Exception as e:
     log.exception(f"Error during write_file for '{path}':")
     # Catch potential PermissionError during temp file creation/writing on host
     if isinstance(e, PermissionError):
          raise PermissionError(f"Host permission error during write_file setup: {e}")
     raise ToolExecutionException(f"Failed to write file '{path}': {e}") from e
finally:
     # Clean up temporary host file
     if temp_host_file and os.path.exists(temp_host_file):
         try:
             os.remove(temp_host_file)
             log.debug(f"Cleaned up temporary host file: {temp_host_file}")
         except OSError as e:
             log.error(f"Failed to clean up temporary host file {temp_host_file}: {e}")
Potential other tools: delete_file, move_file, create_directory
EOF echo "Created tools/file_system.py"

--- Create tools/shell.py ---
cat << 'EOF' > tools/shell.py import logging from typing import List, Dict, Union from services.sandbox_manager import SandboxManager from services.tool_executor import ToolExecutionException from config import Config

log = logging.getLogger(name)

def execute_shell_command(session_id: str, sandbox_manager: SandboxManager, command: str, workdir: str = '/workspace') -> str: """ Executes a shell command inside the agent's sandbox environment using 'sh -c'. USE WITH EXTREME CAUTION. Args: command (str): The shell command string to execute. Complex commands (pipes, redirects) are possible. workdir (str): The working directory inside the container (absolute path). Defaults to /workspace. Returns: str: A string containing the exit code and the combined stdout/stderr from the command execution, potentially truncated. Raises: ValueError: If the command contains forbidden patterns. ToolExecutionException: For errors during command execution in the sandbox. """ log.warning(f"Executing shell command for session {session_id} in '{workdir}': {command}")

# --- Security Checks ---
# 1. Basic forbidden commands/patterns (expand this list)
forbidden_patterns = [
    'rm -rf /', 'shutdown', 'reboot', 'mkfs', 'fdisk', # Destructive commands
    'passwd', # Changing passwords
    'docker ', 'kubectl ', # Interacting with Docker/K8s from within sandbox (unless intended)
    'sudo ', # Privilege escalation
    'wget ', 'curl ' # Be cautious with arbitrary downloads, consider a dedicated download tool
    # Add more based on security policy
]
# Check for whole words or patterns starting commands
command_lower = command.lower().strip()
if any(f in command_lower for f in forbidden_patterns):
    log.error(f"Forbidden command pattern detected in '{command}'. Execution denied.")
    raise ValueError("Execution denied: Command contains potentially dangerous patterns.")

# 2. Ensure workdir is absolute and sensible, default to /workspace if invalid
if not workdir or not workdir.startswith('/'):
    log.warning(f"Invalid workdir '{workdir}' provided, defaulting to /workspace.")
    workdir = '/workspace'
# Optionally, add further checks if workdir must be within /workspace
elif Config.ENABLE_STRICT_SANDBOX_SECURITY and not workdir.startswith('/workspace'):
     log.error(f"Execution denied: Workdir '{workdir}' is outside /workspace.")
     raise ValueError("Execution denied: Workdir must be inside /workspace.")


# Execute using the sandbox manager with 'sh -c'
# This allows shell features like pipes, but requires careful quoting within 'command' if needed.
full_command = ["sh", "-c", command]
timeout = Config.SHELL_COMMAND_TIMEOUT

try:
    exit_code, output = sandbox_manager.execute_command(
        session_id,
        full_command,
        workdir=workdir,
        timeout_secs=timeout
    )
except Exception as e:
    # Catch errors from execute_command itself (e.g., Docker API error)
    log.error(f"Failed to execute shell command '{command}' due to sandbox error: {e}", exc_info=True)
    raise ToolExecutionException(f"Sandbox execution failed for command: {e}") from e


# Limit output length sent back to LLM/UI
lines = output.splitlines()
if len(lines) > Config.MAX_SHELL_OUTPUT_LINES:
    truncated_output = "\n".join(lines[:Config.MAX_SHELL_OUTPUT_LINES])
    result_str = f"Exit Code: {exit_code}\nOutput (truncated to {Config.MAX_SHELL_OUTPUT_LINES} lines):\n{truncated_output}\n... (Total {len(lines)} lines)"
    log.warning(f"Shell command output truncated from {len(lines)} lines.")
else:
    result_str = f"Exit Code: {exit_code}\nOutput:\n{output}"

log.info(f"Shell command '{command}' finished with exit code {exit_code}.")
return result_str
EOF echo "Created tools/shell.py"

--- Create tools/web_search.py ---
cat << 'EOF' > tools/web_search.py import logging import requests import json from typing import List, Dict, Any, Optional from config import Config from services.tool_executor import ToolExecutionException

log = logging.getLogger(name)

--- Web Search Tool ---
This example supports Serper.dev and Tavily.
Add more providers by:
1. Adding their endpoint/key info in _get_search_provider_config.
2. Adding specific request formatting logic.
3. Adding specific response parsing logic.
def _get_search_provider_config() -> Optional[Tuple[str, str, str]]: """Helper to determine search provider, endpoint and API key.""" api_key = Config.WEB_SEARCH_API_KEY if not api_key: log.warning("WEB_SEARCH_API_KEY not found in config. Web search tool will not function.") return None

# Simple detection based on key prefix (adjust if needed)
provider = "unknown"
if api_key.startswith("serper_"): # Example custom prefix if needed
    provider = "serper"
elif len(api_key) > 40: # Tavily keys are typically long
     provider = "tavily"
# Add other detection logic here if necessary
# else: provider = "some_other_provider" # Default or raise error?

log.debug(f"Detected search provider: {provider}")

endpoints = {
    "serper": "https://google.serper.dev/search",
    "tavily": "https://api.tavily.com/search",
}
endpoint = endpoints.get(provider)

if not endpoint:
     log.error(f"Search endpoint for detected provider '{provider}' is not configured in web_search.py.")
     return None

return endpoint, api_key, provider
def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]: """ Performs a web search using a configured search provider API (Serper.dev or Tavily). Args: query (str): The search query. num_results (int): The desired number of search results (provider may limit). Default is 5. Returns: List[Dict[str, str]]: A list of search results, typically including 'title', 'link', and 'snippet'. Returns list with error dict on failure. Raises: ToolExecutionException: If the tool is not configured or encounters an API error. """ log.info(f"Tool 'web_search' called with query: '{query}', num_results: {num_results}")

config_result = _get_search_provider_config()
if not config_result:
    raise ToolExecutionException("Web search tool is not configured. Set WEB_SEARCH_API_KEY in .env")

endpoint, api_key, provider = config_result

headers = {}
params = {}
payload = {}
request_method = 'POST' # Default for Serper/Tavily

# --- Adapt request based on provider ---
if provider == "serper":
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query, "num": num_results})
elif provider == "tavily":
     headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
     payload = json.dumps({
         "query": query,
         "search_depth": "basic", # or "advanced"
         "max_results": num_results,
         # "include_answer": True, # Optional: Tavily can provide a summarized answer
         # "include_raw_content": False, # Optional
         # "include_images": False, # Optional
     })
else:
     # Should not happen if _get_search_provider_config is robust
     raise ToolExecutionException(f"Request formatting not implemented for provider '{provider}'.")

try:
    log.debug(f"Sending {request_method} request to {provider} at {endpoint}")
    response = requests.request(request_method, endpoint, headers=headers, data=json.dumps(payload) if payload else None, params=params if params else None, timeout=20) # Increased timeout
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()
    log.debug(f"Web search response received (status {response.status_code})")

    # --- Parse response based on provider ---
    results = []
    if provider == "serper":
        raw_results = data.get('organic', [])
        for item in raw_results[:num_results]:
            results.append({
                "title": item.get('title', 'N/A'),
                "link": item.get('link', '#'),
                "snippet": item.get('snippet', 'N/A')
            })
    elif provider == "tavily":
         raw_results = data.get('results', [])
         # Optional: Include Tavily's summarized answer if requested and available
         # if data.get("answer"):
         #     results.append({"title": "Tavily Answer", "link": "", "snippet": data["answer"]})
         for item in raw_results[:num_results]:
              results.append({
                   "title": item.get('title', 'N/A'),
                   "link": item.get('url', '#'), # Note: key is 'url'
                   "snippet": item.get('content', 'N/A') # Note: key is 'content'
              })
    # Add parsers for other providers here

    log.info(f"Web search successful for query '{query}'. Found {len(results)} results.")
    if not results:
         return [{"title": "No Results", "link": "", "snippet": "The web search returned no results for this query."}]
    return results

except requests.exceptions.Timeout:
    log.error(f"Timeout during web search request for '{query}' to {provider}.")
    raise ToolExecutionException(f"Web search timed out contacting {provider}.")
except requests.exceptions.RequestException as e:
    log.error(f"Error during web search request for '{query}': {e}")
    raise ToolExecutionException(f"Network or API error during web search: {e}")
except json.JSONDecodeError as e:
     log.error(f"Error decoding JSON response from search API ({provider}): {e}. Response: {response.text[:500]}")
     raise ToolExecutionException(f"Invalid JSON response from search API.")
except Exception as e:
    log.exception(f"An unexpected error occurred during web search for '{query}':")
    raise ToolExecutionException(f"An unexpected error occurred during web search: {e}")
EOF echo "Created tools/web_search.py"

--- Create tools/browser.py (Placeholder) ---
cat << 'EOF' > tools/browser.py import logging from typing import List, Dict, Optional from services.sandbox_manager import SandboxManager from services.tool_executor import ToolExecutionException from config import Config

Requires browser automation library (e.g., Playwright, Selenium)
AND the browser itself installed in the SANDBOX image.
Example using Playwright (conceptual)
log = logging.getLogger(name)

--- Playwright Setup (Conceptual) ---
This setup needs to happen within the sandbox if executing there,
or requires careful setup if controlling browser from host -> sandbox.
Option 1: Execute python script inside sandbox via execute_command (complex script needed)
Option 2: Run a dedicated browser service (e.g., Selenium Grid, Playwright Service) connected to sandbox network.
Option 3: Use a cloud-based browser automation service API.
This placeholder uses a simple requests/BeautifulSoup fallback.
For real browser automation, replace the logic below.
def browse_website(session_id: str, sandbox_manager: SandboxManager, url: str) -> Dict[str, Optional[str]]: """ (Fallback Implementation) Opens a URL and returns simplified text content. Does NOT run JavaScript. For full browser capabilities, implement using Playwright/Selenium within the sandbox. Args: url (str): The URL to browse. Returns: Dict[str, Optional[str]]: Dictionary containing 'status' ('success' or 'error'), 'content' (extracted text or None), and 'error_message' (if status is 'error'). Raises: ToolExecutionException: For network or parsing errors. """ log.info(f"Tool 'browse_website' (fallback) called for session {session_id}, URL: {url}")

# Security: Validate URL format, block non-http(s) schemes
if not url.startswith(('http://', 'https://')):
    raise ValueError("Invalid URL scheme. Only http and https are allowed.")

try:
     import requests
     from bs4 import BeautifulSoup

     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
     response = requests.get(url, timeout=Config.BROWSER_TIMEOUT, headers=headers, allow_redirects=True)
     response.raise_for_status() # Check for HTTP errors

     # Basic text extraction using BeautifulSoup
     soup = BeautifulSoup(response.text, 'html.parser')
     # Remove script, style, nav, footer elements for cleaner text
     for element in soup(["script", "style", "nav", "footer", "aside", "header", "form"]):
         element.decompose()
     # Get text, strip lines, join with newlines
     text = '\n'.join(line.strip() for line in soup.get_text().splitlines() if line.strip())

     # Limit content length
     max_len = 8000 # Increased limit for browse content
     if len(text) > max_len:
          text = text[:max_len] + f"... (content truncated, total {len(text)} chars)"
     log.info(f"browse_website fallback successful for {url}. Extracted ~{len(text)} chars.")
     return {"status": "success", "content": text, "error_message": None}

except requests.exceptions.Timeout:
    log.error(f"Fallback browse timed out for {url}")
    raise ToolExecutionException(f"Timeout fetching URL: {url}")
except requests.exceptions.RequestException as e:
     log.error(f"Fallback browse using requests failed for {url}: {e}")
     raise ToolExecutionException(f"Failed to fetch URL: {e}")
except Exception as e:
     log.exception(f"Unexpected error during fallback browse for {url}:")
     raise ToolExecutionException(f"An unexpected error occurred during browsing: {e}")
def extract_text(session_id: str, sandbox_manager: SandboxManager, url: str) -> str: """ (Placeholder) Extracts the main text content from a webpage URL using the browse_website tool. Args: url (str): The URL to extract text from. Returns: str: The extracted text