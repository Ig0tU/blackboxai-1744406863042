
Built by https://www.blackbox.ai

---

```markdown
# NLP Agent Sandbox (Gemini / Hugging Face Edition)

This project provides a framework for building an AI agent capable of interacting with various tools within a secure sandboxed environment (Docker). It utilizes Flask for the backend, SocketIO for real-time communication, and allows integration with either Google Gemini or Hugging Face LLMs for natural language processing and tool execution.

## Project Overview
The NLP Agent Sandbox is designed to provide a modular and extensible architecture for creating AI agents. Features include modular components for agents, tools, services, and routes; support for both Google Gemini and Hugging Face models; sandboxed execution of tools; and a web interface for interacting with agents.

## Installation
### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Bash (commonly available on Linux/macOS/WSL)

### Steps to Install
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nlp-agent-sandbox-llm.git
   cd nlp-agent-sandbox-llm
   ```

2. **Run the initialization script:**
   ```bash
   bash create_nlp_sandbox_llm.sh
   ```

3. **Configure environment variables:**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set your secrets and choose the LLM provider.

4. **Build and run the application with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

5. **Access the Application:**
   Open your web browser to `http://localhost:5000` (or the port specified by `APP_PORT` in your `.env`).

## Usage
1. Select an agent persona from the dropdown (available personas include `Assistant`, `Lovable Agent`, `Cursor Agent`, etc.).
2. Enter your initial goal or task in the text area.
3. Click "Start Agent" to begin the interaction.
4. Communicate with the agent using the message input field and observe the interaction log.
5. The workspace allows access to files created in the agent's Docker container.

## Features
- **Modular Architecture:** Components for agents, tools, services, and routes are well separated.
- **LLM Agnostic (Gemini/HF):** Easily switch between Google Gemini and Hugging Face's models.
- **Tool Calling:**
  - Gemini supports native function calling.
  - Hugging Face depends on prompt engineering for tool invocation.
- **Sandboxed Execution:** Isolated environments using Docker for tool operations.
- **Web Interface:** User-friendly UI for agent management and monitoring.
- **Real-time Updates:** Leverages WebSockets for live updates and communications.

## Dependencies
The following dependencies are defined in `requirements.txt`:

```plaintext
Flask>=2.3.0
Flask-SocketIO>=5.3.0
gevent-websocket>=0.10.1
gunicorn>=21.2.0
python-dotenv>=1.0.0
docker>=6.1.0
requests>=2.30.0
beautifulsoup4>=4.12.0
google-generativeai>=0.5.0
huggingface_hub>=0.20.0
```

Additional libraries may be required based on tool functionality (like pandas, numpy, etc.) as they are implemented.

## Project Structure
```plaintext
nlp-agent-sandbox-llm/
├── agents/                 # Agent logic (BaseAgent, specific personas)
│   ├── __init__.py
│   ├── base_agent.py       # Core agent class with LLM interaction (Gemini/HF)
│   ├── lovable_agent.py    # Example persona
│   └── cursor_agent.py     # Example persona
├── routes/                 # Flask API endpoints and WebSocket handlers
│   ├── __init__.py
│   ├── main_routes.py      # Basic web routes (index page, health check)
│   └── agent_routes.py     # API/WebSocket routes for agent management & workspace
├── services/               # Backend services (Docker interaction, tool execution)
│   ├── __init__.py
│   ├── sandbox_manager.py   # Manages Docker sandbox containers
│   └── tool_executor.py     # Dispatches calls to specific tools
├── tools/                  # Implementations of available tools
│   ├── __init__.py
│   ├── file_system.py       # Interactions with file system
│   └── shell.py             # Execute shell commands
├── templates/              # HTML templates for frontend (Jinja2)
│   ├── base.html
│   └── index.html
├── static/                 # Static assets (CSS, JavaScript)
│   ├── css/style.css
│   └── js/main.js
├── sandbox_workspace/      # Directory for agent workspace files
│   └── README.md
├── app.py                  # Main Flask application entry point
├── config.py               # Application configuration
├── requirements.txt        # Python dependencies
├── Dockerfile               # Docker image definition for the application
├── docker-compose.yml      # Docker Compose configuration
├── .env.example            # Example environment variables
├── create_nlp_sandbox_llm.sh # Script to initialize project structure
└── .gitignore              # Git ignore rules
```

## Important Notes
- **Security:** The use of Docker provides a layer of isolation, but caution should be exercised when configuring security settings, particularly when allowing tool invocations that can execute arbitrary shell commands.
- **Error Handling:** The provided implementation may need additional error handling for production scenarios.
- **Development:** Plans to extend the capabilities of the agent (tool implementations, agent personas) are welcome from contributors.

Feel free to contribute or customize based on your needs!
```