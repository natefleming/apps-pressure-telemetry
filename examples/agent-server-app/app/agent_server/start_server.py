"""Entry point: `python -m agent_server.start_server --workers 2` (the MLflow agent template layout)."""

from mlflow.genai.agent_server import AgentServer

import agent_server.agent  # registers the @invoke / @stream handlers
from databricks_apps_telemetry import instrument

agent_server: AgentServer = AgentServer("ResponsesAgent")
app = agent_server.app
instrument(agent_server)  # the only telemetry code in the app


def main() -> None:
    agent_server.run(app_import_string="agent_server.start_server:app")


if __name__ == "__main__":
    main()
