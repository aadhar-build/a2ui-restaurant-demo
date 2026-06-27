# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from agent import RestaurantAgent
from agent_executor import RestaurantAgentExecutor
from browser_a2a import make_browser_a2a_handler
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route  # used in app.router.routes.insert()
from starlette.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))


class MissingAPIKeyError(Exception):
  """Exception for missing API key."""


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=None, type=int)
def main(host, port):
  try:
    if port is None:
      port = int(os.getenv("PORT", "10002"))

    # Check for API key only if Vertex AI is not configured
    if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE":
      if not os.getenv("GEMINI_API_KEY"):
        raise MissingAPIKeyError(
            "GEMINI_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI"
            " is not TRUE."
        )

    # On Railway, RAILWAY_PUBLIC_DOMAIN is set; fall back to localhost for local dev.
    # Strip any accidental scheme prefix (e.g. "https://foo.railway.app" → "foo.railway.app")
    # so the f-string never produces a double-scheme URL.
    # Use "localhost" not the binding host (0.0.0.0) so browser image requests resolve.
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").removeprefix("https://").removeprefix("http://")
    base_url = f"https://{railway_domain}" if railway_domain else f"http://localhost:{port}"

    agent = RestaurantAgent(base_url=base_url)

    agent_executor = RestaurantAgentExecutor(agent)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent.agent_card, http_handler=request_handler
    )
    import uvicorn

    app = server.build()

    # Expose the browser-facing /a2a SSE endpoint before other routes.
    app.router.routes.insert(
        0, Route('/a2a', endpoint=make_browser_a2a_handler(agent), methods=['POST'])
    )

    # CORS: allow all origins so the frontend (any Railway domain) can connect.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve agent images under /static.
    images_dir = os.path.join(_HERE, 'images')
    if os.path.exists(images_dir):
      app.mount("/static", StaticFiles(directory=images_dir), name="static")

    # Serve the built React frontend as a catch-all SPA fallback.
    dist_dir = os.path.join(_HERE, 'dist')
    if os.path.exists(dist_dir):
      app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
      logger.info("Serving React frontend from %s", dist_dir)
    else:
      logger.info("No dist/ found — frontend not served (dev mode or build not run)")

    uvicorn.run(app, host=host, port=port)
  except MissingAPIKeyError as e:
    logger.error(f"Error: {e}")
    exit(1)
  except Exception as e:
    logger.error(f"An error occurred during server startup: {e}")
    exit(1)


if __name__ == "__main__":
  main()
