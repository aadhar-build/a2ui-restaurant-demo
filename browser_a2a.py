# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser-facing /a2a SSE endpoint.

Translates plain-text or A2UI JSON-action POSTs from the React shell
into agent.stream() calls and streams the results back as SSE in the
format the React client.ts expects:

  data: [{"kind": "data", "data": {...}}, ...]
  data: [{"kind": "text", "text": "..."}]
"""

import json
import logging
import uuid

from a2a.types import DataPart, TextPart
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

_MAX_PAYLOAD_BYTES = 1024 * 1024  # 1 MB


def _part_to_dict(part) -> dict | None:
  root = getattr(part, 'root', part)
  if isinstance(root, DataPart):
    return {
      'kind': 'data',
      'data': root.data,
      'mimeType': getattr(root, 'mimeType', 'application/a2ui+json'),
    }
  if isinstance(root, TextPart):
    return {'kind': 'text', 'text': root.text}
  return None


def _action_to_nl_query(body_str: str) -> str:
  """Convert a v0.9 JSON action payload to the natural-language query format
  that agent_executor.py normally constructs. Without this conversion the raw
  JSON reaches Gemini as-is and the model mirrors context objects directly into
  updateDataModel values, producing [object Object] in the rendered UI."""
  try:
    payload = json.loads(body_str)
  except (json.JSONDecodeError, ValueError):
    return body_str

  if not (payload.get('version') == 'v0.9' and 'action' in payload):
    return body_str

  action = payload['action']
  action_name = action.get('name', '')
  ctx = action.get('context', {})

  if action_name == 'book_restaurant':
    restaurant_name = ctx.get('restaurantName', 'Unknown Restaurant')
    address = ctx.get('address', 'Address not provided')
    image_url = ctx.get('imageUrl', '')
    return (
      f"USER_WANTS_TO_BOOK: {restaurant_name}, Address: {address},"
      f" ImageURL: {image_url}"
    )

  if action_name == 'submit_booking':
    restaurant_name = ctx.get('restaurantName', 'Unknown Restaurant')
    party_size = ctx.get('partySize', 'Unknown Size')
    reservation_time = ctx.get('reservationTime', 'Unknown Time')
    dietary_reqs = ctx.get('dietary', 'None')
    image_url = ctx.get('imageUrl', '')
    return (
      f"User submitted a booking for {restaurant_name} for {party_size} people at"
      f" {reservation_time} with dietary requirements: {dietary_reqs}. The image"
      f" URL is {image_url}"
    )

  return f"User submitted an event: {action_name} with data: {ctx}"


def make_browser_a2a_handler(agent):
  """Return an async Starlette endpoint bound to *agent*."""

  async def handler(request: Request):
    body = await request.body()
    if len(body) > _MAX_PAYLOAD_BYTES:
      return JSONResponse({'error': 'Payload too large'}, status_code=413)

    body_str = body.decode('utf-8', errors='replace').strip()
    query = _action_to_nl_query(body_str)

    # Detect A2UI protocol version from the extension URI header.
    # The React middleware sends: https://a2ui.org/a2a-extension/a2ui/v0.9
    # The agent's internal keys are '0.8' and '0.9' (no 'v' prefix).
    extensions = request.headers.get('X-A2A-Extensions', '')
    if 'v0.8' in extensions or '/0.8' in extensions:
      ui_version = '0.8'
    else:
      ui_version = '0.9'

    session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
    logger.info('[/a2a] version=%s session=%s query=%r', ui_version, session_id, query[:120])

    async def generate():
      try:
        async for chunk in agent.stream(
          query, session_id, ui_version=ui_version, use_streaming=True
        ):
          formatted = []

          for part in chunk.get('parts', []):
            d = _part_to_dict(part)
            if d:
              formatted.append(d)

          update = chunk.get('updates')
          if update and not formatted:
            formatted.append({'kind': 'text', 'text': update})

          if formatted:
            yield f'data: {json.dumps(formatted)}\n\n'
      except Exception as exc:
        logger.error('[/a2a] stream error: %s', exc, exc_info=True)
        yield f'data: {json.dumps([{"kind": "error", "text": str(exc)}])}\n\n'

    return StreamingResponse(
      generate(),
      media_type='text/event-stream',
      headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    )

  return handler
