---
name: gemini
description: Use Gemini 3.8 Flash through MCP for text/image prompts, structured JSON output, client function calling and API usage tracking. Use for interactive prompt testing or model calls inside Python controllers.
---

# Gemini

Write your own prompts and response-processing code. The tool adds no perception
instructions, task-solving logic or action selection.
Use the `gemini_generate` tool for interactive calls; in your own Python, connect
with `ctx = Context()` (`from api.runtime import Context`) and call
`ctx.call("gemini_generate", ...)`.

## Recommended practice

Use Gemini for visual and spatial perception: identifying objects, locating them in
images, and interpreting spatial relationships. Keep a hard-coded rule only where it
is verified across layouts; colour, size and dark-pixel rules have broken on new
views. Gemini calls add cost and latency, and outputs can vary between calls.
Test other uses, such as structured decisions, before relying on them.

### Helper module

Start from [`scripts/vlm.py`](scripts/vlm.py) instead of writing your own calls. It
applies the principles below and returns `{ok, ..., why}` instead of raising:

- `check(ctx, image, question)`: a yes / no / unsure question.
- `point(ctx, image, prompt, {name: description})`: named points, as pixels.
- `segment(ctx, image, what)`: masks, as pixel polygons and a full-size mask.
- `usage()`: the calls, time and cost so far.

`image` is an image file, a BGR array, or an observation frame directory (pass
`camera=`). Load the module by path; do not copy it:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("vlm", ".agents/skills/gemini/scripts/vlm.py")
vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm)
```

### Principles

- **Send small images.** Downscale to about 640 px wide JPEG. A full-size request
  timed out after about 65 s and held a large budget reservation.
- **Use Gemini for 2-D questions; use depth for 3-D ones.** Gemini is reliable for
  identity, visibility, gross state, points and masks. It is not reliable for small
  3-D changes such as "lifted?" or "touching?"; measure those with depth.
- **Know the coordinate format.** Points and `box_2d` come back as [y, x],
  normalized to 0-1000 over the full image. Polygon masks usually come back as
  [y, x] too, even when the prompt asks for [x, y]; decide the order per answer by
  which reading falls inside `box_2d`. Draw a new prompt's output on the image once
  before trusting it.
- **Ask for more geometry, not less.** Several points or a segmentation mask
  survive a hard view better than two points.
- **Calls are serialized with robot calls.** A pending Gemini call delays robot
  tools, so ask between motions.
- **Treat a failure as no answer.** A timeout, a refusal or a truncated reply means
  no answer. Check `budget` and stop calling when little remains.

### Pointing for triangulation

Ask for the same named, distinctive point on the target in each camera view.
{% if camera_geometry %}
Accept the answers only when `{{ robot.pixel }}` places them at the same base-frame
point (small spread between views and in the local depth).
{% else %}
Accept the answers only when every view marks the same physical feature, and
confirm with measured motion as in [object-localization](../object-localization/SKILL.md).
Keep a non-Gemini fallback.

{% endif %}
## MCP usage

The tool follows OpenRouter's **chat-completions** format, not the Google SDK's
`contents`/`generationConfig` format. Read [API reference](references/api.md) for
request fields, structured-output schemas, conversation history and a complete
function-calling round trip. Supply only fields you need. The model and credentials
are fixed for you.

For a visual prompt from Python:

```python
import base64
import json
from pathlib import Path

png_bytes = Path("output/crop.png").read_bytes()  # Or in-memory PNG bytes.
image = {"type": "image", "mimeType": "image/png",
         "data": base64.b64encode(png_bytes).decode("ascii")}
reply = ctx.call("gemini_generate", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the visible objects and their relative positions."},
        image,
    ],
}], max_tokens=256)
result = json.loads(reply["content"][0]["text"])
print(result["message"], result["usage"], result["budget"])
```

`max_tokens` covers reasoning as well as the visible answer; even a one-word reply
spends tokens on reasoning, and 2048 has truncated a pointing answer. Set
`max_tokens` high (up to 16384) and control speed with `reasoning.effort`
(`minimal` for segmentation, `low` for pointing and checks).
`finish_reason="length"` means the answer is incomplete: retry; never act on a
partial answer.

Use ordinary Python for file reads, image processing and base64 encoding. For JPEG,
set `mimeType` to `image/jpeg`.
Any readable workspace image works.
The tool accepts image bytes, not file paths or remote URLs. No SDK helper is needed.
Do not print large base64 payloads into agent context just to pass them back.

For a live image, reuse an image content block returned by `ctx.call("{{ robot.observe }}")`
directly (select the intended camera from the response order/metadata).

Limits: eight images per call, 4 MiB per image, 8 MiB total, at most 4096×4096
pixels per image; PNG/JPEG only, no animation. Resize or compress locally if needed.
Supplied images are controller input, not trusted evidence of task success.
The tool calls Gemini 3.8 Flash through OpenRouter; credentials stay with the host
robot service. Results contain `message`, `finish_reason`, `usage`, `cost_usd`
and cumulative/remaining `budget`; missing provider cost is `null`.
Spending is capped; see [Spending limit](references/api.md#spending-limit), and check
the returned `budget` before further calls. Function requests in
`message.tool_calls` are data, never automatically executed. Validate and dispatch
them in your script; robot actions still go through robot MCP. Preserve the whole
assistant message, including `reasoning_details`, when returning tool results.
