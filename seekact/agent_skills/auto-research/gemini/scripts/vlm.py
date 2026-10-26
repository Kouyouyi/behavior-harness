"""Gemini (`gemini_generate`) vision helpers: yes / no checks, named points, segmentation masks.

Copy nothing: import this file from your code (see the gemini skill, "Helper module"). Every answer is evidence
for your code, not a decision; geometry (depth, kinematics) stays in your code.

Measured on a robot head camera (2026-10, practice_pnp_box session):
- 640 px wide JPEG (q85, ~35 kB), json_schema, reasoning effort low: 2-6 s, ~$0.004 per call, points within ~6 px
  of depth geometry. A full 1280x720 request timed out after ~65 s and held a large budget reservation.
- max_tokens is shared with reasoning: 2048 truncated a pointing answer; use the 16384 cap, set speed with effort
  ('low' for checks and points, 'minimal' for segmentation, 3-6 s).
- Points and box_2d are [y, x] normalized to 0-1000 over the image sent. Polygon masks came back as [y, x] too,
  even when asked for [x, y]: segment() decides the order per answer by containment in the answer's box_2d.
- The host serializes MCP calls: a pending Gemini call delays robot calls. Ask between motions.
- Reliable for identity, visibility, gross state, points and masks; not for small 3-D changes (lifted / touching):
  measure those with depth. Draw a new prompt's output on the image once before trusting it.
"""
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640                 # px sent to Gemini
MAX_TOKENS = 16384          # the cap; reasoning shares it
MIN_BUDGET_USD = 0.5        # refuse to call below this remaining budget
PT = {"type": "array", "items": {"type": "integer"},
      "description": "[y, x] normalized to 0-1000 over the image height and width"}
SEG_PROMPT = ('Give the segmentation mask for {what}. Output a JSON list of segmentation masks where each entry '
              'contains the 2D bounding box in the key "box_2d" as [ymin, xmin, ymax, xmax] normalized to 0-1000, the '
              'segmentation mask in key "mask" as a polygon of points normalized to 0-1000, and the text label in the '
              'key "label".')

_last = {'remaining': None}
calls = []                  # (seconds, cost, name) of every call in this process


def load_image(src, camera=None):
    """BGR image from an array, an image file, or an observation frame directory (`<camera>.jpg` inside;
    `camera` may be omitted when the directory holds one image)."""
    if isinstance(src, np.ndarray):
        return src
    p = Path(str(src))
    if p.is_dir():
        names = [p / f'{camera}.jpg'] if camera else sorted(p.glob('*.jpg')) + sorted(p.glob('*.png'))
        if len(names) != 1:
            raise ValueError(f'{p}: pass camera=, images here: {[n.name for n in names]}')
        p = names[0]
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(p)
    return im


def image_block(src, width=WIDTH, crop=None, quality=85, camera=None):
    """-> (MCP image block, (w, h) of the full-size region the points refer to, (x0, y0) offset of the crop).
    crop: (x0, y0, x1, y1) in full-size pixels."""
    im = load_image(src, camera)
    x0 = y0 = 0
    if crop is not None:
        x0, y0, x1, y1 = [int(v) for v in crop]
        im = im[y0:y1, x0:x1]
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, round(width * h / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    block = {"type": "image", "mimeType": "image/jpeg", "data": base64.b64encode(buf.tobytes()).decode('ascii')}
    return block, (w, h), (x0, y0)


def to_px(pt, size, offset=(0, 0)):
    """[y, x] in 0-1000 -> [u, v] full-size pixels."""
    return [round(pt[1] / 1000 * size[0] + offset[0]), round(pt[0] / 1000 * size[1] + offset[1])]


def _call(ctx, name, content, response_format, effort, max_tokens):
    """One gemini_generate call -> ({ok, why?, seconds}, parsed JSON answer or None). Never raises for a
    provider problem: a timeout, refusal, truncation or non-JSON reply is ok False with the reason."""
    if _last['remaining'] is not None and _last['remaining'] < MIN_BUDGET_USD:
        return {'ok': False, 'why': f"Gemini budget low ({_last['remaining']:.2f} USD left)"}, None
    t0 = time.time()
    try:
        reply = ctx.call("gemini_generate", messages=[{"role": "user", "content": content}],
                         response_format=response_format, reasoning={"effort": effort}, max_tokens=max_tokens)
    except RuntimeError as e:
        return {'ok': False, 'why': f'gemini error: {e}', 'seconds': round(time.time() - t0, 1)}, None
    r = json.loads(reply["content"][0]["text"])
    dt = round(time.time() - t0, 1)
    _last['remaining'] = (r.get('budget') or {}).get('session_cost_remaining_usd')
    calls.append((dt, r.get('cost_usd'), name))
    meta = {'seconds': dt, 'cost_usd': r.get('cost_usd'), 'remaining_usd': _last['remaining']}
    msg = r.get('message') or {}
    if r.get('finish_reason') != 'stop' or msg.get('refusal') or not msg.get('content'):
        return dict(meta, ok=False, why=f"gemini finish {r.get('finish_reason')} refusal {msg.get('refusal')}"), None
    try:
        return dict(meta, ok=True), json.loads(msg['content'])
    except ValueError:
        return dict(meta, ok=False, why=f"gemini returned non-JSON: {msg['content'][:200]}"), None


def ask(ctx, question, images, schema, name='answer', effort='low', max_tokens=MAX_TOKENS, log=None):
    """One structured question about up to 8 image blocks -> {ok, answer, seconds, cost_usd, why}."""
    fmt = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    r, ans = _call(ctx, name, [{"type": "text", "text": question}] + list(images), fmt, effort, max_tokens)
    if log:
        log(f"gemini {name} {r.get('seconds')}s ${r.get('cost_usd')}: {json.dumps(ans)[:300] if r['ok'] else r['why']}")
    return dict(r, answer=ans) if r['ok'] else r


def check(ctx, frames, question, log=None, crop=None, camera=None):
    """Yes / no question about one or more images -> {ok, yes, answer, reason, seconds}; ok False = no usable
    answer (including 'unsure')."""
    schema = {"type": "object", "additionalProperties": False, "required": ["answer", "reason"],
              "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                             "reason": {"type": "string", "description": "one short sentence"}}}
    frames = frames if isinstance(frames, (list, tuple)) else [frames]
    blocks = [image_block(f, crop=crop, camera=camera)[0] for f in frames]
    r = ask(ctx, question + ' Answer yes, no or unsure, with a one-sentence reason.', blocks, schema,
            name='check', log=log)
    if not r['ok']:
        return r
    a = r['answer']
    return {'ok': a['answer'] != 'unsure', 'yes': a['answer'] == 'yes', 'answer': a['answer'],
            'reason': a['reason'], 'seconds': r['seconds']}


def point(ctx, frame, prompt, points, flags=None, lists=None, log=None, crop=None, camera=None):
    """Named points (and boolean flags / string lists) on one image.
    points: {name: 'description'} for one point, or {name: ('description', n)} for a list of about n points.
    -> {ok, px: {name: [u, v] or [[u, v], ...]}, flags, lists, seconds, raw}"""
    props, req = {}, []
    for k, d in points.items():
        props[k] = ({"type": "array", "items": PT, "description": f'{d[0]} (about {d[1]} points)'}
                    if isinstance(d, tuple) else dict(PT, description=f'{d}; [y, x] normalized to 0-1000'))
        req.append(k)
    for k, d in (flags or {}).items():
        props[k] = {"type": "boolean", "description": d}
        req.append(k)
    for k, d in (lists or {}).items():
        props[k] = {"type": "array", "items": {"type": "string"}, "description": d}
        req.append(k)
    schema = {"type": "object", "additionalProperties": False, "required": req, "properties": props}
    block, size, off = image_block(frame, crop=crop, camera=camera)
    r = ask(ctx, prompt + ' Points are [y, x] normalized to 0-1000.', [block], schema, name='point', log=log)
    if not r['ok']:
        return r
    a = r['answer']
    try:
        px = {k: ([to_px(p, size, off) for p in a[k] if len(p) == 2] if isinstance(d, tuple)
                  else to_px(a[k], size, off) if len(a[k]) == 2 else None) for k, d in points.items()}
    except (TypeError, KeyError, IndexError) as e:
        return {'ok': False, 'why': f'malformed points {e}: {a}', 'seconds': r['seconds']}
    return {'ok': True, 'px': px, 'flags': {k: a[k] for k in (flags or {})},
            'lists': {k: a[k] for k in (lists or {})}, 'seconds': r['seconds'], 'raw': a}


def segment(ctx, frame, what, effort='minimal', log=None, camera=None):
    """Segmentation masks of `what` in one image (Gemini's box_2d + polygon + label format).
    -> {ok, masks: [{label, box_px (u0, v0, u1, v1), poly_px [[u, v], ...]}], mask (uint8 full-size, 1 inside any
    polygon), seconds}. Each polygon's [y, x] / [x, y] order is decided by which reading lies inside its box_2d."""
    full = load_image(frame, camera)
    H, W = full.shape[:2]
    block, (w, h), (x0, y0) = image_block(full)
    r, ans = _call(ctx, 'segment', [{"type": "text", "text": SEG_PROMPT.format(what=what)}, block],
                   {"type": "json_object"}, effort, MAX_TOKENS)
    if not r['ok']:
        return r
    if isinstance(ans, dict):
        ans = next((v for v in ans.values() if isinstance(v, list)), [ans])
    mask, masks = np.zeros((H, W), np.uint8), []
    for m in ans:
        try:
            yb0, xb0, yb1, xb1 = m['box_2d']
            poly = np.array(m.get('mask') or [], float)
        except (KeyError, TypeError, ValueError):
            continue
        box = [xb0 / 1000 * w + x0, yb0 / 1000 * h + y0, xb1 / 1000 * w + x0, yb1 / 1000 * h + y0]
        pp = np.zeros((0, 2), np.int32)
        if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3:
            def inside(cols, rows):
                return float(np.mean((cols >= xb0 - 15) & (cols <= xb1 + 15) & (rows >= yb0 - 15) & (rows <= yb1 + 15)))
            yx = inside(poly[:, 1], poly[:, 0]) >= inside(poly[:, 0], poly[:, 1])
            cols, rows = (poly[:, 1], poly[:, 0]) if yx else (poly[:, 0], poly[:, 1])
            pp = np.c_[cols / 1000 * w + x0, rows / 1000 * h + y0].astype(np.int32)
            cv2.fillPoly(mask, [pp], 1)
        masks.append({'label': m.get('label', ''), 'box_px': [round(v) for v in box], 'poly_px': pp.tolist()})
    if log:
        log(f"gemini segment {r['seconds']}s ${r.get('cost_usd')}: {len(masks)} masks, {int(mask.sum())} px")
    return {'ok': bool(masks), 'masks': masks, 'mask': mask, 'seconds': r['seconds'],
            'why': '' if masks else 'no masks'}


def usage():
    """Calls made by this process: count, total seconds, total cost."""
    return {'calls': len(calls), 'seconds': round(sum(c[0] for c in calls), 1),
            'cost_usd': round(sum(c[1] or 0 for c in calls), 4)}
