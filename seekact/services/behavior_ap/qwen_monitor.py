"""Stateless visual interaction verifier; never imports or advances simulation."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError
import http.client

CONTINUE = '继续'
COMPLETED = '已完成'

def parse_verdict(value):
    if not isinstance(value,str):raise ValueError('Qwen response must be text')
    text = value.strip()
    if text not in (CONTINUE, COMPLETED):
        raise ValueError('Qwen must output exactly 继续 or 已完成')
    return text

class CompletionVotes:
    def __init__(self):
        self.reset()

    def reset(self):
        self.streak = 0
        self.last_step = None

    def add(self, verdict, step):
        verdict = parse_verdict(verdict)
        if self.last_step is not None and step <= self.last_step:
            raise ValueError('Completion votes require distinct increasing physical frames')
        self.last_step = step
        self.streak = self.streak + 1 if verdict == COMPLETED else 0
        return self.streak >= 2

class QwenMonitor:
    def __init__(self, options):
        self.options = dict(options)
        self.opener = build_opener(ProxyHandler({}))
        self.key_path = Path(options['api_key_file'])

    def check(self, observation, goal, directory, stop):
        """Only one current-frame image + current local goal. No history/state/task oracle."""
        directory = Path(directory)
        images = observation['image_paths']
        if len(images) != 3:
            raise ValueError('Expected head and two wrist images for current frame')
        # One current frame, tiled camera views, no previous frames or telemetry.
        from PIL import Image
        import io
        views = [Image.open(p).convert('RGB') for p in images]
        width, height = max(v.width for v in views), max(v.height for v in views)
        frame = Image.new('RGB', (width * 3, height))
        for i, view in enumerate(views):
            frame.paste(view, (i * width, 0))
        buffer = io.BytesIO(); frame.save(buffer, format='JPEG', quality=90)
        frame_bytes = buffer.getvalue()
        (directory / 'current_frame.jpg').write_bytes(frame_bytes)
        prompt = ('判断当前画面中这个局部交互任务是否已经完成：' + goal +
            '\n仅输出“继续”或“已完成”，不要解释。只根据当前帧可见证据判断。'
            '抓取须看到物体被拿起并受夹爪支撑，放置须看到物体已释放且在指定位置，'
            '开机须看到设备已开启的证据；证据不足输出“继续”。')
        body = dict(model=self.options['model'], temperature=0, max_tokens=16,
            chat_template_kwargs={'enable_thinking':False},
            messages=[{'role':'user','content':[
                {'type':'text','text':prompt},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
                    base64.b64encode(frame_bytes).decode()}}]}])
        payload = json.dumps(body, ensure_ascii=False).encode()
        (directory / 'request.json').write_bytes(payload)
        attempt = 0
        while not stop.is_set():
            started = time.monotonic()
            try:
                req = Request(self.options['base_url'].rstrip('/')+'/chat/completions', data=payload,
                    headers={'Content-Type':'application/json',
                             'Authorization':'Bearer '+self.key_path.read_text().strip()})
                with self.opener.open(req, timeout=30) as response:
                    result = json.load(response)
                (directory / 'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                verdict = parse_verdict(result['choices'][0]['message']['content'])
                record = dict(verdict=verdict, observation_step=observation['step'],
                    interaction_goal=goal, frame_sha256=hashlib.sha256(frame_bytes).hexdigest(),
                    elapsed_seconds=time.monotonic()-started, attempt=attempt,
                    model=self.options['model'], official_success_claim=False)
                (directory / 'verdict.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
                return record
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException,
                    ValueError, KeyError, IndexError, TypeError) as exc:
                # Credentials and HTTP response bodies never enter logs. Invalid replies don't vote.
                with (directory / 'waiting.jsonl').open('a') as log:
                    log.write(json.dumps(dict(attempt=attempt, error_type=type(exc).__name__,
                        elapsed_seconds=time.monotonic()-started, physics_frozen=True))+'\n')
                if isinstance(exc, HTTPError) and exc.code in (401, 403, 404):
                    raise RuntimeError('Qwen authentication/model endpoint rejected request') from None
                attempt += 1
                stop.wait(min(attempt, 10))
        raise RuntimeError('Qwen request cancelled; no additional action admitted')
