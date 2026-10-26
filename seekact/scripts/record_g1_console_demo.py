#!/usr/bin/env python3
"""Record the real console UI with synthetic data; no robot, model or microphone.

Requires Playwright + Chromium, ffmpeg and ffprobe. See docs/g1/CONSOLE.md.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.console.runtime import Console, STAGES
from services.console.scenario import matches
from services.console.server import make_server


class Clock:
    def __init__(self):
        self.value = time.time()

    def __call__(self):
        return self.value


class PresentationConsole(Console):
    """Use a normal-temperature fixture for this simulated successful task recording."""
    def _demo_telemetry(self):
        value = super()._demo_telemetry()
        for joint in value['joints']:
            if joint['id'] in (4, 10):
                joint.update(case_c=39.2, winding_c=46.6)
        return value


def title_card(page, ending=False):
    page.evaluate('''ending => {
      const old = document.getElementById('recording-title'); if (old) old.remove();
      const card = document.createElement('div'); card.id = 'recording-title';
      Object.assign(card.style, {position:'fixed',inset:'0',zIndex:'999',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',background:'radial-gradient(ellipse at 50% 30%, #1a3832, #071416 75%)',color:'#e3efe8',fontFamily:'inherit',opacity:'1',transition:'opacity .65s'});
      const lines = [
        ['SEEKACT ROBOTICS', '14px', '#8dc4a8', '5px'],
        ['Humanoid Agent', '70px', '#e3efe8', '-2px'],
        [ending ? '从一句指令，到一次完整的任务闭环。' : '让每一次行动，都有清晰的过程。', '26px', '#a9cbbb', '2px'],
        [ending ? '任务拆解  /  观测判断  /  工具调用  /  反馈验证' : '一句指令 · 具身智能交互演示', '16px', '#84a997', '2px'],
        ['交互原型 · 工具与执行反馈为模拟，历史画面非直播', '13px', '#cbb589', '1px']
      ];
      for (const [text,size,color,spacing] of lines) {
        const line=document.createElement('div');line.textContent=text;
        Object.assign(line.style,{fontSize:size,color,letterSpacing:spacing,marginTop:size==='70px'?'18px':'22px'});card.append(line);
      }
      document.body.append(card);
    }''', ending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', help='Optional Chromium executable path')
    parser.add_argument('--output', type=Path, default=ROOT / '.local-runtime/console-preview/Humanoid-Agent-SeekAct-Robotics-demo.mp4')
    parser.add_argument('--instruction', default='把桌上的米老师玩偶放到沙发上')
    args = parser.parse_args()
    if not matches(args.instruction):
        parser.error('This recording fixture only supports the Mickey-to-sofa demonstration.')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('ffmpeg and ffprobe must be installed.')
    from playwright.sync_api import sync_playwright, expect

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    console = PresentationConsole(clock=clock)
    server, token = make_server(port=0, console=console)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    chapters, errors = [], []
    with tempfile.TemporaryDirectory(prefix='humanoid-demo-') as temporary:
        try:
            with sync_playwright() as playwright:
                options = dict(headless=True, args=['--no-sandbox'])
                if args.browser:
                    options['executable_path'] = args.browser
                browser = playwright.chromium.launch(**options)
                context = browser.new_context(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1,
                                              record_video_dir=temporary, record_video_size={'width': 1920, 'height': 1080})
                page = context.new_page()
                began = time.monotonic()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/?token={token}')
                expect(page.locator('.motor')).to_have_count(29)
                page.wait_for_function("document.querySelector('#camera-image').naturalWidth === 640")
                title_card(page)
                chapters.append({'at_s': round(time.monotonic() - began, 2), 'title': '品牌片头'})
                page.wait_for_timeout(2600)
                page.evaluate("document.querySelector('#recording-title').style.opacity='0'")
                page.wait_for_timeout(700)
                page.evaluate("document.querySelector('#recording-title').remove()")
                chapters.append({'at_s': round(time.monotonic() - began, 2), 'title': '输入用户指令'})
                field = page.locator('#message-input')
                field.click()
                field.press_sequentially(args.instruction, delay=110)
                page.wait_for_timeout(800)
                page.locator('#send-button').click()
                expect(field).to_have_value('')
                expect(page.locator('#user-text')).to_have_text(args.instruction)
                expect(page.locator('.plan-step')).to_have_count(6)
                assert console.task['instruction'] == args.instruction
                print('录制开始：指令已通过真实消息接口提交到模拟运行时。', flush=True)
                for index, stage in enumerate(STAGES):
                    chapters.append({'at_s': round(time.monotonic() - began, 2), 'title': stage[1], 'tool': stage[3]})
                    for tick in range(28):
                        clock.value += .25
                        console.tick()
                        page.wait_for_timeout(250)
                        if tick == 8:
                            expect(page.locator('#tool-name')).to_have_text(stage[3])
                            expect(page.locator('#tool-status')).to_have_text('模拟调用中')
                        if tick == 21:
                            expect(page.locator('#tool-status')).to_have_text('模拟返回 ✓')
                            expect(page.locator('#tool-result')).to_contain_text('"simulated":true')
                            if index == 0:
                                page.screenshot(path=str(output.with_name('humanoid-plan.png')))
                            if index == 2:
                                page.screenshot(path=str(output.with_name('humanoid-tool-feedback.png')))
                    print(f'已录制 {index + 1}/6：{stage[1]} / {stage[3]}', flush=True)
                expect(page.locator('#task-status')).to_have_text('演示完成')
                expect(page.locator('#plan-count')).to_have_text('6 / 6')
                expect(page.locator('#agent-text')).to_contain_text('不代表真机搬运已完成')
                page.wait_for_timeout(2500)
                page.screenshot(path=str(output.with_name('humanoid-completed.png')))
                trace = console.export()
                assert trace['task']['simulated_verified'] and not trace['task']['verified']
                assert len([e for e in trace['events'] if e['kind'] == 'tool_result']) == 6
                assert all(j['level'] == 'normal' for j in trace['telemetry']['joints'])
                assert not errors, errors
                chapters.append({'at_s': round(time.monotonic() - began, 2), 'title': '模拟验收完成与片尾'})
                title_card(page, ending=True)
                page.wait_for_timeout(3500)
                video = page.video
                context.close()
                raw = output.with_suffix('.webm')
                video.save_as(str(raw))
                browser.close()
                output.with_suffix('.trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n')
            print('录制完成，正在导出 1080p MP4。', flush=True)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(raw),
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                            '-r', '30', '-movflags', '+faststart', '-an',
                            '-metadata', 'title=Humanoid Agent | SeekAct Robotics | Simulated interaction demo',
                            str(output)], check=True)
            metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                    'format=duration,size:stream=codec_name,width,height,r_frame_rate', '-of', 'json', str(output)], text=True))
            report = dict(instruction=args.instruction, chapters=chapters, page_errors=errors, video=metadata,
                          source='simulated', audio=False, physical_motion=False,
                          note='Browser interaction recording. Historic image; nominal synthetic temperatures; scripted tool replies.')
            output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'output': str(output), **report}, ensure_ascii=False, indent=2), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
            console.close()


if __name__ == '__main__':
    main()
