#!/usr/bin/env python3
"""Loopback-only, read-only viewer exposing exactly four approved video artifacts.

No directory serving, arbitrary paths, shell actions, external assets or uploads.
"""
import argparse
import html
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import urllib.parse


def span(range_header,size):
    if range_header is None:return 0,size-1,False
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header)
    if not match or not any(match.groups()):raise ValueError('Invalid range')
    first,last=match.groups()
    if not first:
        length=int(last)
        if length<=0:raise ValueError('Invalid suffix')
        return max(0,size-length),size-1,True
    start=int(first);end=min(size-1,int(last))if last else size-1
    if start>=size or end<start:raise ValueError('Unsatisfiable range')
    return start,end,True


def viewer(root,prefix):
    artifacts={};cards=[]
    rows=[('putting_shoes_on_rack','鞋子上架','安全阻塞停止，官方未成功','10744','80','32'),
          ('collecting_childrens_toys','收集玩具','一小时超时，官方未成功','15159','99','51')]
    for name,title,result,steps,decisions,switches in rows:
        videos=[]
        for role,label in [('third_person','第三人称'),('robot_cameras','机器人视角')]:
            path=root/name/'zero_shot_interaction_v1/sim/videos_annotated'/(role+'.mp4')
            if not path.is_file():raise FileNotFoundError(path)
            url=prefix+'/media/'+name+'-'+role+'.mp4';artifacts[url]=path
            videos.append('<figure><figcaption>'+label+'</figcaption><video controls playsinline preload="metadata"><source src="'+url+'" type="video/mp4"></video></figure>')
        cards.append('<section><h2>'+title+'</h2><p>'+result+' · '+steps+'步 · '+decisions+'段调度 · '+switches+'次AP/冠军切换</p><div class="views">'+''.join(videos)+'</div></section>')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AP / 冠军 · 实验视频</title><style>
body{font:16px system-ui,sans-serif;background:#0b1120;color:#e2e8f0;max-width:1440px;margin:0 auto;padding:28px}h1{margin-bottom:12px}p{line-height:1.7;color:#b9c6da}.legend{display:flex;gap:16px;flex-wrap:wrap}.ap,.champ{padding:10px 16px;border-radius:12px;background:#152136}.ap{color:#22d3ee}.champ{color:#fbbf24}section{background:#111c2e;border:1px solid #26364d;border-radius:18px;padding:20px;margin:28px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;min-width:0}figcaption{margin-bottom:10px;color:#cbd5e1}video{display:block;width:100%;background:#000;border-radius:10px}small{color:#94a3b8}@media(max-width:900px){.views{grid-template-columns:1fr}body{padding:16px}}</style></head><body><h1>主动感知 AP / 冠军 VLA · 全过程视频</h1><div class="legend"><span class="ap">⌕ 青色放大镜：AP</span><span class="champ">★ 金色星形：CHAMPION</span></div><p>视频顶部显示当前物理动作来源、步数和调用编号。四段视频共51806帧已与动作及ACK逐帧核对。视频保留全部物理过程，不包含仿真冻结时的GPT或模型等待。</p>'''+''.join(cards)+'''<small>本页只在本机提供这四段带标识视频。原视频与验收记录保持不变。</small></body></html>'''
    return page.encode('utf-8'),artifacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18886);a=p.parse_args();root=a.output.resolve()
    prefix='/results/'+secrets.token_hex(8)
    page,artifacts=viewer(root,prefix)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_HEAD(self):self.respond(False)
        def do_GET(self):self.respond(True)
        def do_POST(self):self.send_error(405)
        def respond(self,body):
            if self.headers.get('Host') not in ('127.0.0.1:'+str(a.port),'localhost:'+str(a.port)):
                self.send_error(403);return
            path=urllib.parse.urlsplit(self.path).path
            if path in (prefix,prefix+'/'):
                self.send_response(200);self.headers_common('text/html; charset=utf-8',len(page));self.end_headers()
                if body:self.wfile.write(page)
                return
            source=artifacts.get(path)
            if source is None:self.send_error(404);return
            size=source.stat().st_size
            try:start,end,ranged=span(self.headers.get('Range'),size)
            except ValueError:
                self.send_response(416);self.send_header('Content-Range','bytes */'+str(size));self.send_header('Content-Length','0');self.end_headers();return
            self.send_response(206 if ranged else 200)
            self.headers_common('video/mp4',end-start+1);self.send_header('Accept-Ranges','bytes')
            if ranged:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not body:return
            try:
                with source.open('rb')as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        data=stream.read(min(262144,remaining))
                        if not data:break
                        self.wfile.write(data);remaining-=len(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def headers_common(self,mime,size):
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(size));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'none'; media-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'")
            self.send_header('Referrer-Policy','no-referrer')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    receipt={'url':'http://127.0.0.1:'+str(a.port)+prefix+'/','bind':'127.0.0.1','port':a.port,'scope':'generated viewer and exactly four selected MP4s','files':{k:str(v)for k,v in artifacts.items()},'directory_listing':False,'uploads':False}
    (root/'video_viewer_server.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False),flush=True)
    server.serve_forever()

if __name__=='__main__':main()
