'use strict';

// Speech is an explicit browser action. It only fills the draft; sending is separate.
(() => {
  const input = $('message-input');
  const form = $('message-form');
  const voiceButton = $('voice-button');
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let sending = false, composing = false, recognition = null, voiceTimeout = null, stopTimeout = null;
  let notice = '', noticeKind = '', pendingRequest = null;

  function status(text, kind = '') { notice = text; noticeKind = kind; update(); }
  function update() {
    const chat = ui.snapshot?.capabilities?.chat === true;
    $('send-button').disabled = !ui.connected || !chat || sending || !!recognition || !input.value.trim() || input.value.length > 1000;
    $('send-button').setAttribute('aria-label', sending ? '正在发送' : '发送消息');
    input.readOnly = sending || !!recognition;
    $('message-count').textContent = `${input.value.length} / 1000`;
    const auto = ui.snapshot?.capabilities?.agent_autowake;
    const running = ui.snapshot?.operator_inbox?.dispatcher?.phase === 'running';
    $('composer-mode').textContent = ui.snapshot?.mode === 'demo' ? '演示会话 · 不执行机器人动作' : chat ? (auto ? (running ? 'Agent 执行中 · 可发送补充指令' : 'Agent 待命 · 发送任务自动开始') : ui.snapshot?.capabilities?.agent_connected ? 'Agent 已连接 · 消息将交给 Agent 处理' : 'Agent 离线 · 消息仅暂存') : '实机观测 · 发送服务待接入';
    voiceButton.disabled = sending;
    voiceButton.setAttribute('aria-disabled', String(sending));
    voiceButton.setAttribute('aria-pressed', String(!!recognition));
    voiceButton.setAttribute('aria-label', recognition ? '停止语音输入' : '开始语音输入');
    voiceButton.title = recognition ? '点击结束语音输入' : !Recognition ? '当前浏览器不支持语音识别，点击查看提示' : '语音转文字（由浏览器提供，可能使用在线服务）';
    $('composer-status').textContent = notice || (!ui.connected ? '连接中断 · 草稿已保留，恢复连接后可发送' : !chat ? '当前为只读观测，可编辑草稿；消息发送服务尚未接入' : 'Enter 发送 · Shift + Enter 换行 · 语音识别后确认发送');
    $('composer-status').className = 'composer-status' + (noticeKind ? ' ' + noticeKind : '');
    form.setAttribute('aria-busy', String(sending));
  }
  input.addEventListener('input', () => { if (!recognition) { notice = ''; noticeKind = ''; } update(); });
  input.addEventListener('compositionstart', () => { composing = true; });
  input.addEventListener('compositionend', () => { composing = false; update(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !composing && event.keyCode !== 229) {
      event.preventDefault();
      if (!$('send-button').disabled) form.requestSubmit();
    }
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (sending || recognition || !ui.connected || !ui.snapshot?.capabilities?.chat || !input.value.trim()) return;
    if (input.value.length > 1000) { status('消息不能超过 1000 字，请缩短后发送。', 'error'); return; }
    const draft = input.value;
    if (!pendingRequest || pendingRequest.text !== draft.trim()) pendingRequest = {text: draft.trim(), request_id: crypto.randomUUID()};
    sending = true; status('正在发送…');
    try {
      const response = await fetch('/api/v1/messages', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(pendingRequest), signal: AbortSignal.timeout(10000)});
      const result = await response.json();
      if (!response.ok || !result.accepted) throw new Error(result.error || `发送失败 (${response.status})`);
      input.value = '';
      pendingRequest = null;
      status(result.action === 'demo_task_started' ? '已开始交互演示 · 可观察任务拆解、工具调用与模拟反馈' : result.source === 'demo' ? '已发送至演示会话 · 真实 Agent 尚未接入' : result.reply || '消息已发送');
      await refresh();
    } catch (error) {
      status(error.name === 'TimeoutError' || error instanceof TypeError ? '暂未确认发送结果，草稿已保留；请检查会话后再重试。' : error.message || '发送失败，草稿已保留。', 'error');
    } finally { sending = false; update(); input.focus(); }
  });

  function release(session) {
    if (recognition !== session) return;
    recognition = null;
    clearTimeout(voiceTimeout); clearTimeout(stopTimeout);
    update(); input.focus();
  }
  function abortVoice(message) {
    const session = recognition;
    if (!session) return;
    release(session); // Ignore late events after abort (including permission requests).
    try { session.abort(); } catch { /* Already ended. */ }
    status(message);
  }
  function stopVoice() {
    const session = recognition;
    if (!session) return;
    status('正在结束识别…', 'listening');
    try { session.stop(); } catch { abortVoice('语音已停止，草稿已保留。'); return; }
    clearTimeout(stopTimeout);
    stopTimeout = setTimeout(() => abortVoice('语音已停止，请检查输入框中的文字。'), 2000);
  }
  voiceButton.addEventListener('click', () => {
    if (recognition) { stopVoice(); return; }
    if (!Recognition) { status('当前浏览器不支持语音识别，请使用文本输入。', 'error'); return; }
    if (!window.isSecureContext) { status('语音输入需要安全连接或本机地址，请使用文本输入。', 'error'); return; }
    if (sending) return;
    const base = input.value;
    let session;
    try { session = new Recognition(); } catch { status('无法启动浏览器语音识别，请使用文本输入。', 'error'); return; }
    recognition = session;
    session.lang = 'zh-CN'; session.continuous = false; session.interimResults = true; session.maxAlternatives = 1;
    let failed = false, heard = false, truncated = false;
    session.onstart = () => { if (recognition === session) status('正在聆听 · 点击麦克风结束，文字确认后再发送', 'listening'); };
    session.onresult = event => {
      if (recognition !== session) return;
      // The result list is cumulative and includes replaceable interim results.
      let transcript = '';
      for (let index = 0; index < event.results.length; index++) transcript += event.results[index][0]?.transcript || '';
      if (!transcript) return;
      heard = true;
      const draft = base + (base && !/\s$/.test(base) ? ' ' : '') + transcript;
      truncated = draft.length > 1000;
      input.value = draft.slice(0, 1000);
      update();
    };
    session.onerror = event => {
      if (recognition !== session) return;
      failed = true;
      const messages = {'not-allowed': '麦克风权限未获授权，请允许麦克风访问或使用文本输入。', 'service-not-allowed': '浏览器未允许语音识别服务，请使用文本输入。', 'audio-capture': '未检测到可用麦克风，请检查设备。', 'no-speech': '没有听到语音，请重试或使用文本输入。', network: '语音识别服务连接失败，请使用文本输入。', 'language-not-supported': '当前识别服务不支持中文，请使用文本输入。', aborted: '语音识别已停止，草稿已保留。'};
      release(session);
      try { session.abort(); } catch { /* Nothing left to stop. */ }
      status(messages[event.error] || '语音识别失败，草稿已保留。', 'error');
    };
    session.onend = () => {
      if (recognition !== session) return;
      release(session);
      if (!failed) status(truncated ? '识别内容超过 1000 字，已截取，请编辑后发送。' : heard ? '语音已转为文字，请检查后点击发送。' : '没有识别到文字，可以重新点击麦克风。', truncated ? 'error' : '');
    };
    status('正在开启浏览器语音识别（可能使用在线服务）…', 'listening');
    try { session.start(); voiceTimeout = setTimeout(stopVoice, 30000); }
    catch { release(session); status('无法启动语音识别，请检查麦克风权限或使用文本输入。', 'error'); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) abortVoice('语音已停止，草稿已保留。'); });
  window.addEventListener('pagehide', () => abortVoice('语音已停止。'));
  document.addEventListener('console-state', update);
  update();
})();
