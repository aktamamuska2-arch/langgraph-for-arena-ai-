/* SIGMA console — vanilla JS, no build step (keeps the artifact budget tiny).
   Talks to the FastAPI surface only; all orchestration lives in the LangGraph
   server, never in the browser. */
'use strict';

const state = { sessionId: null, es: null, seq: 0, checkpoints: [] };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ helpers */

function fmt(n, digits = 2) {
  return n === null || n === undefined ? '—' : Number(n).toFixed(digits);
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function showTab(name) {
  $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  $$('.tabpanel').forEach((panel) =>
    panel.classList.toggle('active', panel.id === `tab-${name}`));
}

function pushTimeline(node, kind, message) {
  const list = $('#timeline');
  const empty = list.querySelector('.empty');
  if (empty) empty.remove();
  const item = el('li', kind === 'interrupt' ? 'interrupt' : kind === 'error' ? 'error' : '');
  item.appendChild(el('span', 'idx', String(++state.seq).padStart(3, '0')));
  const body = el('div');
  body.appendChild(el('div', 'node', node));
  body.appendChild(el('div', 'msg', message));
  item.appendChild(body);
  list.appendChild(item);
  list.scrollTop = list.scrollHeight;
}

/* -------------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const health = await api('/api/health');
    const budget = health.budget_mb || 125;
    const total = health.artifact_mb || 0;
    const badge = $('#size-badge');
    badge.textContent = `artefak ${fmt(total)} MB / ${budget} MB`;
    badge.classList.add(total < budget ? 'good' : 'bad');
    $('#health-badge').textContent = `langgraph ${health.langgraph}`;
    $('#health-badge').classList.add('good');
    $('#provider-badge').textContent = `provider: ${health.provider}`;
    $('#footer-db').textContent = `checkpoint db ${(health.checkpoint_db_bytes / 1024).toFixed(0)} KB`;
  } catch (error) {
    $('#health-badge').textContent = 'offline';
    $('#health-badge').classList.add('bad');
  }
}

/* -------------------------------------------------------------------- running */

function closeStream() {
  if (state.es) { state.es.close(); state.es = null; }
}

function runAnalysis(event) {
  event.preventDefault();
  closeStream();
  const question = $('#question').value.trim();
  if (question.length < 3) return;

  $('#timeline').innerHTML = '';
  state.seq = 0;
  $('#gate-card').hidden = true;
  $('#brief-body').className = 'empty';
  $('#brief-body').textContent = 'Menjalankan graf…';
  $('#run-btn').disabled = true;

  const params = new URLSearchParams({
    question,
    seed: $('#seed').value || '7',
    portfolio_id: 'sigma-core',
    acceptance_threshold: $('#threshold').value || '0.78',
    hitl: $('#hitl').checked ? 'true' : 'false',
  });

  const source = new EventSource(`/api/sessions/stream?${params}`);
  state.es = source;

  source.addEventListener('start', (message) => {
    const data = JSON.parse(message.data);
    state.sessionId = data.session_id;
    pushTimeline('session', 'start', `session ${data.session_id}`);
  });

  source.addEventListener('node', (message) => {
    const data = JSON.parse(message.data);
    const update = data.update || {};
    const parts = Object.keys(update)
      .filter((key) => key !== 'node')
      .map((key) => `${key}=${JSON.stringify(update[key])}`)
      .slice(0, 3);
    pushTimeline(data.node, 'node', parts.join(' ') || '—');
  });

  source.addEventListener('interrupt', (message) => {
    const data = JSON.parse(message.data);
    pushTimeline('human_gate', 'interrupt', 'menunggu keputusan manusia');
    // `data.interrupts` is a list of {id, value} pairs; renderGate takes values.
    renderGate({ interrupts: (data.interrupts || []).map((item) => item.value),
                 state: data.state });
    source.close();
    state.es = null;
    $('#run-btn').disabled = false;
  });

  source.addEventListener('final', (message) => {
    const data = JSON.parse(message.data);
    pushTimeline('finalizer', 'final', 'selesai');
    renderResult(data.state);
    source.close();
    state.es = null;
    $('#run-btn').disabled = false;
  });

  source.addEventListener('error', (message) => {
    pushTimeline('stream', 'error', message.data || 'stream error');
    source.close();
    state.es = null;
    $('#run-btn').disabled = false;
  });
}

/* ---------------------------------------------------------------------- gate */

function renderGate(data) {
  const payload = ((data.interrupts || [])[0]) || {};
  const card = $('#gate-card');
  card.hidden = false;
  $('#gate-score').textContent = `skor ${fmt(payload.score, 3)}`;
  const body = $('#gate-body');
  body.innerHTML = '';
  if (payload.summary) body.appendChild(el('div', 'summary', payload.summary));
  if (payload.recommendations && payload.recommendations.length) {
    const list = el('ul');
    payload.recommendations.forEach((line) => list.appendChild(el('li', null, line)));
    body.appendChild(list);
  }
  if (payload.open_gaps && payload.open_gaps.length) {
    body.appendChild(el('div', 'gaps',
      `${payload.open_gaps.length} gap terbuka: ` +
      payload.open_gaps.map((gap) => gap.kind).join(', ')));
  }
  showTab('brief');
  renderResult(data.state);
}

async function decide(decision) {
  if (!state.sessionId) return;
  const note = $('#gate-note').value.trim();
  $('#gate-note').value = '';
  $('#gate-card').hidden = true;
  pushTimeline('human', 'decision', `${decision}: ${note || '(tanpa catatan)'}`);
  try {
    const view = await api(`/api/sessions/${state.sessionId}/resume`, {
      method: 'POST',
      body: JSON.stringify({ decision, note }),
    });
    renderResult(view);
    if (view.interrupted) {
      renderGate({ interrupts: view.interrupts.map((i) => i.interrupts[0].value),
                   state: view });
    } else {
      pushTimeline('finalizer', 'final', `status ${view.status}`);
    }
  } catch (error) {
    pushTimeline('resume', 'error', String(error.message || error));
  }
}

/* -------------------------------------------------------------------- render */

function renderResult(view) {
  if (!view) return;
  state.lastView = view;
  $('#state-body').textContent = JSON.stringify(view, null, 2);
  renderBrief(view);
  renderRubric(view);
  loadTimeline(state.sessionId);
}

function renderBrief(view) {
  const brief = view.brief || {};
  const body = $('#brief-body');
  body.innerHTML = '';
  body.className = '';

  const headline = brief.headline || 'Brief belum tersedia';
  body.appendChild(el('h3', 'headline', headline));
  body.appendChild(el('p', 'summary', brief.summary || ''));

  if (brief.recommendations && brief.recommendations.length) {
    const list = el('ol', 'recs');
    brief.recommendations.forEach((line) => list.appendChild(el('li', null, line)));
    body.appendChild(list);
  }

  const findings = (view.findings || []).filter((f) => !f.superseded);
  if (findings.length) {
    body.appendChild(el('h3', 'headline', 'Temuan terdukung bukti'));
    findings.slice(0, 8).forEach((finding) => {
      const box = el('div', 'finding');
      box.appendChild(el('div', 'claim',
        `${finding.claim}${finding.metric !== null ? ` [${fmt(finding.metric)} ${finding.unit || ''}]` : ''}`));
      box.appendChild(el('div', 'prov',
        `${finding.evidence_ids.length} bukti · confidence ${fmt(finding.confidence)}`));
      body.appendChild(box);
    });
  }

  if (brief.caveats && brief.caveats.length) {
    const list = el('ul', 'caveats');
    brief.caveats.forEach((line) => list.appendChild(el('li', null, line)));
    body.appendChild(list);
  }
}

function renderRubric(view) {
  const body = $('#rubric-body');
  body.innerHTML = '';
  const evaluation = view.evaluation;
  if (!evaluation) { body.className = 'empty'; body.textContent = 'Belum ada evaluasi.'; return; }
  body.className = '';

  const scores = evaluation.scores || {};
  Object.entries(scores).forEach(([name, value]) => {
    const row = el('div', 'metric-row');
    row.appendChild(el('span', 'name', name));
    const bar = el('div', 'bar');
    const fill = el('span');
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el('span', 'val', fmt(value)));
    body.appendChild(row);
  });

  const history = view.evaluations || [];
  if (history.length > 1) {
    body.appendChild(el('p', 'hint',
      `Siklus koreksi-diri: ${history.map((e) => fmt(e.score, 3)).join(' → ')} ` +
      `(revisi ke-${evaluation.revision})`));
  }
  const gaps = evaluation.gaps || [];
  if (gaps.length) {
    const list = el('ul', 'caveats');
    gaps.forEach((gap) => list.appendChild(el('li', null, `[${gap.kind}] ${gap.detail}`)));
    body.appendChild(list);
  }
}

/* ----------------------------------------------------------------- time travel */

async function loadTimeline(sessionId) {
  if (!sessionId) return;
  try {
    const data = await api(`/api/sessions/${sessionId}/timeline`);
    state.checkpoints = data.checkpoints || [];
    const body = $('#timeline-body');
    body.innerHTML = '';
    if (!state.checkpoints.length) { body.className = 'empty'; body.textContent = 'Tidak ada checkpoint.'; return; }
    body.className = '';
    state.checkpoints.slice(0, 40).forEach((cp) => {
      const row = el('div', 'cp');
      const left = el('div');
      left.appendChild(el('div', null, `#${cp.step} · ${(cp.next || []).join(',') || 'END'}`));
      left.appendChild(el('div', 'cp-meta',
        `checkpoint ${String(cp.checkpoint_id).slice(0, 8)} · rev ${cp.revision} · ` +
        `skor ${fmt(cp.score, 3)} · bukti ${cp.evidence}`));
      row.appendChild(left);
      const fork = el('button', 'ghost', 'fork →');
      if (cp.forkable === false) {
        fork.disabled = true;
        fork.title = 'checkpoint sebelum permintaan diproses';
      } else {
        fork.onclick = () => forkFrom(cp.checkpoint_id);
      }
      row.appendChild(fork);
      body.appendChild(row);
    });
  } catch (error) {
    $('#timeline-body').textContent = String(error.message || error);
  }
}

async function forkFrom(checkpointId) {
  if (!state.sessionId) return;
  pushTimeline('fork', 'fork', `fork dari ${String(checkpointId).slice(0, 8)}`);
  try {
    const view = await api(`/api/sessions/${state.sessionId}/fork`, {
      method: 'POST',
      body: JSON.stringify({ checkpoint_id: checkpointId }),
    });
    state.sessionId = view.session_id;
    renderResult(view);
    pushTimeline('fork', 'fork', `cabang baru ${view.session_id}`);
  } catch (error) {
    pushTimeline('fork', 'error', String(error.message || error));
  }
}

/* -------------------------------------------------------------------- sessions */

async function loadSessions() {
  try {
    const data = await api('/api/sessions?limit=20');
    const body = $('#sessions-body');
    body.innerHTML = '';
    if (!data.sessions.length) { body.className = 'empty'; body.textContent = 'Belum ada sesi.'; return; }
    body.className = '';
    data.sessions.forEach((session) => {
      const row = el('div', 'session-row');
      const left = el('div');
      left.appendChild(el('div', 'sid', `${session.session_id.slice(0, 12)} · ${session.status}`));
      left.appendChild(el('div', 'q', session.question || '—'));
      row.appendChild(left);
      const right = el('div');
      right.appendChild(el('span', 'pill', `skor ${fmt(session.score, 3)}`));
      row.appendChild(right);
      row.onclick = async () => {
        state.sessionId = session.session_id;
        const view = await api(`/api/sessions/${session.session_id}`);
        renderResult(view);
        showTab('brief');
      };
      body.appendChild(row);
    });
  } catch (error) {
    $('#sessions-body').textContent = String(error.message || error);
  }
}

/* ----------------------------------------------------------------------- wire */

$('#run-form').addEventListener('submit', runAnalysis);
$('#reset-btn').addEventListener('click', () => {
  closeStream();
  $('#timeline').innerHTML = '<li class="empty">Belum ada eksekusi.</li>';
  $('#gate-card').hidden = true;
  state.seq = 0;
  state.sessionId = null;
});
$$('[data-decision]').forEach((button) => {
  button.addEventListener('click', () => decide(button.dataset.decision));
});
$$('.tab').forEach((tab) => tab.addEventListener('click', () => showTab(tab.dataset.tab)));
$('#refresh-sessions').addEventListener('click', loadSessions);

loadHealth();
loadSessions();
