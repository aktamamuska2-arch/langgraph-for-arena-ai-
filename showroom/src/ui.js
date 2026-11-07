/**
 * ui.js — semua manipulasi DOM: set list, bars Motion QA, indikator ketukan.
 */

export const QA_BARS = [
  { key: 'activity', label: 'aktivitas' },
  { key: 'amplitude', label: 'amplitudo' },
  { key: 'smoothness', label: 'kehalusan' },
  { key: 'footwork', label: 'pijakan' },
  { key: 'balance', label: 'balance' },
  { key: 'loop', label: 'loop' },
  { key: 'rhythm', label: 'ritme' },
];

export function buildPlaylist(container, dances, onSelect) {
  container.innerHTML = '';
  dances.forEach((d, i) => {
    const li = document.createElement('li');
    li.className = 'song';
    li.dataset.id = d.id;
    li.style.color = d.accentColor;
    li.innerHTML = `
      <span class="song__bar"></span>
      <span>
        <span class="song__name">${d.name}</span>
        <span class="song__meta">${d.tagline} · ${d.beats} ketukan</span>
      </span>
      <span class="song__bpm">${d.bpm}</span>
    `;
    li.addEventListener('click', () => onSelect(i));
    container.appendChild(li);
  });
}

export function buildClipChips(container, clips, onSelect) {
  container.innerHTML = '';
  clips.forEach((c) => {
    const b = document.createElement('button');
    b.className = 'chip chip--btn';
    b.dataset.clip = c.id;
    b.textContent = c.name;
    b.addEventListener('click', () => onSelect(c));
    container.appendChild(b);
  });
}

export function buildBars(container) {
  container.innerHTML = '';
  const refs = {};
  for (const bar of QA_BARS) {
    const row = document.createElement('div');
    row.className = 'bar';
    row.innerHTML = `
      <span>${bar.label}</span>
      <span class="bar__track"><span class="bar__fill"></span></span>
      <span class="bar__val">–</span>
    `;
    container.appendChild(row);
    refs[bar.key] = {
      fill: row.querySelector('.bar__fill'),
      val: row.querySelector('.bar__val'),
    };
  }
  return refs;
}

export function buildBeats(container, count = 16) {
  container.innerHTML = '';
  const dots = [];
  for (let i = 0; i < count; i++) {
    const d = document.createElement('span');
    d.className = 'beat' + (i % 4 === 0 ? ' is-bar' : '');
    container.appendChild(d);
    dots.push(d);
  }
  return dots;
}

export function updateBeats(dots, beatPos, playing) {
  const beat = Math.floor(beatPos || 0);
  const active = beat % dots.length;
  dots.forEach((d, i) => d.classList.toggle('is-on', playing && i === active));
}

export function renderQA(refs, summary, els) {
  if (!summary) return;
  const { scoreValue, scoreRing, danceName, poseName, notes } = els;
  scoreValue.textContent = summary.overall;
  scoreRing.style.setProperty('--p', summary.overall);
  danceName.textContent = summary.danceId;
  poseName.textContent = summary.beatSync !== undefined ? '' : '';
  for (const bar of QA_BARS) {
    const v = summary.scores[bar.key] ?? 0;
    const ref = refs[bar.key];
    if (!ref) continue;
    ref.fill.style.width = `${v}%`;
    ref.val.textContent = v;
  }
  notes.innerHTML = `
    energi <b>${summary.energy.toFixed(2)}</b> m/s ·
    slip <b>${summary.footSlip.toFixed(2)}</b> m/s ·
    lantai <b>${summary.floorMin >= -0.005 ? 'aman' : 'tembus'}</b><br />
    aksen <b>${Math.round(summary.beatSync * 100)}%</b> ketukan ·
    offset <b>${summary.offsetMs > 0 ? '+' : ''}${summary.offsetMs ?? 0}</b> ms ·
    loop <b>${(summary.loopSeam * 100).toFixed(1)}</b> cm
  `;
}
