// ============================================================
// AI 조련사 — 과일 감별 AI 훈련소
// 최근접 이웃(1-NN) 분류기를 사용자가 직접 라벨링해 학습시킨다.
// ============================================================

const HOMEPAGE = 'https://everythingmine.com';
const NOISE = 0.6; // 엉터리 데이터의 라벨 오류 비율
const LABELS = {
    apple: { name: '사과', emoji: '🍎', color: '#e11d48', region: 'rgba(225, 29, 72, 0.16)' },
    tangerine: { name: '귤', emoji: '🍊', color: '#f59e0b', region: 'rgba(245, 158, 11, 0.20)' }
};

// 과일 종류별 특징 분포 (크기 cm, 색상값 0=빨강 ~ 120=초록)
const KINDS = {
    red:   { cls: 'apple',     size: [8.2, 0.9], hue: [8, 7],  hueRange: [0, 30] },
    green: { cls: 'apple',     size: [8.2, 0.9], hue: [95, 8], hueRange: [75, 115] },
    tang:  { cls: 'tangerine', size: [6.4, 0.8], hue: [32, 6], hueRange: [16, 50] }
};
const SIZE_MIN = 4, SIZE_MAX = 11, HUE_MAX = 120;

// ------------------------------------------------------------
// 데이터 & 모델
// ------------------------------------------------------------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function gauss(mean, sd) {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeFruit(kind) {
    const k = KINDS[kind];
    return {
        kind,
        cls: k.cls,
        size: +clamp(gauss(...k.size), 4.5, 10.8).toFixed(1),
        hue: Math.round(clamp(gauss(...k.hue), ...k.hueRange))
    };
}

function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// counts: { red: 5, tang: 5 }
function makeSet(counts) {
    const out = [];
    for (const [kind, n] of Object.entries(counts)) {
        for (let i = 0; i < n; i++) out.push(makeFruit(kind));
    }
    return shuffle(out);
}

const feat = f => [(f.size - SIZE_MIN) / (SIZE_MAX - SIZE_MIN), f.hue / HUE_MAX];

// 가장 비슷한(가까운) 학습 예시의 라벨을 그대로 따른다
function predict(train, f) {
    if (!train.length) return null;
    const [x, y] = feat(f);
    let best = null, bestD = Infinity;
    for (const t of train) {
        const [tx, ty] = feat(t);
        const d = (tx - x) ** 2 + (ty - y) ** 2;
        if (d < bestD) { bestD = d; best = t; }
    }
    return { label: best.label };
}

function runTest(train, set) {
    const results = set.map(f => {
        const p = predict(train, f);
        return { f, pred: p && p.label, ok: p && p.label === f.cls };
    });
    const correct = results.filter(r => r.ok).length;
    return { results, acc: Math.round(correct / set.length * 100), correct };
}

const labeled = (fruits, noise = 0) => fruits.map(f => {
    const flip = Math.random() < noise;
    return { ...f, label: flip ? (f.cls === 'apple' ? 'tangerine' : 'apple') : f.cls };
});

// ------------------------------------------------------------
// 그리기
// ------------------------------------------------------------
function fruitSVG(f, px) {
    const color = `hsl(${f.hue}, 85%, ${f.kind === 'green' ? 45 : 52}%)`;
    const r = 18 + (f.size - SIZE_MIN) / (SIZE_MAX - SIZE_MIN) * 26;
    const body = f.cls === 'apple'
        ? `<path d="M50 ${52 - r * 0.78} C ${50 + r * 0.5} ${52 - r * 1.05}, ${50 + r * 1.1} ${52 - r * 0.6}, ${50 + r} 52
             C ${50 + r} ${52 + r * 0.8}, ${50 + r * 0.4} ${52 + r}, 50 ${52 + r * 0.9}
             C ${50 - r * 0.4} ${52 + r}, ${50 - r} ${52 + r * 0.8}, ${50 - r} 52
             C ${50 - r * 1.1} ${52 - r * 0.6}, ${50 - r * 0.5} ${52 - r * 1.05}, 50 ${52 - r * 0.78} Z"
             fill="${color}" stroke="#16161a" stroke-width="2.5"/>
           <path d="M50 ${52 - r * 0.78} q 2 -10 6 -14" stroke="#5b3a1a" stroke-width="3" fill="none" stroke-linecap="round"/>`
        : `<ellipse cx="50" cy="54" rx="${r}" ry="${r * 0.86}" fill="${color}" stroke="#16161a" stroke-width="2.5"/>
           <circle cx="${50 - r * 0.35}" cy="${54 - r * 0.2}" r="1.6" fill="#16161a" opacity=".25"/>
           <circle cx="${50 + r * 0.3}" cy="${54 + r * 0.25}" r="1.6" fill="#16161a" opacity=".25"/>
           <circle cx="50" cy="${54 - r * 0.86 + 3}" r="2.2" fill="#3f6212"/>`;
    const leafY = f.cls === 'apple' ? 52 - r * 0.78 - 8 : 54 - r * 0.86;
    return `<svg width="${px}" height="${px}" viewBox="0 0 100 100" aria-hidden="true">
        ${body}
        <path d="M52 ${leafY} q 10 -8 18 -2 q -8 8 -18 2 Z" fill="#4d7c0f" stroke="#16161a" stroke-width="1.5"/>
    </svg>`;
}

function drawMap(canvas, train, focus) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const L = 34, B = 26, T = 10, R = 10;
    const pw = w - L - R, ph = h - T - B;
    const px = f => L + (f.size - SIZE_MIN) / (SIZE_MAX - SIZE_MIN) * pw;
    const py = f => T + ph - f.hue / HUE_MAX * ph;

    // AI의 판단 영역
    if (train.length) {
        const cell = 8;
        for (let x = 0; x < pw; x += cell) {
            for (let y = 0; y < ph; y += cell) {
                const probe = {
                    size: SIZE_MIN + (x + cell / 2) / pw * (SIZE_MAX - SIZE_MIN),
                    hue: (1 - (y + cell / 2) / ph) * HUE_MAX
                };
                ctx.fillStyle = LABELS[predict(train, probe).label].region;
                ctx.fillRect(L + x, T + y, cell, cell);
            }
        }
    } else {
        ctx.fillStyle = '#9a9aa3';
        ctx.font = '500 14px "Noto Sans KR", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('아직 아무것도 배우지 않았어요', L + pw / 2, T + ph / 2);
    }

    // 색상 축 그라데이션
    const grad = ctx.createLinearGradient(0, T + ph, 0, T);
    grad.addColorStop(0, 'hsl(0, 85%, 52%)');
    grad.addColorStop(0.3, 'hsl(36, 90%, 52%)');
    grad.addColorStop(0.6, 'hsl(72, 80%, 45%)');
    grad.addColorStop(1, 'hsl(120, 70%, 40%)');
    ctx.fillStyle = grad;
    ctx.fillRect(L - 12, T, 7, ph);

    // 축
    ctx.strokeStyle = '#16161a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(L, T); ctx.lineTo(L, T + ph); ctx.lineTo(L + pw, T + ph);
    ctx.stroke();

    ctx.fillStyle = '#5c5c66';
    ctx.font = '500 11px "Noto Sans KR", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('작다', L + 2, T + ph + 16);
    ctx.textAlign = 'right';
    ctx.fillText('크다 → 크기', L + pw, T + ph + 16);
    ctx.save();
    ctx.translate(L - 18, T + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('빨강 ← 색깔 → 초록', 0, 0);
    ctx.restore();

    // 학습 데이터 점
    train.forEach(t => {
        ctx.beginPath();
        ctx.arc(px(t), py(t), 5.5, 0, Math.PI * 2);
        ctx.fillStyle = LABELS[t.label].color;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#16161a';
        ctx.stroke();
    });

    // 지금 보고 있는 과일
    if (focus) {
        ctx.beginPath();
        ctx.setLineDash([4, 3]);
        ctx.arc(px(focus), py(focus), 12, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#3b82f6';
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#3b82f6';
        ctx.font = '900 13px "Noto Sans KR", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('?', px(focus), py(focus) + 4.5);
    }
}

// ------------------------------------------------------------
// 화면 조각
// ------------------------------------------------------------
const $app = document.getElementById('app');
const $chapter = document.getElementById('chapter');
let currentMap = null; // { train, focus }

function mapHTML(title, desc) {
    return `<div class="map-wrap card">
        <h3>${title}</h3>
        <p>${desc}</p>
        <canvas class="map" id="map"></canvas>
        <div class="legend">
            <span><i style="background:${LABELS.apple.color}"></i>사과라고 배운 것</span>
            <span><i style="background:${LABELS.tangerine.color}"></i>귤이라고 배운 것</span>
            <span>배경색 = AI가 그 자리를 무엇으로 판단하는지</span>
        </div>
    </div>`;
}

function render(html, map) {
    $app.innerHTML = html;
    currentMap = map || null;
    paintMap();
    window.scrollTo({ top: 0 });
}

function paintMap() {
    const c = document.getElementById('map');
    if (c && currentMap) drawMap(c, currentMap.train, currentMap.focus);
}
window.addEventListener('resize', paintMap);

function on(id, fn) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', fn);
}

function resultsHTML(test) {
    return `<div class="results">${test.results.map(r => `
        <div class="res ${r.ok ? 'ok' : 'bad'}" title="정답: ${LABELS[r.f.cls].name} / AI: ${LABELS[r.pred].name}">
            ${fruitSVG(r.f, 40)}<span class="mark">${r.ok ? '✓' : '✗'}</span>
        </div>`).join('')}</div>`;
}

// ------------------------------------------------------------
// 공통: 라벨링 화면
// ------------------------------------------------------------
function labelingScreen({ chapter, title, desc, queue, train, onDone }) {
    let i = 0;

    function step() {
        if (i >= queue.length) return onDone();
        const f = queue[i];
        const p = predict(train, f);
        const guess = p
            ? `🤖 AI의 추측: <b>${LABELS[p.label].emoji} ${LABELS[p.label].name}</b> <span style="color:var(--muted)">(가장 비슷한 예시를 따라감)</span>`
            : '🤖 AI의 추측: <b>모르겠어요</b> <span style="color:var(--muted)">(아직 본 게 없음)</span>';

        render(`
            <p class="eyebrow">${chapter}</p>
            <h2>${title}</h2>
            <p class="lead">${desc}</p>
            <div class="play">
                <div class="card">
                    <div class="progress">${queue.map((_, j) => `<span class="${j < i ? 'done' : ''}"></span>`).join('')}</div>
                    <div class="fruit-stage">${fruitSVG(f, 150)}</div>
                    <p class="sees-title">AI가 보는 것은 이 숫자 두 개뿐</p>
                    <div class="sees">
                        <div>크기<b>${f.size}cm</b></div>
                        <div>색깔값<b>${f.hue}</b></div>
                    </div>
                    <p class="guess">${guess}</p>
                    <div class="answer">
                        <button class="btn apple" id="a-apple">🍎 사과</button>
                        <button class="btn tang" id="a-tang">🍊 귤</button>
                    </div>
                </div>
                ${mapHTML('AI의 머릿속', '파란 점선이 지금 과일의 위치예요. 가르칠 때마다 판단 영역이 바뀝니다.')}
            </div>`, { train, focus: f });

        const answer = label => { train.push({ ...f, label }); i++; step(); };
        on('a-apple', () => answer('apple'));
        on('a-tang', () => answer('tangerine'));
    }
    step();
}

// ------------------------------------------------------------
// 게임 흐름
// ------------------------------------------------------------
const G = {};

function intro() {
    $chapter.textContent = '';
    render(`
        <div class="stack" style="max-width:640px">
            <p class="eyebrow">AI 원리 체험 게임 · 약 5분</p>
            <h1>AI 조련사:<br>과일 감별 AI 훈련소</h1>
            <p class="lead">여기 사과와 귤을 구분하지 못하는 아기 AI가 있습니다. 당신이 직접 예시를 보여주며 가르쳐 보세요. AI가 어떻게 배우고, 왜 틀리는지 직접 보게 됩니다.</p>
            <div class="card">
                <ul class="lessons">
                    <li><div><b>1장. 가르치기</b><br><span style="color:var(--muted)">과일 10개를 보여주고 시험을 봅니다.</span></div></li>
                    <li><div><b>2장. 데이터의 양과 질</b><br><span style="color:var(--muted)">데이터를 늘리면, 혹은 엉터리 데이터를 섞으면?</span></div></li>
                    <li style="border:0"><div><b>3장. 본 적 없는 과일</b><br><span style="color:var(--muted)">AI가 처음 보는 것을 만나면 무슨 일이 생길까요?</span></div></li>
                </ul>
            </div>
            <button class="btn" id="start">훈련 시작하기</button>
        </div>`);
    on('start', chapter1);
}

function chapter1() {
    $chapter.textContent = '1장 / 3';
    G.train = [];
    G.testSet = makeSet({ red: 15, tang: 15 });
    labelingScreen({
        chapter: 'CHAPTER 1 · 가르치기',
        title: '이 과일은 무엇인가요?',
        desc: '당신은 모양만 봐도 알지만, AI는 크기와 색깔 숫자만 봅니다. 정답을 알려주세요.',
        queue: makeSet({ red: 5, tang: 5 }),
        train: G.train,
        onDone: chapter1Test
    });
}

function chapter1Test() {
    const test = runTest(G.train, G.testSet);
    const wrongLabels = G.train.filter(t => t.label !== t.cls).length;
    G.history = [{ n: G.train.length, acc: test.acc, noisy: false }];
    render(`
        <p class="eyebrow">CHAPTER 1 · 시험 결과</p>
        <h2>AI가 처음 보는 과일 30개로 시험을 봤어요</h2>
        <div class="play">
            <div class="card">
                <div class="score">${test.acc}%<small> 정답률 (${test.correct}/30)</small></div>
                ${resultsHTML(test)}
                ${wrongLabels ? `<p style="color:var(--bad);font-size:14px">가르칠 때 ${wrongLabels}개를 잘못 알려줬어요. AI는 그대로 믿고 배웠습니다.</p>` : ''}
            </div>
            ${mapHTML('AI가 배운 규칙', '당신이 가르친 10개의 점으로 AI가 세상을 나눈 결과예요.')}
        </div>
        <div class="lesson">
            <strong>💡 AI는 '규칙'이 아니라 '예시'로 배웁니다</strong>
            아무도 "빨갛고 크면 사과"라고 코딩하지 않았어요. AI는 당신이 보여준 예시 중 가장 비슷한 것을 찾아 판단합니다. 이것이 머신러닝의 기본 원리입니다.
        </div>
        <button class="btn" id="next">2장으로 →</button>`, { train: G.train });
    on('next', chapter2);
}

function chapter2() {
    $chapter.textContent = '2장 / 3';
    const last = G.history[G.history.length - 1];
    const maxAcc = Math.max(...G.history.map(h => h.acc));
    render(`
        <p class="eyebrow">CHAPTER 2 · 데이터의 양과 질</p>
        <h2>데이터를 더 모아 볼까요?</h2>
        <p class="lead">버튼을 누를 때마다 과일 10개가 학습 데이터에 추가되고, 같은 시험지로 다시 시험을 봅니다.</p>
        <div class="play">
            <div class="card">
                <div class="score">${last.acc}%<small> 현재 정답률 · 학습 데이터 ${G.train.length}개</small></div>
                <div class="history">${G.history.map(h => `
                    <div class="col"><span>${h.acc}</span><div class="bar ${h.noisy ? 'noisy' : ''}" style="height:${h.acc}%"></div></div>`).join('')}
                </div>
                <div class="history-x">${G.history.map(h => `<span>${h.n}개</span>`).join('')}</div>
                <p style="font-size:13px;color:var(--muted);margin:6px 0 16px">파랑 = 정확한 데이터 추가, 빨강 = 엉터리 데이터 추가</p>
                <div class="row">
                    <button class="btn" id="add-good" ${G.history.length >= 8 ? 'disabled' : ''}>✅ 정확한 데이터 +10</button>
                    <button class="btn ghost" id="add-bad" ${G.history.length >= 8 ? 'disabled' : ''}>🤪 엉터리 데이터 +10</button>
                </div>
                <p style="font-size:13px;color:var(--muted);margin-top:10px">엉터리 데이터: 라벨이 대부분 틀린 데이터예요.</p>
            </div>
            ${mapHTML('AI의 머릿속', '점이 늘어날수록 경계선이 어떻게 변하는지 보세요.')}
        </div>
        ${G.history.length >= 3 ? `
        <div class="lesson">
            <strong>💡 데이터는 양보다 질입니다</strong>
            이미 충분히 배운 AI는 정확한 데이터를 더 넣어도 정답률이 조금씩만 오릅니다. 반면 틀린 데이터는 바로 성능을 떨어뜨립니다. "Garbage in, garbage out" — 쓰레기를 넣으면 쓰레기가 나옵니다.
            ${maxAcc > last.acc ? '<br>방금 정답률이 최고점보다 떨어졌죠? 그게 바로 엉터리 데이터의 효과입니다.' : ''}
        </div>
        <button class="btn" id="next">3장으로 →</button>` : '<p style="margin-top:20px;color:var(--muted)">버튼을 두 번 이상 눌러보면 다음 장이 열립니다.</p>'}`,
        { train: G.train });

    const add = noisy => {
        G.train.push(...labeled(makeSet({ red: 5, tang: 5 }), noisy ? NOISE : 0));
        G.history.push({ n: G.train.length, acc: runTest(G.train, G.testSet).acc, noisy });
        chapter2();
    };
    on('add-good', () => add(false));
    on('add-bad', () => add(true));
    on('next', chapter3);
}

function chapter3() {
    $chapter.textContent = '3장 / 3';
    G.train = labeled(makeSet({ red: 20, tang: 20 }));
    G.testSet3 = makeSet({ green: 10, red: 5, tang: 5 });
    render(`
        <p class="eyebrow">CHAPTER 3 · 본 적 없는 과일</p>
        <h2>새로운 AI를 준비했어요</h2>
        <p class="lead">이 AI는 정확한 데이터 40개로 잘 훈련됐습니다. 그런데 이 AI가 본 사과는 전부 <b>빨간 사과</b>였어요. 오늘 시험지에는 처음 보는 과일이 섞여 있습니다.</p>
        <div class="play">
            <div class="card">
                <p class="sees-title" style="margin-bottom:8px">오늘의 시험지에 섞인 과일</p>
                <div class="fruit-stage">${fruitSVG(makeFruit('green'), 140)}</div>
                <p style="text-align:center;font-weight:700">🍏 풋사과</p>
                <p style="text-align:center;color:var(--muted);font-size:14px;margin-bottom:16px">사람에겐 당연히 사과. AI에게는?</p>
                <button class="btn" id="test" style="width:100%">시험 보기</button>
            </div>
            ${mapHTML('AI의 머릿속', '위쪽(초록색 영역)에는 학습 데이터가 하나도 없어요.')}
        </div>`, { train: G.train });
    on('test', chapter3Test);
}

function chapter3Test() {
    const test = runTest(G.train, G.testSet3);
    const greenWrong = test.results.filter(r => r.f.kind === 'green' && !r.ok).length;
    render(`
        <p class="eyebrow">CHAPTER 3 · 시험 결과</p>
        <h2>${greenWrong >= 5 ? '풋사과를 귤이라고 하네요 😅' : '시험 결과'}</h2>
        <div class="play">
            <div class="card">
                <div class="score">${test.acc}%<small> 정답률 · 풋사과 10개 중 ${10 - greenWrong}개 정답</small></div>
                ${resultsHTML(test)}
            </div>
            ${mapHTML('왜 틀렸을까?', '초록색 쪽에 사과 예시가 없으니, 그나마 색이 가까운 귤 쪽으로 판단했어요.')}
        </div>
        <div class="lesson">
            <strong>💡 AI는 본 적 없는 것 앞에서 자신 있게 틀립니다</strong>
            AI는 "모르겠다"고 말하지 않고, 아는 것 중 가장 비슷한 답을 냅니다. 학습 데이터가 한쪽으로 치우치면(데이터 편향) 현실에서 엉뚱한 판단을 하게 됩니다.
        </div>
        <button class="btn" id="teach">풋사과 5개 가르치기 →</button>`, { train: G.train });
    on('teach', () => labelingScreen({
        chapter: 'CHAPTER 3 · 편향 고치기',
        title: '풋사과를 가르쳐 주세요',
        desc: '빈 곳을 채워주면 AI의 판단 영역이 어떻게 바뀌는지 보세요.',
        queue: makeSet({ green: 5 }),
        train: G.train,
        onDone: chapter3Retest
    }));
}

function chapter3Retest() {
    const test = runTest(G.train, G.testSet3);
    const greenOk = test.results.filter(r => r.f.kind === 'green' && r.ok).length;
    render(`
        <p class="eyebrow">CHAPTER 3 · 재시험</p>
        <h2>같은 시험지로 다시 봤어요</h2>
        <div class="play">
            <div class="card">
                <div class="score">${test.acc}%<small> 정답률 · 풋사과 10개 중 ${greenOk}개 정답</small></div>
                ${resultsHTML(test)}
            </div>
            ${mapHTML('AI의 머릿속', '풋사과 예시 5개만으로 초록 영역이 사과로 바뀌었어요.')}
        </div>
        <div class="lesson">
            <strong>💡 고치는 방법도 데이터입니다</strong>
            다양한 경우를 골고루 보여주면 AI는 다시 똑똑해집니다. 실제 AI 서비스가 끊임없이 데이터를 모으고 다듬는 이유입니다.
        </div>
        <button class="btn" id="next">훈련 완료 →</button>`, { train: G.train });
    on('next', ending);
}

function ending() {
    $chapter.textContent = '완료';
    render(`
        <div class="stack" style="max-width:640px">
            <p class="eyebrow">TRAINING COMPLETE</p>
            <h1>🎉 훈련 완료!</h1>
            <p class="lead">당신은 방금 머신러닝의 핵심 세 가지를 직접 해봤습니다.</p>
            <div class="card">
                <ul class="lessons">
                    <li><div><b>AI는 예시로 배운다</b><br><span style="color:var(--muted)">사람이 규칙을 짜는 대신, 정답이 붙은 데이터에서 패턴을 찾습니다.</span></div></li>
                    <li><div><b>데이터의 양과 질이 성능을 정한다</b><br><span style="color:var(--muted)">좋은 데이터는 AI를 똑똑하게, 틀린 데이터는 어리석게 만듭니다.</span></div></li>
                    <li style="border:0"><div><b>본 적 없는 것은 모른다</b><br><span style="color:var(--muted)">치우친 데이터로 배운 AI는 낯선 상황에서 자신 있게 틀립니다.</span></div></li>
                </ul>
            </div>
            <div class="row">
                <button class="btn ghost" id="again">처음부터 다시</button>
                <a class="btn" href="${HOMEPAGE}">Flat AI 다른 게임 보기</a>
            </div>
        </div>`);
    on('again', intro);
}

intro();
