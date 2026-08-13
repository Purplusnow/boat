/* 경주 주행 재생기 — 미사리 수면을 위에서 본 화면.
 *
 * 서버가 주는 것은 배마다 '구간 경계에서의 시각·가로 오프셋·슬립각' 뿐이고,
 * 좌표 계산은 전부 여기서 한다. 정적 사이트에서 그대로 돈다.
 *
 * ## 왜 드리프트를 그리는가
 *
 * 말은 다리로 방향을 바꾸지만 배는 못 바꾼다. 직선에서 싣고 온 속도를 선회
 * 속도까지 깎아내면서 동시에 방향을 틀어야 하고, 그 둘을 한꺼번에 하는 방법이
 * **선미를 바깥으로 던지는 것**이다. 그래서 선회 중인 배는 선체가 진행 방향보다
 * 안쪽을 향한 채 옆으로 미끄러진다. 이 각도(슬립각)를 안 그리면 배가 레일 위를
 * 미끄러지는 기차처럼 보이고, 경정으로 읽히지 않는다.
 *
 * 슬립각은 서버가 물리로 계산한다(깎아야 할 속도의 비율에 비례). 여기서는
 * **선회 중간에서 최대가 되도록** sin 곡선으로 펴서 그린다 — 진입에서 던지고
 * 정점에서 가장 많이 밀리며 탈출에서 되잡는, 실제 선회의 순서다.
 */
(function () {
  var host = document.getElementById('player');
  if (!host) return;
  var dataEl = document.getElementById('player-data');
  var canvas = document.getElementById('rp-canvas');
  if (!dataEl || !canvas || !canvas.getContext) return;

  var data;
  try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }
  if (!data || !data.boats || !data.boats.length) return;

  var ctx = canvas.getContext('2d');
  var boats = data.boats;
  var C = data.course;
  var ARC = Math.PI * C.r_inner;              // 반원 하나의 중심선 길이
  var LAP = C.straight * 2 + ARC * 2;
  var DUR = data.duration;

  /* 실제 경정정 색. 1 백 · 2 흑 · 3 적 · 4 청 · 5 황 · 6 녹.
     [선체, 번호] 순 — 흰·노란 선체는 번호를 어둡게 쓴다. */
  var HULL = [
    ['#ffffff', '#111111'], ['#242424', '#ffffff'], ['#e0392b', '#ffffff'],
    ['#2160c4', '#ffffff'], ['#f2c521', '#111111'], ['#2e9c52', '#ffffff']
  ];
  function hull(lane) { return HULL[(lane - 1) % 6]; }

  // ── 코스 기하 ────────────────────────────────────────────────
  //
  // **경정장은 트랙이 아니다.** 열린 수면 위에 턴마크 부표 두 개가 떠 있고,
  // 그 사이를 가르는 차단 구조물(센터폰툰)이 있을 뿐이다. 배는 어디로든 갈 수
  // 있고 다만 두 부표를 돌아야 한다 — 그래서 주로 외곽선을 그리면 안 된다.
  // 부표는 course.straight(=부표 간격 300m) 만큼 떨어져 있다.
  //
  // 경정은 **좌선회**(반시계)다: 아래를 왼→오른, 오른쪽 부표를 돌아 위를
  // 오른→왼, 왼쪽 부표를 돌아 제자리.
  var geo = {};
  var VW = 720, VH = 300;                     // 논리 크기(CSS 픽셀)
  var view = 'whole';                         // 'whole' | 'follow'

  /* 두 가지 시야.
     **전체**가 기본이다. 부표 두 개와 센터폰툰, 여섯 척의 위치 관계가 먼저
     보여야 전개가 읽힌다.
     **따라가기**는 무리를 따라가며 확대한다. 300m 를 한 화면에 담으면 축척이
     1.4px/m 라 선회와 드리프트가 잘 안 보이는데, 확대하면 물리는 그대로 두고
     그 움직임이 읽힌다. */
  var FOLLOW_SPAN = 115;                      // 따라가기에서 가로로 보이는 거리(m)

  function layout(focus) {
    var pad = 46;
    if (view === 'whole') {
      var spread = C.r_inner + 26;
      var wm = C.straight + spread * 2, hm = spread * 2;
      var sc = Math.min((VW - pad * 2) / wm, (VH - pad * 2) / hm);
      geo = { sc: sc, ox: VW / 2, oy: VH / 2,
              half: C.straight * sc / 2, R: C.r_inner * sc };
    } else {
      var sc2 = VW / FOLLOW_SPAN;
      // 무리의 한가운데를 화면 중앙에 둔다. focus 는 world 좌표(px, 미확대 기준).
      geo = { sc: sc2, ox: VW / 2 - (focus ? focus.x : 0) * sc2,
              oy: VH / 2 - (focus ? focus.y : 0) * sc2,
              half: C.straight * sc2 / 2, R: C.r_inner * sc2 };
    }
  }

  /* world 좌표(미터, 코스 중심이 원점) → 화면 좌표. */
  function toScreen(wx, wy) {
    return { x: geo.ox + wx * geo.sc, y: geo.oy + wy * geo.sc };
  }

  /* 코스 진행거리 s(m) 와 가로 오프셋 off(m) → **world 좌표(미터)**.
     원점은 코스 중앙, x 는 오른쪽, y 는 아래. 화면 변환은 toScreen 이 맡는다.
     s=0 은 스타트라인이며, 스타트라인은 아래 직선의 start_offset 지점에 있다. */
  var HALF_M = C.straight / 2;
  function world(s, off) {
    var p = ((s + C.start_offset) % LAP + LAP) % LAP;
    var R = C.r_inner + off;
    if (p < C.straight) {                       // 아래 직선: 왼 → 오른
      return { x: -HALF_M + (p / C.straight) * C.straight, y: R };
    }
    p -= C.straight;
    if (p < ARC) {                              // 오른쪽 부표: 아래 → 위
      var a = Math.PI / 2 - (p / ARC) * Math.PI;
      return { x: HALF_M + R * Math.cos(a), y: R * Math.sin(a) };
    }
    p -= ARC;
    if (p < C.straight) {                       // 위 직선: 오른 → 왼
      return { x: HALF_M - (p / C.straight) * C.straight, y: -R };
    }
    p -= C.straight;                            // 왼쪽 부표: 위 → 아래
    var a2 = -Math.PI / 2 - (p / ARC) * Math.PI;
    return { x: -HALF_M + R * Math.cos(a2), y: R * Math.sin(a2) };
  }
  function point(s, off) { var w = world(s, off); return toScreen(w.x, w.y); }

  /* 진행 방향(탄젠트)은 미분식을 쓰지 않고 조금 앞을 찍어 구한다 —
     구간 경계에서 부호를 틀릴 여지가 없다. */
  function heading(s, off) {
    var a = point(s, off), b = point(s + 1.2, off);
    return Math.atan2(b.y - a.y, b.x - a.x);
  }

  // ── 시각 → 배의 상태 ─────────────────────────────────────────
  function stateAt(b, tt) {
    var m = C.marks, kinds = C.kinds, n = kinds.length;
    if (tt <= b.t[0]) return { s: 0, off: b.off[0], slip: 0, done: false, waiting: true };
    for (var k = 0; k < n; k++) {
      if (tt < b.t[k + 1] || k === n - 1) {
        var span = b.t[k + 1] - b.t[k];
        var u = span > 0 ? Math.min((tt - b.t[k]) / span, 1) : 1;
        var s = m[k] + u * (m[k + 1] - m[k]);
        var off = b.off[k] + (b.off[k + 1] - b.off[k]) * u;
        var slip = 0;
        if (kinds[k] === 'T') {
          // 선회 중간에서 가장 크게 밀린다. 진입에서 던지고 정점에서 최대,
          // 탈출에서 되잡는다.
          var bell = Math.sin(Math.PI * u);
          slip = (b.beta[k + 1] || 0) * bell;
          // 밀린 만큼 실제 항적도 바깥으로 부푼다.
          off += (b.beta[k + 1] || 0) / 42 * 5.5 * bell;
        }
        return { s: s, off: off, slip: slip, done: tt >= b.t[n], waiting: false };
      }
    }
    return { s: m[n], off: b.off[n], slip: 0, done: true, waiting: false };
  }

  // ── 그리기 ───────────────────────────────────────────────────
  var dark = matchMedia('(prefers-color-scheme: dark)').matches;

  function drawCourse() {
    // 수면 — 화면 전체가 물이다. 주로 외곽선은 없다.
    var g = ctx.createLinearGradient(0, 0, 0, VH);
    if (dark) { g.addColorStop(0, '#0d2030'); g.addColorStop(1, '#0a1721'); }
    else { g.addColorStop(0, '#2f7fa8'); g.addColorStop(1, '#1f5f80'); }
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, VW, VH);

    // 잔물결 — 수면임을 알리는 최소한의 결. 너무 진하면 배를 가린다.
    ctx.strokeStyle = dark ? 'rgba(255,255,255,.035)' : 'rgba(255,255,255,.07)';
    ctx.lineWidth = 1;
    for (var y = 8; y < VH; y += 13) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      for (var x = 0; x <= VW; x += 18) ctx.lineTo(x, y + (x % 36 === 0 ? 1.6 : -1.6));
      ctx.stroke();
    }

    // 센터폰툰 — 두 부표 사이를 가르는 차단 구조물.
    var c0 = toScreen(0, 0);
    var pw = C.straight * 0.62 * geo.sc, ph = Math.max(9, C.r_inner * 0.22 * geo.sc);
    ctx.fillStyle = dark ? '#26313a' : '#3f4a52';
    ctx.strokeStyle = dark ? '#161d23' : '#2b333a';
    ctx.lineWidth = 1;
    roundRect(c0.x - pw / 2, c0.y - ph / 2, pw, ph, 3);
    ctx.fill(); ctx.stroke();
    ctx.strokeStyle = dark ? 'rgba(255,255,255,.08)' : 'rgba(255,255,255,.14)';
    var step = Math.max(12, 8 * geo.sc);
    for (var px = c0.x - pw / 2 + step; px < c0.x + pw / 2; px += step) {
      ctx.beginPath(); ctx.moveTo(px, c0.y - ph / 2 + 1);
      ctx.lineTo(px, c0.y + ph / 2 - 1); ctx.stroke();
    }

    // 턴마크 부표 두 개 — 실제처럼 빨강·흰 줄무늬
    var m1 = toScreen(HALF_M, 0), m2 = toScreen(-HALF_M, 0);
    drawBuoy(m1.x, m1.y, '1턴마크', -1);
    drawBuoy(m2.x, m2.y, '2턴마크', 1);

    // 스타트/결승선
    var sp = point(0, -3), sp2 = point(0, 24);
    ctx.beginPath();
    ctx.moveTo(sp.x, sp.y); ctx.lineTo(sp2.x, sp2.y);
    ctx.strokeStyle = 'rgba(255,255,255,.9)';
    ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,.85)';
    ctx.font = '600 10px sans-serif';
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('스타트 / 결승', sp2.x + 5, sp2.y + 3);
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* 턴마크 — 빨강·흰 줄무늬 부표. 배가 반드시 바깥으로 돌아야 하는 지점이다. */
  function drawBuoy(x, y, label, side) {
    var rr = Math.max(6, Math.min(2.2 * geo.sc, 16));
    ctx.save();
    ctx.beginPath(); ctx.arc(x, y + 2, rr * 1.15, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,.25)'; ctx.fill();      // 물그림자
    ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill();
    ctx.save(); ctx.clip();
    ctx.fillStyle = '#e03a26';
    ctx.fillRect(x - rr, y - rr, rr * 2, rr * 0.66);
    ctx.fillRect(x - rr, y + rr * 0.2, rr * 2, rr * 0.66);
    ctx.restore();
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(0,0,0,.45)';
    ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    ctx.font = '600 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + side * (rr + 12) + (side < 0 ? 0 : 4));
    ctx.restore();
  }

  /* 항적 — 배가 지나온 자리에 남는 물살. 드리프트 중에는 옆으로 길게 남는다. */
  function drawWake(b, tt) {
    var pts = [];
    for (var d = 0.15; d <= 1.2; d += 0.15) {
      var st = stateAt(b, tt - d);
      if (st.waiting) break;
      pts.push({ p: point(st.s, st.off), slip: st.slip });
    }
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].p.x, pts[0].p.y);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].p.x, pts[i].p.y);
    ctx.strokeStyle = 'rgba(255,255,255,.55)';
    ctx.lineWidth = 3.2; ctx.lineCap = 'round';
    ctx.stroke();
  }

  /* 경정정 — 위에서 본 활주형 선체. 선수가 +x 를 향하도록 그리고,
     진행 방향 + 슬립각만큼 회전시킨다. */
  function drawBoat(b, st) {
    var p = point(st.s, st.off);
    var dir = heading(st.s, st.off);
    // 좌선회(반시계)이므로 선체는 진행 방향보다 **안쪽**을 향한다.
    // 캔버스는 y 가 아래로 자라므로 시계 반대 방향은 각을 빼는 쪽이다.
    var ang = dir - st.slip * Math.PI / 180;

    var col = hull(b.lane);
    // 실제 경정정은 길이 약 3m 다. 축척대로 그리면 6px 도 안 되어 정번이 아예
    // 안 보인다. **읽히는 것이 우선**이라 축척보다 크게 그린다 — 위치와 각도는
    // 물리 그대로이고 크기만 과장한 것이다.
    // 실제 경정정은 길이 약 3m. 따라가기 시야(5.6px/m)에서 그대로 그리면
    // 17px 라 정번이 빠듯하다. 1.8배쯤 키워 읽히게 한다 — 위치·각도는 물리
    // 그대로이고 크기만 과장이다.
    var L = Math.max(28, geo.sc * 5.5), W = L * 0.46;

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(ang);

    // 물보라 — 많이 밀릴수록 선미 바깥으로 크게 튄다.
    if (st.slip > 6) {
      var spray = st.slip / 42;
      ctx.beginPath();
      for (var i = 0; i < 7; i++) {
        var rx = -L * (0.35 + Math.random() * 0.8);
        var ry = W * (0.4 + Math.random() * 1.9) * spray;
        ctx.moveTo(rx, ry);
        ctx.arc(rx, ry, 1.1 + Math.random() * 1.6, 0, Math.PI * 2);
      }
      ctx.fillStyle = 'rgba(255,255,255,' + (0.30 + 0.45 * spray) + ')';
      ctx.fill();
    }

    // 선체
    ctx.beginPath();
    ctx.moveTo(L * 0.52, 0);
    ctx.lineTo(L * 0.16, -W * 0.46);
    ctx.lineTo(L * 0.04, -W * 0.92);      // 스펀슨(활주면)
    ctx.lineTo(-L * 0.12, -W * 0.80);
    ctx.lineTo(-L * 0.22, -W * 0.42);
    ctx.lineTo(-L * 0.50, -W * 0.46);
    ctx.lineTo(-L * 0.50, W * 0.46);
    ctx.lineTo(-L * 0.22, W * 0.42);
    ctx.lineTo(-L * 0.12, W * 0.80);
    ctx.lineTo(L * 0.04, W * 0.92);
    ctx.lineTo(L * 0.16, W * 0.46);
    ctx.closePath();
    ctx.fillStyle = col[0];
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,.55)';
    ctx.lineWidth = 1; ctx.stroke();

    // 선외기
    ctx.fillStyle = '#3d3d3d';
    ctx.fillRect(-L * 0.62, -W * 0.24, L * 0.14, W * 0.48);

    // 정번 — 선체 위에 얹는다. 글자는 늘 똑바로 서야 읽힌다.
    ctx.rotate(-ang);
    // 정번은 늘 똑바로 서야 읽힌다. 선체가 어두우면 밝은 테두리를 둘러
    // 어느 색 위에서도 떨어져 보이게 한다.
    ctx.font = '800 ' + Math.round(L * 0.54) + 'px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.strokeStyle = col[1] === '#ffffff' ? 'rgba(0,0,0,.55)' : 'rgba(255,255,255,.75)';
    ctx.strokeText(String(b.lane), 0, 0);
    ctx.fillStyle = col[1];
    ctx.fillText(String(b.lane), 0, 0);
    ctx.restore();
  }

  // ── 순위판 ───────────────────────────────────────────────────
  var orderEl = document.getElementById('rp-order');
  function drawOrder(tt) {
    if (!orderEl) return;
    var rows = boats.map(function (b) {
      return { b: b, st: stateAt(b, tt) };
    }).sort(function (x, y) {
      if (Math.abs(y.st.s - x.st.s) > 0.5) return y.st.s - x.st.s;
      // 둘 다 결승선을 넘었으면 진행거리가 같다. 완주 시각으로 가른다.
      return x.b.t[x.b.t.length - 1] - y.b.t[y.b.t.length - 1];
    });
    orderEl.innerHTML = rows.map(function (r, i) {
      var c = hull(r.b.lane);
      // 각도만 덩그러니 두면 무슨 각도인지 알 수 없다. 두 줄로 나눠 아래에
      // 이름을 붙인다 — '선회 밀림'은 선체가 진행 방향보다 안쪽을 향한 정도다.
      var sub = r.st.slip >= 8
        ? '<span class="rp-drift">선회 밀림 ' + Math.round(r.st.slip) + '°</span>'
        : '<span class="rp-calm">직주</span>';
      return '<li>' +
        '<span class="rp-top"><span class="rp-rank">' + (i + 1) + '</span>' +
        '<i class="rp-chip" style="background:' + c[0] + ';color:' + c[1] + '">' +
        r.b.lane + '</i>' + r.b.racer_nm + '</span>' + sub + '</li>';
    }).join('');
  }

  // ── 재생 ─────────────────────────────────────────────────────
  var t = 0, playing = false, rate = 1.6, last = 0, raf = null;
  var seek = document.getElementById('rp-seek');
  var btnPlay = document.getElementById('rp-play');
  var clockEl = document.getElementById('rp-clock');

  function render() {
    // 카메라는 무리의 한가운데를 따라간다. 선두만 따라가면 뒤처진 배가 화면
    // 밖으로 사라져 '몇 척이 남았는지'조차 안 보인다.
    var sts = boats.map(function (b) { return stateAt(b, t); });
    var ws = sts.map(function (st, i) { return world(st.s, st.off); });
    var focus = {
      x: ws.reduce(function (a, w) { return a + w.x; }, 0) / ws.length,
      y: ws.reduce(function (a, w) { return a + w.y; }, 0) / ws.length
    };
    layout(focus);
    drawCourse();
    boats.forEach(function (b) {
      var st = stateAt(b, t);
      if (!st.waiting) drawWake(b, t);
    });
    // 1턴마크에서는 여섯 척이 겹친다. **뒤선 배부터** 그려 선두가 위에 오게
    // 한다 — 순서를 안 잡으면 선두가 뒤차에 가려 정번이 안 보인다.
    boats.map(function (b) { return { b: b, st: stateAt(b, t) }; })
      .sort(function (x, y) { return x.st.s - y.st.s; })
      .forEach(function (o) { drawBoat(o.b, o.st); });
    drawOrder(t);
    if (clockEl) clockEl.textContent = t.toFixed(1) + 's / ' + DUR.toFixed(1) + 's';
    if (seek && document.activeElement !== seek) {
      seek.value = String(Math.round(t / DUR * 1000));
    }
  }

  function tick(now) {
    if (!playing) return;
    var dt = (now - last) / 1000;
    last = now;
    t += dt * rate;
    if (t >= DUR) { t = DUR; playing = false; setPlayLabel(); }
    render();
    if (playing) raf = requestAnimationFrame(tick);
  }
  function setPlayLabel() {
    if (btnPlay) btnPlay.textContent = playing ? '⏸ 멈춤' : (t >= DUR ? '↻ 다시' : '▶ 재생');
  }
  function play() {
    if (t >= DUR) t = 0;
    playing = true; last = performance.now(); setPlayLabel();
    raf = requestAnimationFrame(tick);
  }
  function pause() { playing = false; if (raf) cancelAnimationFrame(raf); setPlayLabel(); }

  if (btnPlay) btnPlay.addEventListener('click', function () { playing ? pause() : play(); });
  var btnReplay = document.getElementById('rp-replay');
  if (btnReplay) btnReplay.addEventListener('click', function () { t = 0; play(); });
  if (seek) seek.addEventListener('input', function () {
    pause(); t = Number(seek.value) / 1000 * DUR; render();
  });
  Array.prototype.forEach.call(host.querySelectorAll('.rp-view'), function (btn) {
    btn.addEventListener('click', function () {
      view = btn.dataset.view;
      Array.prototype.forEach.call(host.querySelectorAll('.rp-view'), function (o) {
        o.classList.toggle('on', o === btn);
      });
      render();
    });
  });
  Array.prototype.forEach.call(host.querySelectorAll('.rp-rate'), function (btn) {
    btn.addEventListener('click', function () {
      rate = Number(btn.dataset.rate);
      Array.prototype.forEach.call(host.querySelectorAll('.rp-rate'), function (o) {
        o.classList.toggle('on', o === btn);
      });
    });
  });

  // 캔버스를 화면 폭에 맞춘다. 고해상도 화면에서 흐려지지 않게 배율을 준다.
  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    VW = host.clientWidth || 720;
    VH = Math.round(Math.min(Math.max(VW * 0.50, 260), 440));
    canvas.style.width = VW + 'px';
    canvas.style.height = VH + 'px';
    canvas.width = Math.round(VW * dpr);
    canvas.height = Math.round(VH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // 이후 그리기는 전부 CSS 픽셀 기준
    render();
  }
  window.addEventListener('resize', resize);
  resize();
  setPlayLabel();
})();
