/* 용어 사전 — 화면 쪽.
 *
 * 용어 목록과 수치는 utils/visualization_example/glossary.py 가 보내 준다.
 * 여기는 그것을 세 가지 캔버스(격자 / 구간 / 벡터) 위에 표시하는 일만 한다.
 */

(function () {
  "use strict";

  var G = null;        // 서버가 준 사전
  var picked = null;   // 지금 고른 용어 id
  var overlay = null;  // 짝 비교용 추가 강조 {id, cls}

  var $ = function (id) { return document.getElementById(id); };
  var elBanner = $("banner"), elList = $("glist"), elCard = $("termcard"), elPairs = $("pairs");

  // 축소판 격자의 종단 패턴 — 계단 모양. 토큰 i 가 몇 평면까지 살아 있는가.
  var TERM_PLANE = [8, 8, 3, 8, 5, 6, 2, 8, 4, 7, 2, 8, 3, 5, 8, 4];

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  /** 아주 작은 인라인 마크다운: **굵게**, `코드`, ```블록```, 줄바꿈 */
  function md(s) {
    var parts = String(s || "").split("```");
    var out = "";
    parts.forEach(function (p, i) {
      if (i % 2 === 1) { out += "<pre>" + esc(p.replace(/^\n/, "")) + "</pre>"; return; }
      var t = esc(p)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
      out += t.split(/\n\n+/).filter(Boolean)
        .map(function (x) { return "<p>" + x.replace(/\n/g, "<br>") + "</p>"; }).join("");
    });
    return out;
  }

  function term(id) {
    return G.terms.filter(function (t) { return t.id === id; })[0];
  }

  /* ---------------- 서버 ---------------- */

  fetch("/api/glossary")
    .then(function (r) { return r.json().then(function (d) {
      if (!r.ok) throw new Error(d.error || "HTTP " + r.status); return d; }); })
    .then(function (d) {
      G = d;
      $("scaleNote").innerHTML = md(d.scale_note) + '<p style="margin:6px 0 0">수치 출처: <code>' +
        esc(d.real.source) + "</code></p>";
      renderList();
      renderPairs();
      pick("token");
    })
    .catch(function (e) {
      elBanner.hidden = false;
      elBanner.textContent = "서버 오류 — " + e.message +
        " (python -m utils.visualization_example 로 서버가 떠 있는지 확인)";
    });

  /* ---------------- 격자 캔버스 ---------------- */

  function isLive(row, col) { return row < TERM_PLANE[col]; }

  function drawGrid(hl, hl2) {
    var d = G.demo, T = d.n_tokens, P = d.n_planes;
    var hits = {};           // "r,c" -> class
    var rowHit = -1, colHit = -1;

    function mark(spec, cls) {
      if (!spec) return;
      if (spec.kind === "column") {
        colHit = spec.index;
        for (var r = 0; r < P; r++) hits[r + "," + spec.index] = cls;
      } else if (spec.kind === "row") {
        rowHit = spec.index;
        for (var c = 0; c < T; c++) hits[spec.index + "," + c] = cls;
      } else if (spec.kind === "chunk") {
        var len = d[spec.unit] || spec.len || 4;
        rowHit = spec.row;
        for (var i = 0; i < len; i++) hits[spec.row + "," + (spec.start + i)] = cls;
      } else if (spec.kind === "cell") {
        hits[spec.row + "," + spec.col] = cls;
        rowHit = spec.row; colHit = spec.col;
      } else if (spec.kind === "all") {
        for (var r2 = 0; r2 < P; r2++)
          for (var c2 = 0; c2 < T; c2++) hits[r2 + "," + c2] = cls;
      } else if (spec.kind === "live") {
        for (var r3 = 0; r3 < P; r3++)
          for (var c3 = 0; c3 < T; c3++) if (isLive(r3, c3)) hits[r3 + "," + c3] = cls;
      } else if (spec.kind === "dead") {
        for (var r4 = 0; r4 < P; r4++)
          for (var c4 = 0; c4 < T; c4++) if (!isLive(r4, c4)) hits[r4 + "," + c4] = cls;
      }
    }
    mark(hl, "hit");
    mark(hl2, "hit2");

    var h = '<div class="grid-rows">';
    for (var r = 0; r < P; r++) {
      h += '<div class="grow' + (r === rowHit ? " hit" : "") + '">' +
        '<span class="grow-lab">b' + (P - 1 - r) + '</span><span class="gcells">';
      for (var c = 0; c < T; c++) {
        var k = hits[r + "," + c];
        var cls = "gcell " + (isLive(r, c) ? "live" : "dead");
        if (k) cls += " " + k;
        h += '<span class="' + cls + '"></span>';
      }
      h += "</span></div>";
    }
    h += "</div>";

    h += '<div class="gaxis">';
    for (var c2 = 0; c2 < T; c2++)
      h += '<span class="' + (c2 === colHit ? "hit" : "") + '">' + c2 + "</span>";
    h += "</div>";
    h += '<div class="gaxis-title">토큰 (가로) / 비트평면 (세로, 위가 MSB)</div>';

    h += '<div class="glegend">' +
      '<span><i style="background:var(--line-soft);border-color:var(--line)"></i>아직 읽어야 함</span>' +
      '<span><i style="background:transparent;border-color:var(--line);border-style:dashed"></i>종단됨</span>' +
      '<span><i style="background:var(--prod);border-color:var(--prod)"></i>지금 고른 용어</span>' +
      (hl2 ? '<span><i style="background:var(--row);border-color:var(--row)"></i>비교 대상</span>' : "") +
      "</div>";

    $("canvasGrid").innerHTML = h;
  }

  /* ---------------- 구간 캔버스 ---------------- */

  function drawBracket(which) {
    // 손으로 검산 가능한 예제 — docs/background/attention_walkthrough.md §4-4 와 같은 수
    var L = 1667, S = 2432, U = 3314, TH = 1667, TRUE = 2696;
    var lo = 1400, hi = 3500, W = 640, H = 150;
    var x = function (v) { return 40 + (v - lo) / (hi - lo) * (W - 80); };
    function on(k) { return which === k ? " hit" : ""; }

    var h = '<div class="bline"><svg viewBox="0 0 ' + W + " " + H + '">';
    h += '<line class="span' + (which === "gap" ? " hit" : "") + '" x1="' + x(L) +
      '" y1="70" x2="' + x(U) + '" y2="70"/>';
    h += '<line class="axis" x1="20" y1="70" x2="' + (W - 10) + '" y2="70"/>';

    [["l", L, "L_m", 40], ["s_m", S, "S_m", 105], ["upper", U, "S_m + R_m", 40],
     ["theta", TH, "θ", 130]].forEach(function (m) {
      var k = m[0], v = m[1], lab = m[2], ty = m[3];
      h += '<line class="tick' + on(k) + '" x1="' + x(v) + '" y1="50" x2="' + x(v) + '" y2="90"/>';
      h += '<text class="' + (which === k ? "hit" : "") + '" x="' + x(v) +
        '" y="' + ty + '" text-anchor="middle">' + lab + " = " + v + "</text>";
    });

    h += '<circle class="truth" cx="' + x(TRUE) + '" cy="70" r="5"/>';
    h += '<text class="' + (which === "truth" ? "hit" : "") + '" x="' + x(TRUE) +
      '" y="30" text-anchor="middle">참값 s = ' + TRUE + "</text>";
    h += "</svg></div>";
    h += '<div class="gaxis-title">점수 축 — 평면 2장 처리 시점 (walkthrough §4-4 의 수)</div>';

    $("canvasBracket").innerHTML = h;
  }

  /* ---------------- 벡터 캔버스 ---------------- */

  function drawVector(kind) {
    var q = [5, -3, 7, 2, -1, 4, 6, -2];
    var bit = [1, 0, 1, 0, 1, 1, 0, 0];
    function row(lab, cells, hit) {
      return '<div class="vecrow' + (hit ? " hit" : "") + '">' +
        '<span class="vecrow-lab">' + lab + '</span><span class="veccells">' +
        cells + "</span></div>";
    }
    var qc = q.map(function (v) {
      return '<span class="veccell ' + (v >= 0 ? "pos" : "neg") +
        (kind === "q" || kind === "pair" ? " hit" : "") + '">' + v + "</span>";
    }).join("");
    var kc = bit.map(function (b) {
      return '<span class="veccell ' + (b ? "bit1" : "bit0") +
        (kind === "k" || kind === "pair" || kind === "bit" ? " hit" : "") + '">' + b + "</span>";
    }).join("");

    var sum = q.reduce(function (a, v, i) { return a + (bit[i] ? v : 0); }, 0);
    var qp = q.filter(function (v) { return v > 0; }).reduce(function (a, v) { return a + v; }, 0);
    var qm = q.filter(function (v) { return v < 0; }).reduce(function (a, v) { return a + v; }, 0);

    var h = row("q", qc, kind === "q" || kind === "pair");
    h += row("K 한 평면", kc, kind === "k" || kind === "pair" || kind === "bit");
    h += '<div class="vecnote">head_dim = ' + q.length + " (실제 " + G.real.head_dim + ")</div>";
    if (kind === "bit") {
      h += '<div class="vecnote">비트가 1인 자리의 q 만 더한다 → P = ' + sum + " (곱셈 없음)</div>";
    } else if (kind === "q") {
      h += '<div class="vecnote">Q+ = ' + qp + " · Q− = " + qm + " (스텝당 한 번만 계산)</div>";
    }
    $("canvasVector").innerHTML = h;
  }

  /* ---------------- 캔버스 전환 ---------------- */

  var TAG = { grid: "평면 × 토큰", bracket: "점수 축 (구간)", vector: "q · K 벡터" };
  var HINT = {
    grid: "가로 = 토큰, 세로 = 비트평면. 진한 칸이 아직 읽어야 하는 자리다.",
    bracket: "참값이 반드시 이 구간 안에 있다. 평면을 처리할수록 좁아진다.",
    vector: "부분 내적이 일어나는 자리. 비트가 1인 칸의 q 만 더한다."
  };

  function show(canvas, hl, hl2) {
    ["Grid", "Bracket", "Vector"].forEach(function (n) {
      $("canvas" + n).hidden = (n.toLowerCase() !== canvas);
    });
    $("canvasTag").textContent = TAG[canvas];
    $("canvasHint").textContent = HINT[canvas];
    if (canvas === "grid") drawGrid(hl, hl2);
    else if (canvas === "bracket") drawBracket(hl && hl.which);
    else drawVector(hl && hl.kind);
  }

  /* ---------------- 용어 선택 ---------------- */

  function pick(id, extra) {
    picked = id;
    overlay = extra || null;
    var t = term(id);
    if (!t) return;

    var hl2 = null;
    if (overlay) {
      var o = term(overlay);
      if (o && o.canvas === t.canvas) hl2 = o.highlight;
    }
    show(t.canvas, t.highlight, hl2);

    var h = '<div class="tc-head"><span class="tc-term">' + esc(t.term) + "</span>" +
      '<span class="tc-en">' + esc(t.en) + "</span>" +
      '<span class="tc-group">' + esc(t.group) + "</span></div>";
    h += '<p class="tc-one">' + md(t.one).replace(/^<p>|<\/p>$/g, "") + "</p>";
    h += '<div class="tc-real">실제 값 · <b>' + esc(t.real) + "</b></div>";
    h += '<div class="tc-detail">' + md(t.detail) + "</div>";

    if (t.confuse && t.confuse.length) {
      h += '<div class="tc-confuse"><b>⚠ 헷갈리기 쉬움</b> — 눌러서 같이 보기<br>' +
        t.confuse.map(function (c) {
          var o2 = term(c);
          return o2 ? '<button type="button" data-pick="' + t.id + '" data-with="' + c +
            '">' + esc(o2.term) + "</button>" : "";
        }).join("") + "</div>";
    }
    if (overlay) {
      var ov = term(overlay);
      if (ov) h += '<div class="tc-confuse" style="color:var(--row)">지금 <b style="color:var(--row)">' +
        esc(ov.term) + "</b> 을(를) 주황색으로 겹쳐 놓았다.</div>";
    }

    elCard.innerHTML = h;
    renderList();
  }

  elCard.addEventListener("click", function (e) {
    var b = e.target.closest("[data-with]");
    if (!b) return;
    pick(b.dataset.pick, b.dataset.with);
  });

  /* ---------------- 목록 ---------------- */

  function renderList() {
    elList.innerHTML = G.groups.map(function (g) {
      var items = G.terms.filter(function (t) { return t.group === g; });
      return '<div><div class="ggroup-h">' + esc(g) + '</div><div class="gitems">' +
        items.map(function (t) {
          return '<button type="button" class="gitem' + (t.id === picked ? " on" : "") +
            '" data-id="' + t.id + '">' +
            '<span class="gitem-top"><span class="gitem-term">' + esc(t.term) + "</span>" +
            '<span class="gitem-en">' + esc(t.en) + "</span></span>" +
            '<span class="gitem-one">' + md(t.one).replace(/<\/?p>/g, "") + "</span></button>";
        }).join("") + "</div></div>";
    }).join("");
  }

  elList.addEventListener("click", function (e) {
    var b = e.target.closest(".gitem");
    if (b) pick(b.dataset.id);
  });

  /* ---------------- 헷갈리는 짝 ---------------- */

  function miniCells(unit) {
    var n = G.demo.n_tokens, len = G.demo[unit] || 4;
    var out = "";
    for (var i = 0; i < n; i++) out += "<i" + (i < len ? ' class="a"' : "") + "></i>";
    return out;
  }

  function renderPairs() {
    elPairs.innerHTML = G.pairs.map(function (p) {
      var a = term(p.a), b = term(p.b);
      if (!a || !b) return "";
      var mini = "";
      if (a.canvas === "grid" && b.canvas === "grid" &&
          a.highlight.kind === "chunk" && b.highlight.kind === "chunk") {
        mini = '<div class="pair-mini">' +
          '<div class="mini"><span class="mini-lab a">' + esc(a.term) + " = " +
          G.demo[a.highlight.unit] + '칸</span><span class="mini-cells">' +
          miniCells(a.highlight.unit) + "</span></div>" +
          '<div class="mini"><span class="mini-lab b">' + esc(b.term) + " = " +
          G.demo[b.highlight.unit] + '칸</span><span class="mini-cells">' +
          miniCells(b.highlight.unit).replace(/class="a"/g, 'class="b"') + "</span></div>" +
          "</div>";
      }
      return '<div class="pair"><div class="pair-h"><b>' + esc(p.title) + "</b>" +
        '<span class="pair-btns">' +
        '<button type="button" class="a" data-id="' + a.id + '">' + esc(a.term) + "</button>" +
        '<button type="button" class="b" data-id="' + a.id + '" data-with="' + b.id + '">' +
        "같이 보기</button></span></div>" +
        '<p class="pair-why">' + md(p.why).replace(/<\/?p>/g, "") + "</p>" + mini + "</div>";
    }).join("");
  }

  elPairs.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-id]");
    if (!b) return;
    pick(b.dataset.id, b.dataset.with || null);
    document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
