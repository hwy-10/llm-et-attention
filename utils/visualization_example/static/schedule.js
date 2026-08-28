/* 조기 종단 스케줄 — 화면 쪽.
 *
 * ★ 여기도 산술을 하지 않는다.
 *   어느 사이클에 어느 토큰이 어느 레인에 실리는지는 전부
 *   utils/visualization_example/schedule_demo.py 가 만들어 보낸 것이고,
 *   그 트레이스 길이는 서버에서 src/schedule.py 의 apply() 사이클 수와
 *   대조된 뒤에야 내려온다. 여기는 그걸 칸으로 옮길 뿐이다.
 */

(function () {
  "use strict";

  var CONTROLS = [
    "n_tokens", "n_planes", "survival", "clustering",
    "lanes", "word_tokens", "n_ports", "batch_size", "two_phase_split",
    "compaction_cost_cycles"
  ];
  var CHECKS = ["mem_overlap"];   // 범위가 아니라 켜고 끄는 값

  var TAGS = {
    none:       "종단 무시 — 죽은 토큰도 그대로 레인을 차지한다",
    batch:      "묶음이 통째로 끝날 때까지 유지 — 흩어져 죽으면 회수가 안 된다",
    compaction: "매 평면 생존 토큰을 앞으로 당긴다 — 대신 평면마다 압축 비용",
    two_phase:  "앞은 전체 스캔, 뒤는 압축된 상태 — 압축은 한 번만"
  };

  // "real" 은 서버가 config/hardware.yaml 에서 읽어 준 값으로 채운다 (아래 applyPreset).
  var PRESETS = {
    tiny:      { n_tokens: 16,  n_planes: 6, survival: 0.62, clustering: 0,
                 lanes: 4,  word_tokens: 4,  n_ports: 2, mem_overlap: true,
                 batch_size: 8,  two_phase_split: 2, compaction_cost_cycles: 2 },
    membound:  { n_tokens: 128, n_planes: 8, survival: 0.62, clustering: 0,
                 lanes: 32, word_tokens: 8,  n_ports: 1, mem_overlap: true,
                 batch_size: 32, two_phase_split: 3, compaction_cost_cycles: 2 },
    scattered: { n_tokens: 128, n_planes: 8, survival: 0.60, clustering: 0,
                 lanes: 32, word_tokens: 32, batch_size: 32, two_phase_split: 3, compaction_cost_cycles: 2 },
    clustered: { n_tokens: 128, n_planes: 8, survival: 0.60, clustering: 1,
                 lanes: 32, word_tokens: 32, batch_size: 32, two_phase_split: 3, compaction_cost_cycles: 2 },
    noterm:    { n_tokens: 128, n_planes: 8, survival: 0.98, clustering: 0,
                 lanes: 32, word_tokens: 32, batch_size: 32, two_phase_split: 3, compaction_cost_cycles: 6 },
    partial:   { n_tokens: 200, n_planes: 8, survival: 1.00, clustering: 0,
                 lanes: 8,  word_tokens: 32, batch_size: 64, two_phase_split: 3, compaction_cost_cycles: 2 }
  };

  // 프리셋마다 어느 정책을 먼저 보여 줄지 — 그 상황의 주인공으로 맞춘다
  var PRESET_FOCUS = {
    tiny: "compaction", scattered: "compaction", clustered: "batch",
    noterm: "none", partial: "batch", real: "compaction", membound: "compaction"
  };

  var data = null;
  var selPolicy = "compaction";
  var selPlane = 0;
  var pickToken = null;       // 수명 막대에서 고른 토큰
  var upTo = Infinity;        // 재생 중 여기까지만 보여 준다
  var playTimer = null, reqSeq = 0, timer = null;

  var $ = function (id) { return document.getElementById(id); };
  var elBanner = $("banner"), elTabs = $("tabs"), elTabNote = $("tabNote");
  var elTL = $("timeline"), elTLStat = $("tlStat");
  var elLife = $("life"), elBram = $("bram"), elHero = $("hero"), elNums = $("nums");
  var btnPlay = $("btnPlay"), btnAll = $("btnAll");

  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  /* ---------------- 서버 ---------------- */

  function refresh() {
    var mine = ++reqSeq;
    var p = {};
    CONTROLS.forEach(function (k) { p[k] = parseFloat($(k).value); });
    CHECKS.forEach(function (k) { p[k] = $(k).checked; });

    fetch("/api/schedule", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p)
    }).then(function (res) {
      return res.json().then(function (d) {
        if (!res.ok) throw new Error(d && d.error ? d.error : "HTTP " + res.status);
        return d;
      });
    }).then(function (d) {
      if (mine !== reqSeq) return;
      data = d;
      if (selPlane >= d.spec.n_planes) selPlane = d.spec.n_planes - 1;
      if (pickToken !== null && pickToken >= d.spec.n_tokens) pickToken = null;
      elBanner.hidden = true;
      var hw = d.hardware_defaults || {};
      var hint = document.getElementById("realHint");
      if (hint) {
        hint.textContent = (hw.source || "기본값") + " — lanes " + hw.lanes +
          " · word_tokens " + hw.word_tokens + " · batch " + hw.batch_size +
          " · 평면 " + hw.n_planes + " · 문맥 128토큰";
      }
      render();
    }).catch(function (e) {
      elBanner.hidden = false;
      elBanner.textContent = "서버 오류 — " + e.message;
    });
  }

  function later() { clearTimeout(timer); timer = setTimeout(refresh, 90); }

  /* ---------------- 조작 ---------------- */

  function syncLabels() {
    CONTROLS.forEach(function (k) {
      var v = $(k).value, out = $("v_" + k);
      if (!out) return;
      if (k === "survival") out.textContent = parseFloat(v).toFixed(2);
      else if (k === "clustering") {
        var f = parseFloat(v);
        out.textContent = (f < 0.34 ? "흩어짐 " : f > 0.66 ? "뭉침 " : "중간 ") + f.toFixed(2);
      } else out.textContent = v;
    });
  }

  CONTROLS.concat(CHECKS).forEach(function (k) {
    $(k).addEventListener("input", function () {
      syncLabels(); stopPlay(); upTo = Infinity; later();
    });
  });

  document.querySelectorAll("[data-preset]").forEach(function (b) {
    b.addEventListener("click", function () { applyPreset(b.dataset.preset); });
  });

  function applyPreset(name) {
    var p;
    if (name === "real") {
      // 손으로 베낀 숫자가 아니라 서버가 config/hardware.yaml 에서 읽어 준 값
      var hw = (data && data.hardware_defaults) || {};
      p = {
        n_tokens: 128, survival: 0.62, clustering: 0,
        n_planes: hw.n_planes || 8,
        lanes: hw.lanes || 32,
        word_tokens: hw.word_tokens || 32,
        batch_size: hw.batch_size || 32,
        two_phase_split: hw.two_phase_split || 3,
        compaction_cost_cycles: hw.compaction_cost_cycles === undefined ? 2 : hw.compaction_cost_cycles,
        n_ports: hw.n_ports || 2,
        mem_overlap: hw.mem_overlap !== false
      };
    } else {
      p = PRESETS[name];
    }
    if (!p) return;
    Object.keys(p).forEach(function (k) {
      var el = $(k);
      if (!el) return;
      if (el.type === "checkbox") el.checked = !!p[k];
      else el.value = p[k];
    });
    if (PRESET_FOCUS[name]) selPolicy = PRESET_FOCUS[name];
    syncLabels(); stopPlay(); upTo = Infinity; refresh();
  }

  /* ---------------- 정책 탭 ---------------- */

  function renderTabs() {
    elTabs.innerHTML = data.policy_order.map(function (n) {
      var d = data.policies[n], tr = data.traces[n];
      return '<button type="button" role="tab" class="tab' + (n === selPolicy ? " on" : "") +
        '" data-policy="' + n + '" aria-selected="' + (n === selPolicy) + '">' +
        "<b>" + n + "</b>" +
        "<span>총 " + d.total_cycles_with_memory + " cyc" +
        (d.memory_bound ? " (메모리 병목)" : "") + " · 연산 " + d.cycles +
        " · 워드 " + d.words_bram + "</span></button>";
    }).join("");

    var useful = data.traces[selPolicy].occupancy.slots_useful;
    var same = data.policy_order.every(function (n) {
      return data.traces[n].occupancy.slots_useful === useful;
    });
    elTabNote.innerHTML = TAGS[selPolicy] +
      (same ? " &nbsp;·&nbsp; <b>유효 슬롯 " + useful +
              "개는 네 정책이 모두 같다</b> — 해야 할 일의 양은 정책이 바꾸지 못한다." : "");
  }

  elTabs.addEventListener("click", function (e) {
    var t = e.target.closest(".tab");
    if (!t) return;
    selPolicy = t.dataset.policy;
    stopPlay(); upTo = Infinity;
    render();
  });

  /* ---------------- ★ 타임라인 ---------------- */

  function renderTimeline() {
    var sp = data.spec, tr = data.traces[selPolicy], occ = tr.occupancy;

    elTLStat.innerHTML =
      "<span>사이클 <b>" + tr.n_cycles + "</b></span>" +
      "<span>유효 슬롯 <b class='hi'>" + occ.slots_useful + "</b></span>" +
      "<span>끌려온 종단 <b>" + occ.slots_waste + "</b></span>" +
      "<span>빈 레인 <b>" + occ.slots_empty + "</b></span>" +
      "<span>레인 활용률 <b>" + pct(occ.utilization) + "</b></span>" +
      (upTo < tr.n_cycles ? "<span class='hi'>▶ " + (upTo + 1) + " / " + tr.n_cycles + "</span>" : "");

    if (tr.omitted) {
      elTL.innerHTML = '<div class="tl-omitted">설정이 커서 타임라인을 생략했습니다 (' +
        tr.n_cycles + " 사이클 × " + sp.lanes + " 레인). 토큰 수나 평면 수를 줄여 보라.</div>";
      return;
    }

    // 레인이 많아지면 칸을 줄이고, 너무 좁아지면 숫자를 감춘다 (색만으로도 읽힌다)
    var sw = Math.max(9, Math.min(38, Math.floor(880 / sp.lanes)));
    elTL.style.setProperty("--sw", sw + "px");
    elTL.classList.toggle("narrow", sw < 22);

    var html = "", groupKey = null, open = false;
    var m0 = sp.two_phase_split;

    tr.cycles.forEach(function (c) {
      var key = c.kind === "compaction"
        ? "comp@" + c.batch
        : "plane" + c.plane + (c.batch !== null && selPolicy === "batch" ? "/b" + c.batch : "");

      if (key !== groupKey) {
        if (open) html += "</div>";
        groupKey = key;
        var cls = "tl-group", head;
        if (c.kind === "compaction") {
          cls += " compact";
          head = "<b>압축</b><span>생존 토큰을 앞으로 당긴다 — " +
                 sp.compaction_cost_cycles + " 사이클</span>";
        } else {
          var isPhase1 = (selPolicy === "two_phase" && c.plane < m0);
          if (isPhase1) cls += " phase1";
          head = "<b>평면 b" + (sp.n_planes - 1 - c.plane) + "</b>" +
                 "<span>살아있는 토큰 " + data.per_plane[c.plane].live + " / " + sp.n_tokens + "</span>" +
                 (selPolicy === "batch" && c.batch !== null
                    ? '<span class="tag">배치 ' + c.batch + "</span>" : "") +
                 (isPhase1 ? '<span class="tag">1단계 · 전체 스캔</span>' : "");
        }
        html += '<div class="' + cls + '"><div class="tl-group-h">' + head + "</div>";
        open = true;
      }

      var future = c.index > upTo, now = c.index === upTo;

      if (c.kind === "compaction") {
        html += '<div class="tl-row comp' + (future ? " future" : "") + (now ? " now" : "") + '">' +
          '<span class="tl-cyc">' + c.index + "</span>" +
          '<span class="tl-slots">압축 오버헤드</span></div>';
        return;
      }

      var slots = "", allEmpty = true;
      c.slots.forEach(function (tok, i) {
        var cls2 = "slot ";
        if (tok === null) cls2 += "empty";
        else { allEmpty = false; cls2 += c.waste[i] ? "waste" : "live"; }
        if (tok !== null && tok === pickToken) cls2 += " pick";
        slots += '<span class="' + cls2 + '">' + (tok === null ? "" : tok) + "</span>";
      });

      html += '<div class="tl-row' + (future ? " future" : "") + (now ? " now" : "") + '">' +
        '<span class="tl-cyc">' + c.index + "</span>" +
        '<span class="tl-slots">' + slots + "</span>" +
        (allEmpty ? '<span class="tl-note">★ 토큰이 하나도 없는 사이클</span>' : "") +
        "</div>";
    });
    if (open) html += "</div>";
    elTL.innerHTML = html;
  }

  /* ---------------- 시간축 막대 (연산 vs 메모리) ---------------- */

  function renderTimebars() {
    var d = data.policies[selPolicy], sp = data.spec;
    var scale = Math.max(d.cycles, d.memory_cycles, d.total_cycles_with_memory, 1);

    function row(cls, label, value, tag, hot) {
      return '<div class="tbrow ' + cls + '"><span class="lab">' + label + "</span>" +
        '<span class="track"><i style="width:' + (100 * value / scale).toFixed(1) + '%"></i></span>' +
        '<span class="num">' + value + "</span>" +
        '<span class="tag' + (hot ? " hot" : "") + '">' + (tag || "") + "</span></div>";
    }

    var h = "";
    h += row("cyc", "연산", d.cycles, "레인 " + sp.lanes + "개", !d.memory_bound);
    h += row("mem", "메모리", d.memory_cycles,
             d.words_bram + " 워드 ÷ 포트 " + sp.n_ports, d.memory_bound);
    h += row("tot", "총", d.total_cycles_with_memory,
             sp.mem_overlap ? "max(연산, 메모리) — 완전 중첩 가정"
                            : "연산 + 메모리 — 중첩 없음 가정", false);

    h += '<div class="tb-note' + (d.memory_bound ? " hot" : "") + '">' +
      (d.memory_bound
        ? "★ 메모리가 병목이다 — 연산을 더 줄여도 전체 시간이 줄지 않는다. " +
          "포트를 늘리거나 word_tokens 를 키워야 벗어난다."
        : "연산이 병목이다 — BRAM 읽기는 " + d.memory_cycles +
          " 사이클로 연산 " + d.cycles + " 사이클 뒤에 숨는다.") +
      "</div>";

    document.getElementById("timebars").innerHTML = h;
  }

  /* ---------------- 재생 ---------------- */

  btnPlay.addEventListener("click", function () {
    if (playTimer) { stopPlay(); return; }
    var n = data.traces[selPolicy].n_cycles;
    if (!n) return;
    upTo = -1;
    btnPlay.textContent = "멈추기";
    btnPlay.setAttribute("aria-pressed", "true");
    playTimer = setInterval(function () {
      upTo++;
      if (upTo >= n - 1) { upTo = n - 1; stopPlay(); }
      renderTimeline();
    }, 320);
    renderTimeline();
  });

  function stopPlay() {
    if (!playTimer) return;
    clearInterval(playTimer);
    playTimer = null;
    btnPlay.textContent = "사이클 재생";
    btnPlay.setAttribute("aria-pressed", "false");
  }

  btnAll.addEventListener("click", function () {
    stopPlay(); upTo = Infinity; renderTimeline();
  });

  // mem_overlap 라벨은 체크 상태를 그대로 쓴다 (syncLabels 가 range 만 다룬다)

  /* ---------------- 토큰 수명 ---------------- */

  function renderLife() {
    var sp = data.spec, rl = data.read_live, tp = data.term_plane, h = "";
    for (var i = 0; i < sp.n_tokens; i++) {
      var bar = "";
      for (var t = 0; t < sp.n_planes; t++) {
        bar += '<span class="lifecell' + (rl[t][i] ? " live" : "") + '"></span>';
      }
      h += '<button type="button" class="liferow' + (i === pickToken ? " pick" : "") +
        '" data-token="' + i + '">' +
        '<span class="liferow-lab">t' + i + "</span>" +
        '<span class="lifebar">' + bar + "</span>" +
        '<span class="liferow-end">' + tp[i] + " / " + sp.n_planes + " 평면</span></button>";
    }
    elLife.innerHTML = h;
  }

  elLife.addEventListener("click", function (e) {
    var r = e.target.closest(".liferow");
    if (!r) return;
    var t = +r.dataset.token;
    pickToken = (pickToken === t) ? null : t;
    renderLife();
    renderTimeline();
  });

  /* ---------------- BRAM ---------------- */

  function renderBram() {
    var sp = data.spec, wt = sp.word_tokens;
    var nWords = Math.ceil(sp.n_tokens / wt);
    elBram.style.setProperty("--tc", Math.max(4, Math.min(12, Math.floor(560 / sp.n_tokens))) + "px");

    var h = "";
    for (var t = 0; t < sp.n_planes; t++) {
      var row = data.read_live[t], pp = data.per_plane[t], cells = "";
      for (var w = 0; w < nWords; w++) {
        var bits = "", any = false;
        for (var i = 0; i < wt; i++) {
          var idx = w * wt + i;
          if (idx >= sp.n_tokens) break;
          if (row[idx]) any = true;
          bits += '<span class="mapcell' + (row[idx] ? " live" : "") + '"></span>';
        }
        cells += '<span class="mapword' + (any ? " read" : "") + '">' + bits + "</span>";
      }
      var gain = pp.words_scattered - pp.words_compacted;
      h += '<button type="button" class="bramrow' + (t === selPlane ? " on" : "") +
        '" data-plane="' + t + '">' +
        '<span class="bramrow-lab">b' + (sp.n_planes - 1 - t) + "</span>" +
        '<span class="bramrow-cells">' + cells + "</span>" +
        '<span class="bramrow-stat">live ' + pp.live + " · 워드 " + pp.words_scattered +
        ' → <span class="' + (gain > 0 ? "good" : "") + '">' + pp.words_compacted +
        "</span></span></button>";
    }
    elBram.innerHTML = h;
  }

  elBram.addEventListener("click", function (e) {
    var r = e.target.closest(".bramrow");
    if (!r) return;
    selPlane = +r.dataset.plane;
    renderBram();
    renderHero();
  });

  function wordBlock(bits, isRead) {
    return '<div class="word ' + (isRead ? "read" : "skip") + '">' +
      '<div class="word-bits">' +
      bits.map(function (v) { return '<i class="' + (v ? "live" : "") + '"></i>'; }).join("") +
      '</div><div class="word-tag">' + (isRead ? "읽음" : "건너뜀") + "</div></div>";
  }

  function renderHero() {
    var sp = data.spec, pp = data.per_plane[selPlane], row = data.read_live[selPlane];
    var wt = sp.word_tokens, nWords = Math.ceil(sp.n_tokens / wt);
    elHero.style.setProperty("--bw", Math.max(3, Math.min(9, Math.floor(200 / wt))) + "px");

    var h = '<div class="hero-head">평면 <b>b' + (sp.n_planes - 1 - selPlane) + "</b>" +
      "<span>살아있는 토큰 " + pp.live + " / " + sp.n_tokens + "</span>" +
      "<span>워드 하나 = 토큰 " + wt + "개</span></div>";

    var a = "";
    for (var w = 0; w < nWords; w++) {
      var bits = [], any = false;
      for (var i = 0; i < wt; i++) {
        var idx = w * wt + i, v = idx < sp.n_tokens ? row[idx] : 0;
        bits.push(v);
        if (v) any = true;
      }
      a += wordBlock(bits, any);
    }
    h += '<div class="hero-row"><div class="hero-label"><b>① 있는 그대로</b> — none · batch' +
      '<span class="verdict ' + (pp.words_scattered >= nWords ? "bad" : "") + '">' +
      pp.words_scattered + " / " + nWords + " 워드</span></div>" +
      '<div class="words">' + a + "</div></div>";

    var b = "";
    for (var w2 = 0; w2 < nWords; w2++) {
      var bits2 = [], any2 = false;
      for (var i2 = 0; i2 < wt; i2++) {
        var idx2 = w2 * wt + i2, v2 = idx2 < pp.live ? 1 : 0;
        bits2.push(v2);
        if (v2) any2 = true;
      }
      b += wordBlock(bits2, any2);
    }
    var gain = pp.words_scattered - pp.words_compacted;
    h += '<div class="hero-row"><div class="hero-label"><b>② 앞으로 압축</b> — compaction · two_phase' +
      '<span class="verdict ' + (gain > 0 ? "good" : "") + '">' + pp.words_compacted +
      " / " + nWords + " 워드" + (gain > 0 ? " &nbsp;(" + gain + "개 절약)" : " &nbsp;(차이 없음)") +
      '</span></div><div class="words">' + b + "</div></div>";

    h += '<p class="sec-note" style="margin:0">진한 칸은 <b>' + pp.live +
      "개로 양쪽이 같다</b> — 논리적 읽기 수는 동일하다. " +
      (gain > 0
        ? "그런데 읽어야 하는 워드가 <b>" + pp.words_scattered + " → " + pp.words_compacted +
          "</b> 로 줄었다. <b>배치를 바꿔야만</b> 얻어지는 절감이다."
        : "이 평면은 이미 뭉쳐 있어서 압축해도 워드가 줄지 않는다.") + "</p>";

    elHero.innerHTML = h;
  }

  /* ---------------- 판정 상자 ---------------- */

  function renderVerdict() {
    var v = data.verdict, pol = data.policies;
    var el = document.getElementById("verdict");

    function axis(cls, label, name, extra) {
      return '<div class="vaxis ' + cls + '"><small>' + label + "</small><b>" + name +
             "</b>" + (extra ? "<small>" + extra + "</small>" : "") + "</div>";
    }

    var h = '<div class="verdict-sit">지금 설정 — <em>' + v.situation_label + "</em></div>";
    h += '<p class="verdict-why">' + v.why + "</p>";
    h += '<div class="verdict-axes">' +
      axis("rec", "이 상황의 권장", v.recommended, "") +
      axis("cyc", "사이클 최소", v.best_cycles, pol[v.best_cycles].cycles + " cyc") +
      axis("wrd", "BRAM 워드 최소", v.best_words,
           pol[v.best_words].words_bram + " word · −" + pct(pol[v.best_words].read_saving_bram)) +
      axis("tot", "총 시간 최소", v.best_total,
           pol[v.best_total].total_cycles_with_memory + " cyc") +
      axis("", "레인 활용 최고", v.best_utilization,
           pct(data.traces[v.best_utilization].occupancy.utilization)) +
      "</div>";

    if (v.any_memory_bound) {
      h += '<div class="verdict-split">⚠ 일부(또는 전부) 정책에서 <b>메모리가 병목</b>이다 — ' +
        "연산 사이클을 줄이는 정책을 골라도 시간이 줄지 않는다. " +
        "<b>총 시간 최소</b> 쪽을 보라.</div>";
    }

    if (!v.axes_agree) {
      h += '<div class="verdict-split">⚠ 두 축의 승자가 다르다 — <b>' + v.best_cycles +
        "</b> 는 사이클이, <b>" + v.best_words +
        "</b> 는 메모리가 유리하다. Decode 는 메모리 병목이라 보통 워드 쪽을 본다.</div>";
    }
    el.innerHTML = h;

    document.querySelectorAll(".nums.guide tbody tr").forEach(function (tr) {
      tr.classList.toggle("on", tr.dataset.sit === v.situation);
    });
  }

  /* ---------------- 요약표 ---------------- */

  function renderNums() {
    var order = data.policy_order, pol = data.policies, tra = data.traces, ref = data.reference;
    var rows = [
      ["사이클", function (n) { return pol[n].cycles; }],
      ["레인 활용률", function (n) { return pct(tra[n].occupancy.utilization); }],
      ["유효 슬롯", function (n) { return tra[n].occupancy.slots_useful; }],
      ["끌려온 종단 슬롯", function (n) { return tra[n].occupancy.slots_waste; }],
      ["빈 레인 슬롯", function (n) { return tra[n].occupancy.slots_empty; }],
      ["BRAM 워드 읽기", function (n) { return pol[n].words_bram; }],
      ["메모리 사이클", function (n) { return pol[n].memory_cycles; }],
      ["★ 총 사이클 (연산+메모리)", function (n) { return pol[n].total_cycles_with_memory; }],
      ["메모리 병목인가", function (n) { return pol[n].memory_bound ? "예" : "아니오"; }],
      ["★ 실현 절감 (워드)", function (n) { return pct(pol[n].read_saving_bram); }],
      ["파이프라인 효율", function (n) { return pct(pol[n].pipeline_efficiency); }]
    ];
    var refCol = {
      "사이클": ref.dense_cycles + " (dense)",
      "BRAM 워드 읽기": ref.words_dense + " (dense)"
    };

    var h = "<thead><tr><th>지표</th>" +
      order.map(function (n) { return "<th>" + n + "</th>"; }).join("") +
      "<th>기준</th></tr></thead><tbody>";
    rows.forEach(function (r) {
      h += '<tr><td class="k">' + r[0] + "</td>";
      order.forEach(function (n) {
        h += '<td class="' + (n === selPolicy ? "sel" : "") + '">' + r[1](n) + "</td>";
      });
      h += '<td class="k">' + (refCol[r[0]] || "—") + "</td></tr>";
    });
    h += '<tr><td class="k">이론 절감 (논리 읽기)</td><td colspan="' + (order.length + 1) +
      '" style="text-align:left" class="k">' + ref.reads_ideal + " / " + ref.reads_dense +
      " 쌍 = " + pct(1 - ref.reads_ideal / ref.reads_dense) +
      " &nbsp;— 어느 정책도 이걸 다 realize 하지는 못한다</td></tr></tbody>";
    elNums.innerHTML = h;
  }

  /* ---------------- 렌더 ---------------- */

  function render() {
    if (!data) return;
    renderTabs();
    renderTimeline();
    renderTimebars();
    renderVerdict();
    renderLife();
    renderBram();
    renderHero();
    renderNums();
  }

  syncLabels();
  refresh();
})();
