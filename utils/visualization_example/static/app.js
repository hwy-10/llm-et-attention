/* 행렬 곱 시각화 — 화면 쪽.
 *
 * ★ 이 파일은 산술을 하지 않는다.
 *   곱·합·누적은 전부 utils/visualization_example/matmul.py 가 계산해서
 *   JSON 으로 보내 준다. 여기서 다시 계산하면 두 구현이 조용히 갈라질 수 있고,
 *   그러면 "화면의 숫자가 파이썬이 계산한 값"이라는 보장이 깨진다.
 *
 *   여기가 하는 일은 셋뿐이다 — 그리기, 클릭 받기, 서버에 물어보기.
 */

(function () {
  "use strict";

  var LIMITS = { m: [2, 12], n: [2, 12], p: [2, 12] };

  var A = [], B = [], C = [];
  var dims = { m: 5, n: 4, p: 3 };
  var cellInfo = null;            // 서버가 준 C[i][j] 분해
  var selR = null, selC = null, selK = null;
  var editMode = false, playTimer = null;
  var reqSeq = 0, editTimer = null;

  var elA = document.getElementById("matA");
  var elB = document.getElementById("matB");
  var elC = document.getElementById("matC");
  var elCalc = document.getElementById("calc");
  var elBanner = document.getElementById("banner");
  var board = document.getElementById("board");
  var svg = document.getElementById("guides");
  var NS = "http://www.w3.org/2000/svg";

  var cells = { a: [], b: [], c: [] };
  var labs = { a: { top: [], side: [] }, b: { top: [], side: [] }, c: { top: [], side: [] } };

  /* ------------------------------------------------------------------ *
   * 서버 통신
   * ------------------------------------------------------------------ */

  function post(path, payload) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (res) {
      return res.json().then(function (data) {
        if (!res.ok) throw new Error(data && data.error ? data.error : "HTTP " + res.status);
        return data;
      });
    });
  }

  function fail(msg) {
    elBanner.hidden = false;
    elBanner.textContent = "서버 오류 — " + msg;
  }
  function ok() { elBanner.hidden = true; }

  /** 현재 A·B·선택을 서버에 보내고 C 와 분해를 받아 온다. */
  function sync() {
    var mine = ++reqSeq;
    return post("/api/compute", { A: A, B: B, i: selR, j: selC })
      .then(function (st) {
        if (mine !== reqSeq) return;          // 늦게 온 응답은 버린다
        adopt(st);
        ok();
        render();
      })
      .catch(function (e) { fail(e.message); });
  }

  /** 서버가 준 상태를 그대로 받아들인다. */
  function adopt(st) {
    A = st.A; B = st.B; C = st.C;
    dims = st.dims;
    cellInfo = st.cell;
    document.getElementById("backend").textContent = "계산 백엔드 " + st.backend;
  }

  function requestRandom(m, n, p) {
    return post("/api/random", { m: m, n: n, p: p });
  }

  /* ------------------------------------------------------------------ *
   * DOM 생성
   * ------------------------------------------------------------------ */

  function buildGrid(el, rows, cols, tag, rowSym, colSym, getVal) {
    el.innerHTML = "";
    el.style.gridTemplateColumns = "var(--gut) repeat(" + cols + ", var(--cell))";
    cells[tag] = [];
    labs[tag] = { top: [], side: [] };

    var corner = document.createElement("div");
    corner.className = "lab corner";
    corner.textContent = rowSym + "\\" + colSym;
    el.appendChild(corner);

    for (var c = 0; c < cols; c++) {
      var lt = document.createElement("button");
      lt.type = "button";
      lt.className = "lab top";
      lt.textContent = c + 1;
      lt.dataset.m = tag; lt.dataset.axis = "col"; lt.dataset.i = c;
      lt.setAttribute("aria-label", tag.toUpperCase() + " " + colSym + "=" + (c + 1) + " 열 선택");
      el.appendChild(lt);
      labs[tag].top.push(lt);
    }

    for (var r = 0; r < rows; r++) {
      var ls = document.createElement("button");
      ls.type = "button";
      ls.className = "lab side";
      ls.textContent = r + 1;
      ls.dataset.m = tag; ls.dataset.axis = "row"; ls.dataset.i = r;
      ls.setAttribute("aria-label", tag.toUpperCase() + " " + rowSym + "=" + (r + 1) + " 행 선택");
      el.appendChild(ls);
      labs[tag].side.push(ls);

      cells[tag][r] = [];
      for (var c2 = 0; c2 < cols; c2++) {
        var cell = document.createElement("button");
        cell.type = "button";
        cell.className = "cell";
        cell.dataset.m = tag; cell.dataset.r = r; cell.dataset.c = c2;
        cell.textContent = getVal(r, c2);
        cell.setAttribute("aria-label", tag.toUpperCase() + " " + (r + 1) + "행 " + (c2 + 1) + "열");
        el.appendChild(cell);
        cells[tag][r].push(cell);
      }
    }
  }

  /** 행렬 크기가 바뀌었을 때만 격자를 다시 만든다. */
  function rebuildIfNeeded() {
    var needA = cells.a.length !== dims.m || (cells.a[0] || []).length !== dims.n;
    var needB = cells.b.length !== dims.n || (cells.b[0] || []).length !== dims.p;
    var needC = cells.c.length !== dims.m || (cells.c[0] || []).length !== dims.p;
    if (!(needA || needB || needC)) return false;

    buildGrid(elA, dims.m, dims.n, "a", "i", "k", function (r, c) { return A[r][c]; });
    buildGrid(elB, dims.n, dims.p, "b", "k", "j", function (r, c) { return B[r][c]; });
    buildGrid(elC, dims.m, dims.p, "c", "i", "j", function (r, c) { return C[r][c]; });
    document.getElementById("dimA").textContent = "(" + dims.m + " × " + dims.n + ")";
    document.getElementById("dimB").textContent = "(" + dims.n + " × " + dims.p + ")";
    document.getElementById("dimC").textContent = "(" + dims.m + " × " + dims.p + ")";
    return true;
  }

  function applyEditMode() {
    ["a", "b"].forEach(function (tag) {
      var src = tag === "a" ? A : B;
      cells[tag].forEach(function (row, r) {
        row.forEach(function (cell, c) {
          if (editMode) {
            var inp = cell.querySelector("input");
            if (inp) { if (document.activeElement !== inp) inp.value = src[r][c]; return; }
            cell.innerHTML = "";
            inp = document.createElement("input");
            inp.type = "number";
            inp.value = src[r][c];
            inp.setAttribute("aria-label", tag.toUpperCase() + " " + (r + 1) + "행 " + (c + 1) + "열 값");
            cell.appendChild(inp);
          } else {
            cell.textContent = src[r][c];
          }
        });
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * 그리기
   * ------------------------------------------------------------------ */

  function render() {
    rebuildIfNeeded();
    applyEditMode();

    var active = (selR !== null && selC !== null);

    cells.a.forEach(function (row, r) {
      row.forEach(function (cell, c) {
        cell.className = "cell";
        if (selR === r) {
          cell.classList.add("hl-row");
          if (active && selK === c) cell.classList.add("hl-term");
        }
      });
    });
    cells.b.forEach(function (row, r) {
      row.forEach(function (cell, c) {
        cell.className = "cell";
        if (selC === c) {
          cell.classList.add("hl-col");
          if (active && selK === r) cell.classList.add("hl-term");
        }
      });
    });
    cells.c.forEach(function (row, r) {
      row.forEach(function (cell, c) {
        cell.className = "cell";
        cell.textContent = C[r][c];          // ← 서버가 계산한 값
        if (selR === r && selC === c) cell.classList.add("hl-prod");
        else if (selR === r) cell.classList.add("hl-row-faint");
        else if (selC === c) cell.classList.add("hl-col-faint");
      });
    });

    ["a", "b", "c"].forEach(function (t) {
      labs[t].top.forEach(function (l) { l.className = "lab top"; });
      labs[t].side.forEach(function (l) { l.className = "lab side"; });
    });
    if (selR !== null) {
      if (labs.a.side[selR]) labs.a.side[selR].classList.add("on-row");
      if (labs.c.side[selR]) labs.c.side[selR].classList.add("on-row");
    }
    if (selC !== null) {
      if (labs.b.top[selC]) labs.b.top[selC].classList.add("on-col");
      if (labs.c.top[selC]) labs.c.top[selC].classList.add("on-col");
    }
    if (selK !== null && active) {
      if (labs.a.top[selK]) labs.a.top[selK].classList.add("on-k");
      if (labs.b.side[selK]) labs.b.side[selK].classList.add("on-k");
    }

    renderCalc();
    drawGuides();
  }

  /* ---- 연결 레일 ---------------------------------------------------- */

  function roundPath(pts, r) {
    if (pts.length < 3) return "M" + pts[0][0] + " " + pts[0][1] + " L" + pts[1][0] + " " + pts[1][1];
    var d = "M" + pts[0][0].toFixed(1) + " " + pts[0][1].toFixed(1);
    for (var i = 1; i < pts.length - 1; i++) {
      var p0 = pts[i - 1], p1 = pts[i], p2 = pts[i + 1];
      var l1 = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
      var l2 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
      if (l1 < 0.5 || l2 < 0.5) continue;
      var rr = Math.min(r, l1 / 2, l2 / 2);
      var a = [p1[0] - (p1[0] - p0[0]) / l1 * rr, p1[1] - (p1[1] - p0[1]) / l1 * rr];
      var b = [p1[0] + (p2[0] - p1[0]) / l2 * rr, p1[1] + (p2[1] - p1[1]) / l2 * rr];
      d += " L" + a[0].toFixed(1) + " " + a[1].toFixed(1) +
           " Q" + p1[0].toFixed(1) + " " + p1[1].toFixed(1) +
           " " + b[0].toFixed(1) + " " + b[1].toFixed(1);
    }
    var last = pts[pts.length - 1];
    return d + " L" + last[0].toFixed(1) + " " + last[1].toFixed(1);
  }

  function drawGuides() {
    svg.innerHTML = "";
    var br = board.getBoundingClientRect();
    if (!br.width) return;
    svg.setAttribute("width", br.width);
    svg.setAttribute("height", br.height);
    svg.setAttribute("viewBox", "0 0 " + br.width + " " + br.height);

    function rel(el) {
      var r = el.getBoundingClientRect();
      return {
        l: r.left - br.left, r: r.right - br.left,
        t: r.top - br.top, b: r.bottom - br.top,
        cx: (r.left + r.right) / 2 - br.left,
        cy: (r.top + r.bottom) / 2 - br.top
      };
    }
    function add(tag, attrs) {
      var n = document.createElementNS(NS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      svg.appendChild(n);
    }

    // 행 레일 — A 의 행 → 아래로 우회 → C 의 행 (B 를 가로지르지 않는다)
    if (selR !== null && cells.a[selR] && cells.c[selR]) {
      var aEnd = rel(cells.a[selR][dims.n - 1]);
      var cIn = rel(cells.c[selR][0]);
      var yBus = br.height - 16;
      add("path", {
        d: roundPath([[aEnd.r + 2, aEnd.cy], [aEnd.r + 14, aEnd.cy], [aEnd.r + 14, yBus],
                      [cIn.l - 16, yBus], [cIn.l - 16, cIn.cy], [cIn.l - 9, cIn.cy]], 9),
        fill: "none", stroke: "var(--row)", "stroke-width": 2,
        "stroke-linecap": "round", opacity: 0.85
      });
      add("polygon", {
        points: (cIn.l - 9) + "," + (cIn.cy - 4.5) + " " + (cIn.l - 1) + "," + cIn.cy +
                " " + (cIn.l - 9) + "," + (cIn.cy + 4.5),
        fill: "var(--row)", opacity: 0.85
      });
      add("circle", { cx: aEnd.r + 2, cy: aEnd.cy, r: 2.5, fill: "var(--row)", opacity: 0.85 });
    }

    // 열 레일 — B 의 열 → 위로 우회 → C 의 열
    if (selC !== null && cells.b[0] && cells.c[0]) {
      var bTop = rel(cells.b[0][selC]);
      var cTop = rel(cells.c[0][selC]);
      add("path", {
        d: roundPath([[bTop.cx, bTop.t - 2], [bTop.cx, 14], [cTop.cx, 14], [cTop.cx, cTop.t - 9]], 9),
        fill: "none", stroke: "var(--col)", "stroke-width": 2,
        "stroke-linecap": "round", opacity: 0.85
      });
      add("polygon", {
        points: (cTop.cx - 4.5) + "," + (cTop.t - 9) + " " + cTop.cx + "," + (cTop.t - 1) +
                " " + (cTop.cx + 4.5) + "," + (cTop.t - 9),
        fill: "var(--col)", opacity: 0.85
      });
      add("circle", { cx: bTop.cx, cy: bTop.t - 2, r: 2.5, fill: "var(--col)", opacity: 0.85 });
    }
  }

  /* ---- 계산판 -------------------------------------------------------- */

  function renderCalc() {
    if (!cellInfo) {
      var msg = (selR === null && selC === null)
        ? "A에서 행을, B에서 열을 하나씩 골라 보라. C의 칸을 바로 눌러도 되고, 거터의 번호를 눌러도 된다."
        : (selR === null ? "B의 열을 골랐다. 이제 A에서 행을 하나 고르면 결과 칸이 정해진다."
                         : "A의 행을 골랐다. 이제 B에서 열을 하나 고르면 결과 칸이 정해진다.");
      elCalc.innerHTML = '<div class="empty"><span>' + msg + "</span></div>";
      return;
    }

    var i = cellInfo.i, j = cellInfo.j, terms = cellInfo.terms;
    var h = "";
    h += '<div class="calc-head">';
    h += '<span class="calc-title">C[' + (i + 1) + "][" + (j + 1) + "] = " + cellInfo.total + "</span>";
    h += '<span class="calc-sigma">= Σ<sub>k=1..' + terms.length + "</sub> A[" + (i + 1) +
         "][k] · B[k][" + (j + 1) + "]</span>";
    h += '<span class="badge">' +
         (cellInfo.cross_checked ? "numpy A @ B 와 대조 완료" : "순수 파이썬 (교차 검증 없음)") +
         "</span>";
    h += "</div>";

    h += '<div class="vec"><div class="vec-labels">' +
         "<span>k</span>" +
         '<span class="l-a">A[' + (i + 1) + "][k]</span>" +
         '<span class="l-b">B[k][' + (j + 1) + "]</span>" +
         '<span class="l-p">곱</span>' +
         "<span>누적</span>" +
         "</div>";

    var parts = [];
    terms.forEach(function (t, idx) {
      parts.push(t.prod < 0 ? "(" + t.prod + ")" : String(t.prod));
      var dimCls = (selK !== null && idx > selK) ? " dim" : "";
      h += '<button type="button" class="vcol' + (selK === idx ? " on" : "") + dimCls +
           '" data-k="' + idx + '">' +
           '<span class="v-k">' + (idx + 1) + "</span>" +
           '<span class="v-a">' + t.a + "</span>" +
           '<span class="v-b">' + t.b + "</span>" +
           '<span class="v-p">' + t.prod + "</span>" +
           '<span class="v-s">' + t.cumsum + "</span>" +
           "</button>";
    });
    h += "</div>";

    h += '<div class="total">';
    h += '<span class="sum">' + parts.join(" + ") + "</span><span>=</span>";
    h += '<span class="final">' + cellInfo.total + "</span>";
    if (selK !== null && terms[selK]) {
      h += '<span class="run">k = ' + (selK + 1) + " 까지 &nbsp;" + terms[selK].cumsum + "</span>";
    }
    h += "</div>";

    elCalc.innerHTML = h;
  }

  /* ------------------------------------------------------------------ *
   * 상호작용
   * ------------------------------------------------------------------ */

  [elA, elB, elC].forEach(function (el) {
    el.addEventListener("click", function (e) {
      if (e.target.tagName === "INPUT") return;

      var lab = e.target.closest(".lab");
      if (lab && !lab.classList.contains("corner")) {
        stopPlay();
        var t = lab.dataset.m, ax = lab.dataset.axis, idx = +lab.dataset.i;
        if (ax === "row") { if (t === "a" || t === "c") selR = idx; else selK = idx; }
        else { if (t === "b" || t === "c") selC = idx; else selK = idx; }
        sync();
        return;
      }

      var cell = e.target.closest(".cell");
      if (!cell) return;
      stopPlay();
      var tag = cell.dataset.m, r = +cell.dataset.r, c = +cell.dataset.c;
      if (tag === "a") { selR = r; selK = c; }
      else if (tag === "b") { selC = c; selK = r; }
      else { selR = r; selC = c; selK = null; }
      sync();
    });

    // 값 편집 — 타이핑이 멈추면 서버에 다시 물어본다
    el.addEventListener("input", function (e) {
      if (e.target.tagName !== "INPUT") return;
      var cell = e.target.closest(".cell");
      var v = parseInt(e.target.value, 10);
      if (isNaN(v)) return;
      var r = +cell.dataset.r, c = +cell.dataset.c;
      if (cell.dataset.m === "a") A[r][c] = v; else B[r][c] = v;
      clearTimeout(editTimer);
      editTimer = setTimeout(sync, 160);
    });

    el.addEventListener("keydown", function (e) {
      var cell = e.target.closest(".cell");
      if (!cell) return;
      var map = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] };
      var d = map[e.key];
      if (!d) return;
      var t = cell.dataset.m, r = +cell.dataset.r + d[0], c = +cell.dataset.c + d[1];
      if (!cells[t][r] || !cells[t][r][c]) return;
      e.preventDefault();
      var nx = cells[t][r][c];
      (nx.querySelector("input") || nx).focus();
    });
  });

  // 항 선택은 순전히 표시용이라 서버에 물어볼 필요가 없다
  elCalc.addEventListener("click", function (e) {
    var v = e.target.closest(".vcol");
    if (!v) return;
    stopPlay();
    selK = +v.dataset.k;
    render();
  });

  /* ---- 컨트롤 -------------------------------------------------------- */

  document.querySelectorAll(".stepper").forEach(function (st) {
    st.addEventListener("click", function (e) {
      var btn = e.target.closest("button");
      if (!btn) return;
      var dim = st.dataset.dim, lim = LIMITS[dim];
      var nx = Math.min(lim[1], Math.max(lim[0], dims[dim] + (+btn.dataset.d)));
      if (nx === dims[dim]) return;
      stopPlay();
      applyDims(dim === "m" ? nx : dims.m, dim === "n" ? nx : dims.n, dim === "p" ? nx : dims.p);
    });
  });

  /** 크기를 바꾸되 겹치는 부분의 값은 지킨다. 새 칸은 서버가 뽑아 준다. */
  function applyDims(m, n, p) {
    var oldA = A, oldB = B;
    requestRandom(m, n, p).then(function (st) {
      for (var r = 0; r < m; r++)
        for (var c = 0; c < n; c++)
          if (oldA[r] && oldA[r][c] !== undefined) st.A[r][c] = oldA[r][c];
      for (var k = 0; k < n; k++)
        for (var j = 0; j < p; j++)
          if (oldB[k] && oldB[k][j] !== undefined) st.B[k][j] = oldB[k][j];

      A = st.A; B = st.B;
      dims = { m: m, n: n, p: p };
      if (selR !== null && selR >= m) selR = null;
      if (selC !== null && selC >= p) selC = null;
      if (selK !== null && selK >= n) selK = null;
      syncSteppers();
      sync();
    }).catch(function (e) { fail(e.message); });
  }

  function syncSteppers() {
    ["m", "n", "p"].forEach(function (d) {
      document.getElementById("val" + d).textContent = dims[d];
      var st = document.querySelector('.stepper[data-dim="' + d + '"]');
      st.querySelector('[data-d="-1"]').disabled = dims[d] <= LIMITS[d][0];
      st.querySelector('[data-d="1"]').disabled = dims[d] >= LIMITS[d][1];
    });
  }

  document.getElementById("btnRand").addEventListener("click", function () {
    stopPlay();
    requestRandom(dims.m, dims.n, dims.p).then(function (st) {
      A = st.A; B = st.B;
      sync();
    }).catch(function (e) { fail(e.message); });
  });

  var btnEdit = document.getElementById("btnEdit");
  btnEdit.addEventListener("click", function () {
    editMode = !editMode;
    btnEdit.setAttribute("aria-pressed", String(editMode));
    btnEdit.textContent = editMode ? "입력 끝내기" : "값 직접 입력";
    if (!editMode) {
      ["a", "b"].forEach(function (t) {
        cells[t].forEach(function (row) { row.forEach(function (cell) { cell.innerHTML = ""; }); });
      });
    }
    render();
  });

  var btnPlay = document.getElementById("btnPlay");
  btnPlay.addEventListener("click", function () {
    if (playTimer) { stopPlay(); return; }
    var start = (selR === null || selC === null);
    if (selR === null) selR = 0;
    if (selC === null) selC = 0;
    selK = -1;
    btnPlay.textContent = "멈추기";
    btnPlay.setAttribute("aria-pressed", "true");
    playTimer = setInterval(function () {
      selK++;
      if (!cellInfo || selK >= cellInfo.terms.length) { stopPlay(); return; }
      render();
    }, 750);
    if (start) sync(); else render();
  });

  function stopPlay() {
    if (!playTimer) return;
    clearInterval(playTimer);
    playTimer = null;
    btnPlay.textContent = "항별로 재생";
    btnPlay.setAttribute("aria-pressed", "false");
  }

  /* ------------------------------------------------------------------ *
   * 시작
   * ------------------------------------------------------------------ */

  requestRandom(dims.m, dims.n, dims.p)
    .then(function (st) {
      adopt(st);
      selR = 1; selC = 2;             // 2행 · 3열 을 처음부터 보여 준다
      syncSteppers();
      return sync();
    })
    .catch(function (e) {
      fail(e.message + " — 서버가 떠 있는지 확인해 보라 (python -m utils.visualization_example)");
    });

  window.addEventListener("resize", drawGuides);
  if (window.ResizeObserver) new ResizeObserver(drawGuides).observe(board);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawGuides);
})();
