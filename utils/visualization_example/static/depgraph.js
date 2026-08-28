/* 코드 상관관계 — 그리기만 한다. 좌표와 그래프는 서버가 준다. */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

let G = null;                       // 서버가 준 그래프
let pick = null;                    // 지금 고른 파일 id
let hidden = new Set();             // 범례에서 끈 구역

// ---------------------------------------------------------------------------
function el(tag, attrs, text) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}

function fail(msg) {
  const b = $("banner");
  b.textContent = msg;
  b.hidden = false;
}

function scopeVar(s) { return `var(--sc-${s})`; }
function scopeSoft(s) { return `var(--sc-${s}-soft)`; }

// ---------------------------------------------------------------------------
// 요약
// ---------------------------------------------------------------------------
function drawSummary() {
  const s = G.summary;
  const card = (label, value, sub, warn) =>
    `<div class="card"><dt>${label}</dt>` +
    `<dd class="${warn ? "warn" : ""}">${value}<small>${sub}</small></dd></div>`;

  const untested = s.untested.length;
  $("summary").innerHTML =
    card("담당 파일", s.our_files, `${s.our_lines.toLocaleString()} 줄`) +
    card("그린 파일", G.nodes.length, "담당 + 맞닿은 것") +
    card("import 간선", s.edges, `그중 브리지 ${s.bridge_edges}`) +
    card("검사가 없는 담당 파일", untested,
         untested ? s.untested.join(" · ") : "없다", untested > 0);
}

// ---------------------------------------------------------------------------
// 범례
// ---------------------------------------------------------------------------
function drawLegend() {
  const count = {};
  G.nodes.forEach((n) => { count[n.scope] = (count[n.scope] || 0) + 1; });

  const wrap = $("legend");
  wrap.innerHTML = "";
  Object.keys(G.scopes).forEach((key) => {
    if (!count[key]) return;
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", hidden.has(key) ? "false" : "true");
    b.innerHTML =
      `<span class="sw" style="background:${scopeSoft(key)};border-color:${scopeVar(key)}"></span>` +
      `<span>${G.scopes[key].label}</span><span class="n">${count[key]}</span>`;
    b.addEventListener("click", () => {
      if (hidden.has(key)) hidden.delete(key); else hidden.add(key);
      render();
    });
    wrap.appendChild(b);
  });
}

// ---------------------------------------------------------------------------
// 그래프
// ---------------------------------------------------------------------------
function visible(n) { return !hidden.has(n.scope); }

function related(id) {
  const set = new Set([id]);
  G.edges.forEach((e) => {
    if (e.source === id) set.add(e.target);
    if (e.target === id) set.add(e.source);
  });
  return set;
}

function drawGraph() {
  const svg = $("graph");
  svg.innerHTML = "";
  const W = G.layout.width, H = G.layout.height;
  const { w: NW, h: NH } = G.node_size;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);

  const defs = el("defs");
  [["ar", "var(--ink-faint)"], ["arl", "var(--ink)"], ["arb", "var(--warn)"]].forEach(([id, c]) => {
    const m = el("marker", {
      id, viewBox: "0 0 10 8", refX: "9", refY: "4",
      markerWidth: "8", markerHeight: "6", orient: "auto",
    });
    m.appendChild(el("path", { d: "M0,0 L10,4 L0,8 z", fill: c }));
    defs.appendChild(m);
  });
  svg.appendChild(defs);

  const shown = new Set(G.nodes.filter(visible).map((n) => n.id));
  const near = pick ? related(pick) : null;
  const byId = {};
  G.nodes.forEach((n) => { byId[n.id] = n; });

  // 열 머리말
  const cols = {};
  G.nodes.forEach((n) => { if (shown.has(n.id)) cols[n.col] = n.x; });
  Object.keys(cols).forEach((c) => {
    svg.appendChild(el("text", { x: cols[c], y: 26, class: "col-tag" }, `깊이 ${c}`));
    svg.appendChild(el("line", {
      x1: cols[c], y1: 36, x2: Number(cols[c]) + NW, y2: 36, class: "col-rule",
    }));
  });

  // 간선 — 왼쪽(의존 대상)에서 오른쪽(쓰는 쪽)으로 그린다
  const showNames = $("showNames").checked;
  G.edges.forEach((e) => {
    if (!shown.has(e.source) || !shown.has(e.target)) return;
    const from = byId[e.target], to = byId[e.source];
    const x1 = from.x + NW, y1 = from.y + NH / 2;
    const x2 = to.x, y2 = to.y + NH / 2;
    const mid = (x2 - x1) * 0.5;

    const lit = near && near.has(e.source) && near.has(e.target)
                && (e.source === pick || e.target === pick);
    const cls = ["edge"];
    if (e.crosses_scope) cls.push("cross");
    if (e.bridge) cls.push("bridge");
    if (near) cls.push(lit ? "lit" : "dim");

    const marker = lit ? "arl" : (e.bridge ? "arb" : "ar");
    svg.appendChild(el("path", {
      d: `M${x1},${y1} C${x1 + mid},${y1} ${x2 - mid},${y2} ${x2},${y2}`,
      class: cls.join(" "),
      "marker-end": `url(#${marker})`,
    }));

    if (showNames && e.names.length && (!near || lit)) {
      const names = e.names.slice(0, 2).join(", ") + (e.names.length > 2 ? " …" : "");
      const t = el("text", {
        x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 5,
        "text-anchor": "middle",
        class: "elabel" + (near ? (lit ? " lit" : " dim") : ""),
      }, (e.bridge ? "★ " : "") + names);
      svg.appendChild(t);
    }
  });

  // 노드
  G.nodes.forEach((n) => {
    if (!shown.has(n.id)) return;
    const g = el("g", { class: "node" + (near && !near.has(n.id) ? " dim" : "")
                        + (n.id === pick ? " pick" : "") });
    g.appendChild(el("rect", {
      class: "box", x: n.x, y: n.y, width: NW, height: NH,
      fill: scopeSoft(n.scope), stroke: scopeVar(n.scope),
    }));
    g.appendChild(el("text", { class: "name", x: n.x + 10, y: n.y + 20 }, n.label));
    g.appendChild(el("text", { class: "meta", x: n.x + 10, y: n.y + 36 },
      `${n.dir} · ${n.lines}줄 · in ${n.fan_in} / out ${n.fan_out}`));
    g.addEventListener("click", () => { pick = (pick === n.id ? null : n.id); render(); });
    const title = el("title");
    title.textContent = `${n.id}\n${n.role || "(역할 미기재)"}`;
    g.appendChild(title);
    svg.appendChild(g);
  });
}

// ---------------------------------------------------------------------------
// 상세
// ---------------------------------------------------------------------------
function drawDetail() {
  const box = $("detail");
  if (!pick) {
    box.innerHTML = '<p class="empty">상자를 누르면 그 파일이 무엇을 부르고 ' +
      '무엇에 쓰이는지, 어떤 검사가 덮고 있는지 보여 준다.</p>';
    return;
  }
  const n = G.nodes.find((v) => v.id === pick);
  const uses = G.edges.filter((e) => e.source === pick);
  const usedBy = G.edges.filter((e) => e.target === pick);

  const li = (e, other) =>
    `<li class="${e.bridge ? "bridge" : ""}">${e.bridge ? "★ " : ""}${other}` +
    `<span class="via"> — ${e.names.slice(0, 3).join(", ") || "모듈"}` +
    `${e.names.length > 3 ? " …" : ""}</span></li>`;

  const list = (arr, key) => arr.length
    ? "<ul>" + arr.map((e) => li(e, e[key])).join("") + "</ul>"
    : '<p class="none">없다</p>';

  box.innerHTML =
    `<h3>${n.id}</h3>` +
    `<span class="badge" style="color:${scopeVar(n.scope)};border-color:${scopeVar(n.scope)};` +
    `background:${scopeSoft(n.scope)}">${G.scopes[n.scope].label}</span>` +
    (n.role ? `<p class="role">${n.role}</p>` : "") +
    `<dl><dt>줄 수</dt><dd>${n.lines}</dd>` +
    `<dt>의존 깊이</dt><dd>${n.level}</dd>` +
    `<dt>부르는 / 쓰이는</dt><dd>${n.fan_out} / ${n.fan_in}</dd></dl>` +
    `<h4>이 파일이 부른다</h4>${list(uses, "target")}` +
    `<h4>이 파일을 부른다</h4>${list(usedBy, "source")}` +
    `<h4>덮는 검사</h4>` +
    (n.tests.length
      ? "<ul>" + n.tests.map((t) => `<li>${t}</li>`).join("") + "</ul>"
      : '<p class="none">없다 — 이 파일을 직접 임포트하는 검사가 없다</p>');
}

// ---------------------------------------------------------------------------
// 목록
// ---------------------------------------------------------------------------
function drawTable() {
  const rows = G.nodes.slice().sort((a, b) => {
    const o = { core: 0, extra: 1, integration: 2, bridge: 3, other: 4, test: 5 };
    return o[a.scope] - o[b.scope] || a.id.localeCompare(b.id);
  });
  $("filelist").innerHTML =
    "<thead><tr><th>파일</th><th>구역</th><th>줄</th><th>깊이</th>" +
    "<th>부른다</th><th>쓰인다</th><th>검사</th><th>역할</th></tr></thead><tbody>" +
    rows.map((n) =>
      `<tr data-id="${n.id}" class="${n.id === pick ? "pick" : ""}">` +
      `<td><span class="dot" style="background:${scopeVar(n.scope)}"></span>${n.id}</td>` +
      `<td>${G.scopes[n.scope].short}</td><td>${n.lines}</td><td>${n.level}</td>` +
      `<td>${n.fan_out}</td><td>${n.fan_in}</td>` +
      `<td class="${n.tests.length ? "" : "zero"}">${n.tests.length}</td>` +
      `<td class="role">${n.role || ""}</td></tr>`).join("") +
    "</tbody>";

  $("filelist").querySelectorAll("tbody tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const id = tr.getAttribute("data-id");
      pick = (pick === id ? null : id);
      render();
      document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

// ---------------------------------------------------------------------------
function render() {
  drawLegend();
  drawGraph();
  drawDetail();
  drawTable();
}

async function boot() {
  try {
    const res = await fetch("/api/depgraph");
    if (!res.ok) throw new Error(`서버가 ${res.status} 를 돌려주었습니다`);
    G = await res.json();
  } catch (err) {
    return fail(`그래프를 불러오지 못했습니다: ${err.message}`);
  }
  drawSummary();
  render();
  $("reset").addEventListener("click", () => { pick = null; hidden.clear(); render(); });
  $("showNames").addEventListener("change", render);
}

boot();
