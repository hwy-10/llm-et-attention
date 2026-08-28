/* 함수 해부도 — 화면 쪽.
 *
 * ★ 여기는 코드를 읽지 않는다.
 *   상자의 이름·타입·필드·코드 인용은 전부
 *   utils/visualization_example/anatomy.py 가 inspect 로 실제 소스에서 뽑아
 *   보내 준 것이다. 이 파일은 그걸 상자로 옮기고 클릭을 처리할 뿐이다.
 */

(function () {
  "use strict";

  var idx = null;      // 모듈 목록
  var spec = null;     // 지금 펼친 함수의 해부도
  var picked = null;   // 지금 고른 상자 {kind, id}

  var $ = function (id) { return document.getElementById(id); };
  var elBanner = $("banner"), elList = $("fnlist"), elSec = $("anatomySec");
  var elDetail = $("detail"), elDrift = $("drift");

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  /** 아주 작은 인라인 마크다운: `코드` 와 **굵게** 만 */
  function md(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }

  function fail(msg) {
    elBanner.hidden = false;
    elBanner.textContent = "서버 오류 — " + msg;
  }

  function get(url) {
    return fetch(url).then(function (res) {
      return res.json().then(function (d) {
        if (!res.ok) throw new Error(d && d.error ? d.error : "HTTP " + res.status);
        return d;
      });
    });
  }

  /* ---------------- 함수 목록 ---------------- */

  function renderList() {
    $("modLine").textContent = idx.module + " · " + idx.n_lines + "줄";

    elList.innerHTML = idx.items.map(function (it) {
      var on = spec && it.name === spec.func;
      return '<button type="button" class="fnbox' + (on ? " on" : "") +
        (it.implemented ? "" : " off") + '" data-fn="' + it.name +
        '"' + (it.implemented ? "" : " disabled") + ">" +
        '<span class="fnbox-top">' +
        '<span class="fnbox-name">' + esc(it.name) + "</span>" +
        '<span class="fnbox-kind">' + it.kind + "</span>" +
        '<span class="fnbox-ln">L' + it.lineno + "</span></span>" +
        '<span class="fnbox-sig">' + esc(it.signature) + "</span>" +
        (it.blurb ? '<span class="fnbox-blurb">' + esc(it.blurb) + "</span>" : "") +
        (it.implemented ? "" : '<span class="fnbox-todo">아직 펼치지 않음</span>') +
        "</button>";
    }).join("");
  }

  elList.addEventListener("click", function (e) {
    var b = e.target.closest(".fnbox");
    if (!b || b.disabled) return;
    open(b.dataset.fn);
  });

  function open(fn) {
    get("/api/anatomy?module=schedule&func=" + encodeURIComponent(fn))
      .then(function (d) {
        spec = d;
        picked = null;
        elBanner.hidden = true;
        renderList();
        renderFlow();
        renderDetail();
        renderDrift();
        elSec.hidden = false;
      })
      .catch(function (e) { fail(e.message); });
  }

  /* ---------------- 흐름도 ---------------- */

  function box(kind, id, name, type, sub, count) {
    var on = picked && picked.kind === kind && picked.id === id;
    return '<button type="button" class="box ' + kind + (on ? " on" : "") +
      '" data-kind="' + kind + '" data-id="' + esc(id) + '">' +
      '<span class="box-top"><span class="box-name">' + esc(name) + "</span>" +
      (type ? '<span class="box-type">' + esc(type) + "</span>" : "") + "</span>" +
      (sub ? '<span class="box-sub">' + md(sub) + "</span>" : "") +
      (count ? '<span class="box-count">' + esc(count) + "</span>" : "") +
      "</button>";
  }

  function renderFlow() {
    $("fnTitle").textContent = spec.module.replace("src/", "") + " · " + spec.signature;
    $("fnMeta").textContent = "L" + spec.lineno + " · " + spec.n_lines + "줄";
    $("fnDoc").textContent = spec.doc.split("\n")[0];

    $("colIn").innerHTML = spec.inputs.map(function (i) {
      return box("in", i.name, i.name, i.type, i.desc,
        i.fields.length ? "필드 " + i.fields.length + "개" :
        (i.choices ? i.choices.join(" · ") : (i.default ? "기본값 " + i.default : "")));
    }).join("");

    $("colCore").innerHTML = spec.core.map(function (s) {
      return box("core", s.id, s.title, "", "", "");
    }).join("");

    $("colBranch").innerHTML = spec.branches.map(function (b) {
      return box("branch", b.policy, b.policy, "", b.title, "");
    }).join("");

    $("colOut").innerHTML = spec.outputs.map(function (o) {
      return box("out", o.name, o.name, o.type, o.desc, "필드 " + o.fields.length + "개");
    }).join("");
  }

  document.addEventListener("click", function (e) {
    var b = e.target.closest(".box");
    if (!b) return;
    var kind = b.dataset.kind, id = b.dataset.id;
    if (picked && picked.kind === kind && picked.id === id) picked = null;
    else picked = { kind: kind, id: id };
    renderFlow();
    renderDetail();
    document.getElementById("detail").scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  /* ---------------- 상세 ---------------- */

  function fieldTable(fields, owner) {
    var rows = fields.map(function (f) {
      var cls = [];
      if (f.desc && f.desc.indexOf("★") === 0) cls.push("star");
      if (f.desc && f.desc.indexOf("읽지 않는다") >= 0) cls.push("unused");
      if (!f.documented) cls.push("nodoc");
      return '<tr class="' + cls.join(" ") + '">' +
        '<td class="fname">' + esc(f.name) + "</td>" +
        '<td class="ftype">' + esc(f.type) + (f.default ? " = " + esc(f.default) : "") + "</td>" +
        '<td class="fdesc">' + (f.documented ? md(f.desc) : "설명 없음 — anatomy.py 에 추가할 것") +
        "</td></tr>";
    }).join("");
    return '<table class="ftable"><thead><tr><th>필드</th><th>타입</th><th>' +
      esc(owner) + " 에서의 역할</th></tr></thead><tbody>" + rows + "</tbody></table>";
  }

  function renderDetail() {
    if (!picked) {
      elDetail.className = "detail";
      elDetail.innerHTML = '<div class="detail-empty">위의 <b>상자를 눌러</b> 내부 동작을 펼쳐 보라.</div>';
      return;
    }
    var h = "";
    elDetail.className = "detail k-" + picked.kind;

    if (picked.kind === "in" || picked.kind === "out") {
      var list = picked.kind === "in" ? spec.inputs : spec.outputs;
      var item = list.filter(function (x) { return x.name === picked.id; })[0];
      if (!item) return;
      h += '<div class="detail-h"><b>' + esc(item.name) + "</b>" +
        '<span class="tag">' + (picked.kind === "in" ? "입력" : "출력") + "</span>" +
        '<span class="tag">' + esc(item.type) + "</span></div>";
      h += '<p class="detail-desc">' + md(item.desc || "") + "</p>";
      if (item.choices) {
        h += "<h4>가능한 값</h4><div class=\"choices\">" +
          item.choices.map(function (c) { return "<span>" + esc(c) + "</span>"; }).join("") + "</div>";
      }
      if (item.fields && item.fields.length) {
        h += "<h4>필드</h4>" + fieldTable(item.fields, item.type);
      }

    } else if (picked.kind === "core") {
      var st = spec.core.filter(function (x) { return x.id === picked.id; })[0];
      if (!st) return;
      h += '<div class="detail-h"><b>' + esc(st.title) + "</b>" +
        '<span class="tag">핵심 동작</span></div>';
      h += '<p class="detail-desc">' + md(st.detail) + "</p>";
      h += "<h4>코드</h4><pre>" + esc(st.code) + "</pre>";

    } else {  // branch
      var br = spec.branches.filter(function (x) { return x.policy === picked.id; })[0];
      if (!br) return;
      h += '<div class="detail-h"><b>' + esc(br.policy) + "</b>" +
        '<span class="tag">정책 분기</span></div>';
      h += '<p class="detail-desc"><b>' + esc(br.title) + "</b></p>";
      h += "<h4>무엇을 하나</h4><p class=\"detail-desc\">" + md(br.what) + "</p>";
      h += "<h4>왜 이렇게 하나</h4><p class=\"detail-desc\">" + md(br.why) + "</p>";
      h += "<h4>원본 코드 (inspect.getsource 로 잘라 온 것)</h4><pre>" + esc(br.code) + "</pre>";
    }

    elDetail.innerHTML = h;
  }

  /* ---------------- 어긋남 경고 ---------------- */

  function renderDrift() {
    if (!spec.undocumented || !spec.undocumented.length) {
      elDrift.hidden = true;
      return;
    }
    elDrift.hidden = false;
    elDrift.innerHTML = "⚠ <b>코드에 있는데 설명이 없는 항목 " + spec.undocumented.length +
      "개</b> — 필드가 추가되었는데 <code>anatomy.py</code> 의 설명이 따라가지 못했다: " +
      spec.undocumented.map(function (n) { return "<code>" + esc(n) + "</code>"; }).join(", ");
  }

  /* ---------------- 시작 ---------------- */

  get("/api/anatomy?module=schedule")
    .then(function (d) {
      idx = d;
      renderList();
      open("apply");          // 시범 구현된 것을 바로 펼쳐 준다
    })
    .catch(function (e) {
      fail(e.message + " — 서버가 떠 있는지 확인해 보라 (python -m utils.visualization_example)");
    });
})();
