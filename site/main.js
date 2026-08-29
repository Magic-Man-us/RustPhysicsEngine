/* rust_physics_engine site behaviour.
   No framework and no build step. Five independent pieces, each of which
   no-ops when the elements it wants are not on the page. */

(function () {
  'use strict';

  /* ------------------------------------------------------------ theme */

  var root = document.documentElement;
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.dataset.theme === 'light' ? 'dark' : 'light';
      root.dataset.theme = next;
      try { localStorage.setItem('rpe-theme', next); } catch (e) {}
    });
  }

  /* -------------------------------------------------------- highlight */

  // A token pass rather than a parser. It runs over the already-escaped
  // text of a <code> block, and the alternation is ordered so that a
  // keyword inside a comment or a string is never reached: whichever
  // branch starts first wins the character range.
  var GRAMMAR = {
    rust: {
      kw: /\b(as|async|await|break|const|continue|crate|dyn|else|enum|extern|false|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|true|type|unsafe|use|where|while)\b/,
      ty: /\b(f64|f32|u8|u16|u32|u64|usize|i8|i16|i32|i64|isize|bool|str|String|Vec|Option|Result|Some|None|Ok|Err|Box|[A-Z][A-Za-z0-9_]*)\b/,
      com: /\/\/[^\n]*/,
      str: /"(?:[^"\\\n]|\\.)*"/
    },
    python: {
      kw: /\b(and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b/,
      ty: /\b(int|float|str|bool|list|dict|tuple|set|complex|print|len|range|abs)\b/,
      com: /#[^\n]*/,
      str: /(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/
    },
    toml: {
      kw: /^\s*\[[^\]\n]*\]/m,
      ty: /^\s*[A-Za-z0-9_.-]+(?=\s*=)/m,
      com: /#[^\n]*/,
      str: /"(?:[^"\\\n]|\\.)*"/
    },
    console: {
      kw: /^\s*\$/m,
      ty: /(?!x)x/,
      com: /(?!x)x/,
      str: /"(?:[^"\\\n]|\\.)*"/
    }
  };
  GRAMMAR.bash = GRAMMAR.console;
  GRAMMAR.sh = GRAMMAR.console;
  GRAMMAR.py = GRAMMAR.python;
  GRAMMAR.rs = GRAMMAR.rust;

  var NUM = /\b(?:0[xXbo][0-9a-fA-F_]+|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?)\b/;
  var MACRO = /\b[a-z_][a-z0-9_]*!/;
  var CALL = /\b[a-z_][A-Za-z0-9_]*(?=\()/;

  function source(re) { return re.source; }

  function highlight(el) {
    var lang = (el.className.match(/lang-([a-z0-9]+)/) || [])[1];
    var g = GRAMMAR[lang];
    if (!g) return;
    var re = new RegExp([
      '(' + source(g.com) + ')',
      '(' + source(g.str) + ')',
      '(' + source(MACRO) + ')',
      '(' + source(g.kw) + ')',
      '(' + source(g.ty) + ')',
      '(' + source(NUM) + ')',
      '(' + source(CALL) + ')'
    ].join('|'), 'gm');
    var classes = ['tok-com', 'tok-str', 'tok-mac', 'tok-kw', 'tok-typ', 'tok-num', 'tok-fn'];
    var text = el.textContent;
    var out = '';
    var last = 0;
    text.replace(re, function (match) {
      var index = arguments[arguments.length - 2];
      var groups = Array.prototype.slice.call(arguments, 1, 1 + classes.length);
      var cls = '';
      for (var i = 0; i < groups.length; i++) {
        if (groups[i] !== undefined) { cls = classes[i]; break; }
      }
      out += escapeHtml(text.slice(last, index));
      out += '<span class="' + cls + '">' + escapeHtml(match) + '</span>';
      last = index + match.length;
      return match;
    });
    out += escapeHtml(text.slice(last));
    el.innerHTML = out;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>]/g, function (c) {
      return c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;';
    });
  }

  Array.prototype.forEach.call(document.querySelectorAll('pre > code'), highlight);

  /* ------------------------------------------------------------- copy */

  document.addEventListener('click', function (event) {
    var button = event.target.closest && event.target.closest('.copy');
    if (!button) return;
    var block = button.parentElement.querySelector('code');
    if (!block) return;
    var restore = function () {
      button.textContent = 'Copy';
      button.classList.remove('done');
    };
    var done = function () {
      button.textContent = 'Copied';
      button.classList.add('done');
      setTimeout(restore, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(block.textContent).then(done, function () {
        button.textContent = 'Press ⌘C';
      });
    } else {
      var area = document.createElement('textarea');
      area.value = block.textContent;
      document.body.appendChild(area);
      area.select();
      try { document.execCommand('copy'); done(); } catch (e) {}
      document.body.removeChild(area);
    }
  });

  /* ------------------------------------------------------------- tabs */

  Array.prototype.forEach.call(document.querySelectorAll('[data-tabs]'), function (group) {
    var tabs = group.querySelectorAll('.tab');
    Array.prototype.forEach.call(tabs, function (tab) {
      tab.addEventListener('click', function () {
        Array.prototype.forEach.call(tabs, function (other) {
          var on = other === tab;
          other.setAttribute('aria-selected', on ? 'true' : 'false');
          var panel = document.getElementById(other.getAttribute('aria-controls'));
          if (panel) panel.hidden = !on;
        });
      });
    });
  });

  /* ---------------------------------------------------------- modules */

  var grid = document.getElementById('module-grid');
  if (grid && window.MODULES) {
    var search = document.getElementById('module-search');
    var chipRow = document.getElementById('module-chips');
    var counter = document.getElementById('module-count');
    var area = 'All';

    (window.MODULE_AREAS || []).forEach(function (name) {
      var chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.textContent = name;
      chip.setAttribute('aria-pressed', 'false');
      chipRow.appendChild(chip);
    });

    chipRow.addEventListener('click', function (event) {
      var chip = event.target.closest('.chip');
      if (!chip) return;
      area = chip.getAttribute('aria-pressed') === 'true' ? 'All' : chip.textContent;
      Array.prototype.forEach.call(chipRow.querySelectorAll('.chip'), function (other) {
        other.setAttribute('aria-pressed', other.textContent === area ? 'true' : 'false');
      });
      render();
    });

    if (search) {
      search.addEventListener('input', render);
      // "/" focuses the filter, the way the rustdoc search does.
      document.addEventListener('keydown', function (event) {
        if (event.key === '/' && document.activeElement !== search &&
            !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
          event.preventDefault();
          search.focus();
        }
      });
    }

    function render() {
      var query = (search ? search.value : '').trim().toLowerCase();
      var shown = window.MODULES.filter(function (m) {
        if (area !== 'All' && m.area !== area) return false;
        if (!query) return true;
        return (m.name + ' ' + m.area + ' ' + m.summary).toLowerCase().indexOf(query) >= 0;
      });
      grid.innerHTML = shown.length ? shown.map(card).join('') :
        '<p class="empty">No module matches that. Try a subject: ' +
        '<code>wavelet</code>, <code>orbit</code>, <code>prime</code>.</p>';
      if (counter) {
        counter.textContent = shown.length === window.MODULES.length
          ? window.MODULES.length + ' modules'
          : shown.length + ' of ' + window.MODULES.length + ' modules';
      }
    }

    function card(m) {
      return '<a class="mod" href="api/rust_physics_engine/' + m.name + '/index.html">' +
        '<span class="mod-name">' + m.name +
        '<span class="mod-area">' + esc(m.area) + '</span></span>' +
        '<span class="mod-sum">' + esc(m.summary) + '</span>' +
        '<span class="mod-nums">' +
        '<span>' + plural(m.fns, 'fn') + '</span>' +
        '<span>' + plural(m.types, 'type') + '</span>' +
        '<span>' + plural(m.lines, 'line') + '</span>' +
        '</span></a>';
    }

    function esc(s) { return escapeHtml(String(s)); }

    function plural(n, word) {
      return n.toLocaleString() + ' ' + word + (n === 1 ? '' : 's');
    }

    render();
  }

  /* --------------------------------------------------------- fill-ins */

  // Figures on the front page come from the same generated data as the
  // module map, so a new module changes them without anyone editing HTML.
  if (window.CRATE_TOTALS) {
    Array.prototype.forEach.call(document.querySelectorAll('[data-total]'), function (el) {
      var value = window.CRATE_TOTALS[el.getAttribute('data-total')];
      if (typeof value === 'number') el.textContent = value.toLocaleString();
    });
  }

  /* ----------------------------------------------------------- scroll */

  var tocLinks = document.querySelectorAll('.toc nav a');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    var targets = [];
    Array.prototype.forEach.call(tocLinks, function (link) {
      var id = decodeURIComponent(link.getAttribute('href').slice(1));
      var heading = document.getElementById(id);
      if (heading) { byId[id] = link; targets.push(heading); }
    });
    var visible = [];
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var id = entry.target.id;
        var at = visible.indexOf(id);
        if (entry.isIntersecting && at < 0) visible.push(id);
        if (!entry.isIntersecting && at >= 0) visible.splice(at, 1);
      });
      var first = targets.filter(function (t) { return visible.indexOf(t.id) >= 0; })[0];
      Array.prototype.forEach.call(tocLinks, function (l) { l.classList.remove('current'); });
      if (first && byId[first.id]) byId[first.id].classList.add('current');
    }, { rootMargin: '-80px 0px -70% 0px' });
    targets.forEach(function (t) { observer.observe(t); });
  }
})();
