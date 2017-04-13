window.addEventListener('load', function () {
    // Put each <h1> and subsequent content into its own <section>.
    function makeSections(node) {
        var sections = [];
        while (true) {
            sections.push(document.createElement('section'));
            var section = sections[sections.length - 1];
            var sib;
            while (true) {
                sib = node.nextSibling;
                if (sib === null) {
                    section.appendChild(node);
                    return sections;
                }
                if (sib.nodeName === 'H1') {
                    break;
                }
                section.appendChild(node);
                node = sib;
            }
            node = sib;
        }
    }

    var title = document.getElementsByClassName('title')[0];
    [].forEach.call(makeSections(title), function (x) {
        document.body.appendChild(x);
    });

    /* page number */
    var pagenr = document.createElement('span');
    pagenr.id = "pagenr";
    pagenr.textContent = "page number";
    document.body.appendChild(pagenr);

    var sections = document.getElementsByTagName('section');
    var current = 0;
    function update() {
        [].forEach.call(sections, function (x, i) {
            x.className = (i === 0) ? 'title-slide' : '';
        });
        if (current < 0)
            current = 0;
        if (current >= sections.length)
            current = sections.length - 1;
        sections[current].className += ' current';

        if (current == 0) {
            pagenr.textContent = '';
        } else {
            pagenr.textContent = '' + current;
        }

    }

    update();

    document.body.addEventListener('keydown', function (ev) {
        switch (ev.keyCode) {
            case 39: current++; break;
            case 37: current--; break;
        }
        update();
    });
    
    // by huonw
    // https://github.com/kmcallister/sliderust/pull/2

    // Touch listeners, to change page if a user with a touch devices
    // swipes left or right.
    var start_x, start_y;
    document.body.addEventListener('touchstart', function(ev) {
        ev.preventDefault();
        if (ev.touches.length > 1) return;
        start_x = ev.touches[0].clientX;
        start_y = ev.touches[0].clientY;
    });
    document.body.addEventListener('touchmove', function(ev) { ev.preventDefault(); });
    document.body.addEventListener('touchend', function(ev) {
        if (ev.touches.length > 0) return;

        var dx = ev.changedTouches[0].clientX - start_x;
        var dy = ev.changedTouches[0].clientY - start_y;

        // if the touch is at least 40% of the page wide, and doesn't
        // move vertically too much, it counts as a swipe.
        if (Math.abs(dx) > 0.4 * window.innerWidth && Math.abs(dy) < 0.2 * window.innerHeight) {
            current += -Math.sign(dx);
            update();
        }
    });

});

// countdown from http://codepen.io/SitePoint/pen/MwNPVq
// MIT licensed, by SitePoint
function getTimeRemaining(endtime) {
  var t = Date.parse(endtime) - Date.parse(new Date());
  var seconds = Math.floor((t / 1000) % 60);
  var minutes = Math.floor((t / 1000 / 60) % 60);
  var hours = Math.floor((t / (1000 * 60 * 60)) % 24);
  var days = Math.floor(t / (1000 * 60 * 60 * 24));
  return {
    'total': t,
    'days': days,
    'hours': hours,
    'minutes': minutes,
    'seconds': seconds
  };
}

function initializeClock(id, endtime) {
  var clock = document.getElementById(id);
  var daysSpan = clock.querySelector('.days');
  var hoursSpan = clock.querySelector('.hours');
  var minutesSpan = clock.querySelector('.minutes');
  var secondsSpan = clock.querySelector('.seconds');

  function updateClock() {
    var t = getTimeRemaining(endtime);

    clock.innerHTML = (t.days + " days "  + ('0' + t.hours).slice(-2) + "h " + ('0' + t.minutes).slice(-2) + "m " + ('0' + t.seconds).slice(-2)) + "s";

    if (t.total <= 0) {
      clearInterval(timeinterval);
    }
  }

  updateClock();
  var timeinterval = setInterval(updateClock, 1000);
}

window.addEventListener('load', function () {
	var deadline = new Date(Date.parse("2016-04-13"));
	initializeClock('stableclock', deadline);
});
