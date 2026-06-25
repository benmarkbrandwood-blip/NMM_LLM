/**
 * sounds.js — board sound effects for Nine Men's Morris.
 *
 * Sounds are served from /sounds/ and selected randomly per event type.
 * All sounds are lazily loaded (no preload) — the browser caches them after
 * the first play so subsequent plays are instant.
 */

const SOUND_FILES = {
  place:  ["place1","place2","place3","place4","place5","place6","place7","place8"],
  move:   ["move1","move2","move3","move4","move6","move8"],
  remove: ["remove1","remove2","remove3","remove5","remove6","remove7"],
};

let _muted = false;

export function setMuted(val) { _muted = val; }
export function getMuted()    { return _muted; }

export function playSound(type) {
  if (_muted) return;
  const pool = SOUND_FILES[type];
  if (!pool) return;
  const name = pool[Math.floor(Math.random() * pool.length)];
  const audio = new Audio(`/sounds/${name}.mp3`);
  audio.volume = 0.7;
  audio.play().catch(() => {});  // ignore autoplay policy rejections silently
}
