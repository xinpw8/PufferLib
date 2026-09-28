'use strict';
(function(root){
  const savedRekBindings = Object.freeze([
  {
    "commandId": "move:left_hook_processed",
    "category": 20,
    "codes": [
      "KeyI"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_jab_processed",
    "category": 21,
    "codes": [
      "KeyK"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:double_uppercut_processed",
    "category": 22,
    "codes": [
      "Space",
      "KeyJ"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:right_hook_processed",
    "category": 23,
    "codes": [
      "KeyO"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:right_jab_processed",
    "category": 24,
    "codes": [
      "KeyL"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_jab_right_uppercut_processed",
    "category": 25,
    "codes": [
      "Space",
      "KeyL"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_side_kick_processed",
    "category": 16,
    "codes": [
      "KeyY"
    ],
    "doubleTap": true
  },
  {
    "commandId": "move:left_front_kick_processed",
    "category": 17,
    "codes": [
      "KeyH"
    ],
    "doubleTap": true
  },
  {
    "commandId": "move:right_side_kick_processed",
    "category": 18,
    "codes": [
      "KeyU"
    ],
    "doubleTap": true
  },
  {
    "commandId": "move:right_knee_processed",
    "category": 19,
    "codes": [
      "KeyJ"
    ],
    "doubleTap": true
  },
  {
    "commandId": "move:6_punch_processed",
    "category": 26,
    "codes": [
      "Space",
      "KeyY"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:run_and_punch_processed",
    "category": 27,
    "codes": [
      "Space",
      "KeyU"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_right_jab_processed",
    "category": 28,
    "codes": [
      "Semicolon"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_right_hook_processed",
    "category": 29,
    "codes": [
      "Quote"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:left_hook_right_jab_processed",
    "category": 30,
    "codes": [
      "Space",
      "KeyK"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:double_hook_processed",
    "category": 31,
    "codes": [
      "Space",
      "KeyH"
    ],
    "doubleTap": false
  },
  {
    "commandId": "move:butt_smack_emote_processed",
    "category": 32,
    "codes": [
      "Space",
      "KeyI"
    ],
    "doubleTap": false
  }
].map(row => Object.freeze({...row, codes: Object.freeze(row.codes)})));
  class SavedRekKeyboard {
    constructor(windowMs = 300) { this.windowMs = windowMs; this.reset(); }
    reset() { this.down = new Set(); this.pending = null; }
    release(code) { this.down.delete(code); }
    recognizes(code) { return savedRekBindings.some(row => row.codes.includes(code)); }
    press(code, now, repeat = false) {
      if (repeat || this.down.has(code)) return null;
      this.down.add(code);
      if (this.pending && (now < this.pending.time || now - this.pending.time > this.windowMs)) this.pending = null;
      // Longest held chord wins; saved row order breaks equal-length ties.
      const row = savedRekBindings.filter(binding => binding.codes.includes(code)
        && binding.codes.every(key => this.down.has(key)))
        .sort((a, b) => b.codes.length - a.codes.length)[0];
      if (!row) return null;
      const previous = this.pending;
      this.pending = null;
      if (!row.doubleTap) return row.category;
      if (previous && previous.category === row.category && now - previous.time <= this.windowMs)
        return row.category;
      // The saved four double-tap rows have no single-tap fallback binding.
      this.pending = {category: row.category, time: now};
      return null;
    }
  }

 const api={SavedRekKeyboard,savedRekBindings};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.RekControls=api;
})(globalThis);
