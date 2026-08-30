"""Tick-level trace format for REK and for any candidate clone.

One format, written by both sides, so a recorded REK trajectory and a replayed
clone trajectory are directly comparable. Stdlib only, because the recorder has
to run on the machine with the game on it.

Layout:

    magic   b'REKTRACE\\0'
    uint32  format version
    uint32  header length, then that many bytes of UTF-8 JSON
    frames  repeated: uint64 tick, float64 * len(channels)
    uint32  event-block length, then that many bytes of UTF-8 JSON
    uint64  frame count

The frame count is a footer rather than part of the header because the writer
streams: it does not know how many ticks there will be until the trace ends.
Without it the reader has to guess where the frames stop and the event block
starts, which it cannot do reliably.

The header declares the channels. This module deliberately does not define what
they are: the recorder names what it actually captured, and a channel that was
not captured is simply absent rather than present and zero. See README.md for
the channel-naming convention the recorder should follow.

Two things every trace must carry, and the writer refuses without them:
`build_fingerprint`, from inventory.py, and `source`, either 'rek' or the name
of the clone. A trace that cannot say which build it came from cannot support a
parity claim.

A third applies when the simulation is not local. The client fingerprint pins
the client only; a server can be updated independently, so two traces from one
client build may come from different authoritative simulators. A trace declared
`authority='server'` must therefore also carry a server identity — endpoint and
session at minimum, plus protocol and server-reported version where the
handshake exposes them. The writer refuses without it.
"""

import json
import struct
from pathlib import Path

MAGIC = b'REKTRACE\0'
VERSION = 1

_HDR = struct.Struct('<I')
_TICK = struct.Struct('<Q')


class TraceWriter:
    """Streams frames to disk. Use as a context manager."""

    def __init__(self, path, channels, build_fingerprint, source,
                 authority='unknown', server=None, **meta):
        if not build_fingerprint:
            raise ValueError('build_fingerprint is required: a trace that cannot '
                             'name its build cannot support a parity claim')
        if source not in ('rek',) and not str(source).startswith('clone:'):
            raise ValueError("source must be 'rek' or 'clone:<name>'")
        if authority not in ('local', 'server', 'unknown'):
            raise ValueError("authority must be 'local', 'server' or 'unknown'")
        if authority == 'server':
            # The client hash does not pin the simulator that produced this.
            server = server or {}
            missing = [k for k in ('endpoint', 'session_id') if not server.get(k)]
            if missing:
                raise ValueError(
                    'a server-authoritative trace must identify the server: '
                    f'missing {missing}. The client build fingerprint does not '
                    'pin the authoritative simulator, so without this two traces '
                    'from one client may come from different server builds.')
        if not channels:
            raise ValueError('declare at least one channel')
        if len(set(channels)) != len(channels):
            raise ValueError('duplicate channel names')

        self.channels = list(channels)
        self.header = dict(meta)
        self.header.update({
            'version': VERSION,
            'channels': self.channels,
            'build_fingerprint': build_fingerprint,
            'source': source,
            'authority': authority,
            'server': server or {},
        })
        self._frame = struct.Struct('<%dd' % len(self.channels))
        self._events = []
        self._last_tick = None
        self._frames = 0
        self._f = open(path, 'wb')
        blob = json.dumps(self.header, sort_keys=True).encode()
        self._f.write(MAGIC)
        self._f.write(_HDR.pack(VERSION))
        self._f.write(_HDR.pack(len(blob)))
        self._f.write(blob)

    def append(self, tick, values):
        """One simulation tick. `values` is a dict keyed by channel name."""
        if self._last_tick is not None and tick <= self._last_tick:
            raise ValueError(f'ticks must strictly increase: {tick} after {self._last_tick}')
        missing = [c for c in self.channels if c not in values]
        if missing:
            raise ValueError(f'frame {tick} is missing channels: {missing[:8]}')
        self._last_tick = tick
        self._f.write(_TICK.pack(tick))
        self._f.write(self._frame.pack(*[float(values[c]) for c in self.channels]))
        self._frames += 1

    def event(self, tick, kind, **payload):
        """A discrete occurrence: a hit, a score change, a fall, a round end."""
        self._events.append({'tick': int(tick), 'kind': str(kind), **payload})

    def close(self):
        if self._f is None:
            return
        blob = json.dumps(self._events).encode()
        self._f.write(_HDR.pack(len(blob)))
        self._f.write(blob)
        self._f.write(_TICK.pack(self._frames))
        self._f.close()
        self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Trace:
    """A loaded trace. `channels` maps name -> list of values, tick-aligned."""

    def __init__(self, header, ticks, columns, events):
        self.header = header
        self.ticks = ticks
        self.channels = columns
        self.events = events

    @property
    def source(self):
        return self.header.get('source')

    @property
    def build_fingerprint(self):
        return self.header.get('build_fingerprint')

    @property
    def authority(self):
        return self.header.get('authority', 'unknown')

    @property
    def server(self):
        return self.header.get('server') or {}

    def __len__(self):
        return len(self.ticks)

    @classmethod
    def load(cls, path):
        data = Path(path).read_bytes()
        if not data.startswith(MAGIC):
            raise ValueError(f'{path} is not a REK trace')
        off = len(MAGIC)
        version, = _HDR.unpack_from(data, off); off += _HDR.size
        if version != VERSION:
            raise ValueError(f'{path} is trace version {version}, this reader is {VERSION}')
        hlen, = _HDR.unpack_from(data, off); off += _HDR.size
        header = json.loads(data[off:off + hlen]); off += hlen

        channels = header['channels']
        frame = struct.Struct('<%dd' % len(channels))
        stride = _TICK.size + frame.size

        if len(data) < off + _TICK.size:
            raise ValueError(f'{path} is truncated: no frame-count footer')
        n_frames, = _TICK.unpack_from(data, len(data) - _TICK.size)
        if off + n_frames * stride + _HDR.size > len(data) - _TICK.size:
            raise ValueError(f'{path} is truncated: footer claims {n_frames} frames')

        ticks, columns = [], {c: [] for c in channels}
        for _ in range(n_frames):
            tick, = _TICK.unpack_from(data, off)
            vals = frame.unpack_from(data, off + _TICK.size)
            ticks.append(tick)
            for name, v in zip(channels, vals):
                columns[name].append(v)
            off += stride

        elen, = _HDR.unpack_from(data, off); off += _HDR.size
        events = json.loads(data[off:off + elen]) if elen else []
        return cls(header, ticks, columns, events)
