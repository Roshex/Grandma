"""
parallel.py - the one place GRANDMA talks to multiprocessing.

Why this module exists
----------------------
Both `ops.collapse_groups` and `reconcile.recon_all` fan a large, INVARIANT payload
(the gene trees, the name registry) out to workers that each handle one MUL-tree. How
that payload should travel depends entirely on the start method, and getting it wrong
costs more than everything the algorithms save:

  fork (Linux / macOS / HPC)
      Children inherit the parent's memory. Registering the payload in a module global
      BEFORE the pool is created costs nothing at all - no pickling, no copy. The pool
      must be re-created when the payload changes, which is cheap (a fork is ~1 ms).

  spawn (Windows)
      Every worker is a fresh interpreter that re-imports the package, and anything
      passed through `initargs` is pickled once PER WORKER, sequentially, in the parent
      before any worker starts - the classic "nothing happens for a minute, then the
      progress bar flies" symptom. Here the payload is written ONCE to a single file and
      every task carries a tiny (token, path) key; the workers load it themselves, in
      parallel, and cache it. Because the pool then needs no initializer it is
      payload-independent, so it is created once and reused for the whole run and the
      interpreter start-up is paid once instead of once per stage per iteration.

Two further traps this module closes:
  * `Pool.imap_unordered` pickles the CALLABLE once per task, so binding data into a
    functools.partial re-serialises that data for every single task. Task payloads here
    are always tiny: a state key plus one item.
  * Python 3.14 changes the POSIX default start method away from fork. The context is
    pinned explicitly so that change cannot silently turn a free transfer into a
    per-worker pickle.

Usage
-----
    from .parallel import WorkPool

    def _task(state, item):              # MUST be module level (picklable by reference)
        return heavy(state['trees'], item)

    pool = WorkPool(n_procs, spill_dir=tcf.pickle_dir, logger=logger)
    for result in pool.map_unordered(_task, items, state={'trees': trees},
                                     desc="# Working ", unit="mt"):
        consume(result)
    ...
    pool.cleanup()                       # once, at the end of the run

The same WorkPool instance can be shared by ops, reconcile and main; passing the same
`state` object twice reuses everything, passing a new one re-publishes only the payload.
"""

import os
import sys
import pickle
import tempfile
import multiprocessing as mp
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:                                   # progress bars are optional
    def tqdm(it, **kwargs):
        return it


# --------------------------------------------------------------------------
# START METHOD
# --------------------------------------------------------------------------

_USE_FORK = (not sys.platform.startswith('win')) and hasattr(os, 'fork')

StateKey = Tuple[int, Optional[str]]
_NO_STATE: StateKey = (0, None)


def mp_context():
    """'spawn' on Windows, 'fork' everywhere it exists. Pinned, never inherited."""
    if not _USE_FORK:
        return mp.get_context('spawn')
    try:
        return mp.get_context('fork')
    except ValueError:                                # platform without fork
        return mp.get_context('spawn')


# --------------------------------------------------------------------------
# STATE RESOLUTION (runs in the worker)
# --------------------------------------------------------------------------

_STATE_STORE: Dict[int, Any] = {}                     # parent, and inherited by fork children
_STATE_CACHE: "OrderedDict[int, Any]" = OrderedDict()  # spawn children, loaded from file
_STATE_CACHE_MAX = 2                                  # keep the current + previous payload


def _resolve_state(key: StateKey) -> Any:
    token, path = key
    if token == 0:
        return None
    state = _STATE_STORE.get(token)
    if state is not None:
        return state
    state = _STATE_CACHE.get(token)
    if state is None:
        if path is None:
            raise RuntimeError(
                f"Worker state {token} is neither inherited nor on disk. Under 'spawn' "
                f"the state must be published with a spill file (see WorkPool.publish).")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        _STATE_CACHE[token] = state
        while len(_STATE_CACHE) > _STATE_CACHE_MAX:
            _STATE_CACHE.popitem(last=False)
    return state


def _apply(func: Callable[[Any, Any], Any], task: Tuple[StateKey, Any]) -> Any:
    """Generic worker entry point. `func` is pickled by reference (module + qualname),
    so the per-task payload stays at a few dozen bytes."""
    key, item = task
    return func(_resolve_state(key), item)


# --------------------------------------------------------------------------
# THE POOL
# --------------------------------------------------------------------------

class WorkPool:
    """
    A reusable, start-method-aware worker pool.

    Not thread-safe: one instance per thread, or serialise access. Safe to construct
    even when n_procs == 1 (everything then runs in-process, through the same code path
    so the two branches cannot drift).

    GRANDMA has three task shapes, all of which fit this interface unchanged:
      * collapse   one task per MUL-tree; state = gene trees + registry + switches
      * reconcile  one task per MUL-tree; state = flat gene trees + weights + registry
      * sweep      one task per DONOR CLADE; state = flat gene trees + weights + st_flat
    In every case the invariant payload is the gene-tree set, which is exactly what
    `state` is for: it travels once per worker (spawn) or not at all (fork), never once
    per task as a functools.partial would.
    """

    def __init__(self, n_procs: int = 1, spill_dir: Optional[Path] = None,
                 logger: Any = None, keep_alive: Optional[bool] = None,
                 name: str = "grandma"):
        self.n_procs = max(1, int(n_procs))
        self.spill_dir = Path(spill_dir) if spill_dir else Path(tempfile.gettempdir())
        self.logger = logger
        self.name = name
        # Under spawn a pool is expensive and payload-independent -> hold it open.
        # Under fork it must be re-created whenever the payload changes anyway, and an
        # idle pool pins a copy of the parent heap -> drop it after each batch.
        self.keep_alive = (not _USE_FORK) if keep_alive is None else bool(keep_alive)

        self._pool = None
        self._pool_token: Optional[int] = None
        self._pool_procs = 0
        self._state_obj: Any = None
        self._state_key: StateKey = _NO_STATE
        self._state_seq = 0
        self._spill_files: List[str] = []

    # ---------------------------------------------------------------- state
    def publish(self, state: Any) -> StateKey:
        """
        Make `state` reachable by the workers and return its key.

        Re-publishing the SAME object (identity, not equality) is a no-op, so callers
        may hand the same dict to several batches for free. Under fork the state is only
        registered in a module global; under spawn it is additionally written to one
        file that the workers load in parallel on first use.
        """
        if state is None:
            return _NO_STATE
        if self._state_obj is state:
            # Same object -> same token. Under spawn the workers cache by token, but a
            # worker that has NOT yet seen this token still needs the file, so make sure
            # it is still on disk before handing the key out again.
            token, path = self._state_key
            if path is None or os.path.exists(path):
                return self._state_key
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._spill_files.append(path)
            return self._state_key

        self._state_seq += 1
        token = self._state_seq
        path: Optional[str] = None

        if not _USE_FORK:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            path = str(self.spill_dir / f"{self.name}_state_{os.getpid()}_{token}.pickle")
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._spill_files.append(path)
            while len(self._spill_files) > _STATE_CACHE_MAX:
                self._remove(self._spill_files.pop(0))

        # The parent only ever needs the newest payload: older ones are either already
        # inherited by live fork children or on disk for the spawn children.
        for old in [k for k in _STATE_STORE if k != token]:
            _STATE_STORE.pop(old, None)
        _STATE_STORE[token] = state

        self._state_obj = state
        self._state_key = (token, path)
        return self._state_key

    @staticmethod
    def _remove(path: Optional[str]) -> None:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---------------------------------------------------------------- dispatch
    def map_unordered(self, func: Callable[[Any, Any], Any], items: Iterable[Any],
                      state: Any = None, desc: Optional[str] = None, unit: str = "task",
                      total: Optional[int] = None, disable: bool = False,
                      chunksize: int = 1, min_parallel: Optional[int] = None
                      ) -> Iterator[Any]:
        """
        Yield func(state, item) for every item, in completion order.

        `func` must be a module-level function taking (state, item): it is pickled by
        reference, so nothing heavy may be bound to it.

        chunksize defaults to 1 because GRANDMA's tasks are coarse (one MUL-tree against
        every gene tree); chunking would only delay the first progress update by a whole
        chunk and unbalance the workers.

        Results are yielded lazily. Consumers that abandon the iterator early still get
        the pool torn down correctly (the generator's finally clause runs on close).
        """
        items = list(items)
        if total is None:
            total = len(items)

        if min_parallel is None:
            # Standing up a pool costs interpreter start-up plus, under spawn, a package
            # re-import per worker. Below this it never repays itself. The fork figure
            # matches Reconciler._pool_procs so that swapping WorkPool in does not change
            # which runs are parallel.
            min_parallel = (2 * self.n_procs) if not _USE_FORK else max(2, self.n_procs // 2) + 1

        key = self.publish(state)
        bar = dict(total=total, desc=desc, unit=unit, disable=disable or desc is None,
                   ncols=177)

        if self.n_procs <= 1 or len(items) < min_parallel:
            for item in tqdm(items, **bar):
                yield func(_resolve_state(key), item)
            return

        tasks = [(key, item) for item in items]
        from functools import partial
        worker = partial(_apply, func)

        pool = self._get_pool(key[0])
        try:
            for res in tqdm(pool.imap_unordered(worker, tasks, chunksize=chunksize), **bar):
                yield res
        except (KeyboardInterrupt, GeneratorExit):
            # close() would wait for the in-flight tasks to finish, which is exactly
            # what must NOT happen after Ctrl-C: kill the workers and let the interrupt
            # propagate. GeneratorExit lands here too (a consumer that abandons the
            # iterator), and killing the pool is the right response there as well.
            self.terminate()
            raise
        except BaseException:
            self.close()                      # a broken pool must never be reused
            raise
        finally:
            if not self.keep_alive:
                self.close()

    def map(self, func, items, **kwargs) -> List[Any]:
        """Eager convenience wrapper around map_unordered (order is NOT preserved)."""
        return list(self.map_unordered(func, items, **kwargs))

    # ---------------------------------------------------------------- lifecycle
    def _get_pool(self, token: int):
        """
        fork : the payload travels by inheritance, so the pool must be (re-)created
               after it is registered - cheap, and only when the payload changes.
        spawn: the payload travels by file, so the pool is payload-independent and is
               created once and reused.
        """
        if _USE_FORK and self._pool is not None and self._pool_token != token:
            self.close()
        if self._pool is not None and self._pool_procs == self.n_procs:
            return self._pool
        self.close()
        self._pool = mp_context().Pool(processes=self.n_procs)
        self._pool_token = token
        self._pool_procs = self.n_procs
        if self.logger is not None:
            self.logger.log(f"Started a worker pool: {self.n_procs} processes "
                            f"({'fork' if _USE_FORK else 'spawn'}).", 'd')
        return self._pool

    def terminate(self) -> None:
        """Kill the workers immediately, without waiting for their current task.
        close()/join() blocks until in-flight work finishes; after an interrupt that can
        mean minutes, or forever if a worker is wedged."""
        pool, self._pool, self._pool_token = self._pool, None, None
        if pool is not None:
            try:
                pool.terminate()
            except Exception:
                pass
            try:
                pool.join()
            except Exception:
                pass

    def close(self) -> None:
        """Shut the worker processes down. Safe to call repeatedly."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                try:
                    self._pool.terminate()
                except Exception:
                    pass
            finally:
                self._pool = None
                self._pool_token = None

    def cleanup(self) -> None:
        """Release the workers AND any spilled state files. Call once per run."""
        self.close()
        while self._spill_files:
            self._remove(self._spill_files.pop())
        _STATE_STORE.clear()
        _STATE_CACHE.clear()
        self._state_obj = None
        self._state_key = _NO_STATE

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
