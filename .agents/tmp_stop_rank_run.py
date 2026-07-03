import os
import signal
import subprocess
import time

target = "train_baseline40_then_rank005"
stopped = []

ps = subprocess.run(
    ["ps", "-eo", "pid,ppid,pgid,stat,cmd"],
    text=True,
    capture_output=True,
    check=False,
).stdout

pids = []
pgids = set()
for line in ps.splitlines():
    if target in line:
        parts = line.split(None, 4)
        if len(parts) >= 5:
            pid = int(parts[0])
            pgid = int(parts[2])
            pids.append(pid)
            pgids.add(pgid)

for pgid in sorted(pgids):
    try:
        os.killpg(pgid, signal.SIGTERM)
        stopped.append(("TERM_PG", pgid))
    except ProcessLookupError:
        stopped.append(("gone_pg", pgid))
    except Exception as exc:
        stopped.append(("term_pg_err", pgid, repr(exc)))

time.sleep(5)

ps_after_term = subprocess.run(
    ["ps", "-eo", "pid,ppid,pgid,stat,cmd"],
    text=True,
    capture_output=True,
    check=False,
).stdout

remaining = []
for line in ps_after_term.splitlines():
    if target in line:
        parts = line.split(None, 4)
        if len(parts) >= 5:
            remaining.append(int(parts[0]))

for pid in remaining:
    try:
        os.kill(pid, signal.SIGKILL)
        stopped.append(("KILL", pid))
    except ProcessLookupError:
        stopped.append(("gone_pid", pid))
    except Exception as exc:
        stopped.append(("kill_err", pid, repr(exc)))

time.sleep(2)

ps_final = subprocess.run(
    ["ps", "-eo", "pid,ppid,pgid,stat,cmd"],
    text=True,
    capture_output=True,
    check=False,
).stdout

final_lines = [line for line in ps_final.splitlines() if target in line]
print("initial_pids", pids)
print("initial_pgids", sorted(pgids))
print("stopped", stopped)
print("remaining_train_lines")
print("\n".join(final_lines))
