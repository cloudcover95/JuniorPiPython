# Pi overnight

On a Pi, run JuniorLLM:

    PYTHONPATH=/opt/JuniorLLM python3 -c "from bitnet_night.cycle import run_cycle; from pathlib import Path; run_cycle(Path('/var/lib/junior/night'), 'pi overnight', 32)"

32 ticks is the Pi default (battery). Loopback bitnetd remains 127.0.0.1 only.
