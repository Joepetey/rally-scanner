"""Long-running async tasks with transport-agnostic progress reporting."""

import asyncio
import logging
import queue
import time
from collections.abc import Awaitable, Callable

from rally_ml.core.persistence import load_manifest
from rally_ml.pipeline.retrain import retrain_all

logger = logging.getLogger(__name__)

SendFn = Callable[[str], Awaitable[None]]
SendEmbedFn = Callable[[dict], Awaitable[None]]


async def run_retrain(
    send: SendFn,
    tickers: list[str] | None = None,
) -> None:
    """Run model retraining with live progress updates.

    Args:
        send: async callable that delivers a progress string to the user.
        tickers: optional subset of tickers to retrain.
    """
    try:
        await send(
            "\U0001f504 **Starting model retraining...**\n"
            "This will take 10-30+ minutes depending on the number of tickers."
        )

        start_time = time.time()
        progress_queue: queue.Queue[tuple[int, int, int, int]] = queue.Queue()

        def on_progress(done: int, total: int, success: int, failed: int) -> None:
            try:
                progress_queue.get_nowait()
            except queue.Empty:
                pass
            progress_queue.put((done, total, success, failed))

        async def send_updates() -> None:
            while True:
                await asyncio.sleep(300)
                elapsed_min = int((time.time() - start_time) / 60)
                try:
                    done, total, success, failed = progress_queue.get_nowait()
                    pct = int(done / total * 100) if total else 0
                    await send(
                        f"\u23f3 **Retraining... {elapsed_min}m elapsed**\n"
                        f"\u2022 Progress: {done}/{total} ({pct}%)\n"
                        f"\u2022 Success: {success} | Failed: {failed}"
                    )
                except queue.Empty:
                    await send(
                        f"\u23f3 **Retraining... {elapsed_min}m elapsed** (fetching data)"
                    )

        update_task = asyncio.create_task(send_updates())
        try:
            await asyncio.to_thread(retrain_all, tickers, False, on_progress)
        finally:
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass

        elapsed = time.time() - start_time
        minutes = int(elapsed / 60)
        seconds = int(elapsed % 60)
        try:
            done, total, success, failed = progress_queue.get_nowait()
        except queue.Empty:
            manifest = load_manifest()
            success = len(manifest) if manifest else 0
            failed = 0
            total = success

        await send(
            f"\u2705 **Retraining complete!**\n"
            f"\u2022 Trained: {success}/{total} models\n"
            f"\u2022 Failed: {failed}\n"
            f"\u2022 Time: {minutes}m {seconds}s\n"
            f"\u2022 You can now run scans to find signals!"
        )
    except Exception as e:
        logger.exception("Retrain task failed")
        await send(f"\u274c **Retraining failed:** {str(e)}")


async def run_simulation(
    scenario: str,
    equity_override: float,
    inject_fn: Callable | None,
    send: SendFn,
    send_embed: SendEmbedFn,
) -> None:
    """Run a BTC-USD paper trading simulation.

    Args:
        scenario: one of target/stop/trail/time.
        equity_override: account equity override (0 to auto-fetch from Alpaca).
        inject_fn: trade injection function from scheduler stream, or None.
        send: async callable for text messages.
        send_embed: async callable for embed dicts.
    """
    from integrations.alpaca.broker import simulation_keys
    from simulation.runner import SimulationRunner

    if not scenario:
        await send(
            "Usage: `!simulate <scenario> [equity]`\n"
            "Scenarios: `target` `stop` `trail` `time`"
        )
        return

    try:
        sim_ctx = simulation_keys()
        sim_ctx.__enter__()
    except RuntimeError as e:
        await send(f"\u26a0\ufe0f {e}")
        return

    try:
        if equity_override > 0:
            equity = equity_override
        else:
            try:
                from integrations.alpaca.executor import get_account_equity
                equity = await get_account_equity()
            except Exception as e:
                await send(f"\u26a0\ufe0f Could not fetch account equity: {e}")
                return

        runner = SimulationRunner(inject_fn=inject_fn)

        await send(
            f"\u25b6\ufe0f **Simulation starting** \u2014 scenario: `{scenario}`"
            f" | equity: `${equity:,.0f}`\n"
            f"BTC paper order will be placed on simulation account. Results follow..."
        )
        result = await runner.run(scenario, equity, send_embed)

        if result.success:
            pnl = result.realized_pnl_pct or 0.0
            sign = "+" if pnl >= 0 else ""
            await send(
                f"\u2705 **Simulation complete** \u2014 `{scenario}`\n"
                f"Entry: `${result.entry_price:,.2f}` \u2192 Exit: `${result.exit_price:,.2f}`\n"
                f"Reason: `{result.exit_reason}` | PnL: `{sign}{pnl:.2f}%`"
            )
        else:
            await send(f"\u274c **Simulation failed** \u2014 `{scenario}`\n{result.error}")
    finally:
        sim_ctx.__exit__(None, None, None)
