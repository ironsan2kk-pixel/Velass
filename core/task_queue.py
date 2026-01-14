"""
Task Queue - Async task management for backtests and long-running operations

Provides:
- BacktestQueue for managing backtest tasks
- Progress tracking
- Concurrent execution with limits
- Result caching
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0


@dataclass
class BacktestTask:
    """Backtest task definition."""
    task_id: str
    pair: str
    strategy_name: str
    timeframe: str
    days: int
    params: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    progress_message: str = ""

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metadata
    user_id: Optional[int] = None
    callback_chat_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "pair": self.pair,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "days": self.days,
            "params": self.params,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


class TaskQueue:
    """
    Generic async task queue with concurrency control.

    Features:
    - Configurable max concurrent tasks
    - Progress tracking
    - Task cancellation
    - Result caching
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        max_queue_size: int = 100,
    ):
        """
        Initialize task queue.

        Args:
            max_concurrent: Maximum concurrent tasks
            max_queue_size: Maximum queue size
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size

        # Task storage
        self._tasks: Dict[str, BacktestTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running_count = 0

        # Callbacks
        self._on_complete: Optional[Callable[[BacktestTask], Coroutine]] = None
        self._on_progress: Optional[Callable[[str, float, str], Coroutine]] = None

        # Control
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

        # Thread pool for CPU-bound work
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def start(self) -> None:
        """Start the queue workers."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_concurrent)
        ]
        logger.info(f"Started {self.max_concurrent} queue workers")

    async def stop(self) -> None:
        """Stop the queue workers."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        # Wait for completion
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # Shutdown executor
        self._executor.shutdown(wait=False)
        logger.info("Queue stopped")

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks."""
        while self._running:
            try:
                # Get task from queue
                task_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                task = self._tasks.get(task_id)
                if task is None:
                    continue

                # Execute task
                await self._execute_task(task, worker_id)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _execute_task(self, task: BacktestTask, worker_id: int) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        async with self._lock:
            self._running_count += 1

        try:
            logger.info(f"Worker {worker_id} executing task {task.task_id}")

            # This is where the actual work happens
            # Subclasses should override this
            result = await self._run_backtest(task)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            task.progress_message = "Completed"

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "Task cancelled"
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
        finally:
            task.completed_at = datetime.utcnow()

            async with self._lock:
                self._running_count -= 1

            # Call completion callback
            if self._on_complete:
                try:
                    await self._on_complete(task)
                except Exception as e:
                    logger.error(f"Completion callback error: {e}")

    async def _run_backtest(self, task: BacktestTask) -> Dict[str, Any]:
        """
        Run backtest - override in subclass.

        This is a placeholder that should be overridden with actual
        backtest logic or delegated to StrategyManager.
        """
        # Simulate progress
        for i in range(10):
            await asyncio.sleep(0.1)
            task.progress = (i + 1) * 10
            task.progress_message = f"Processing... {task.progress:.0f}%"

            if self._on_progress:
                await self._on_progress(task.task_id, task.progress, task.progress_message)

        return {"simulated": True, "pair": task.pair}

    def submit(
        self,
        pair: str,
        strategy_name: str,
        days: int = 30,
        timeframe: str = "1h",
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        callback_chat_id: Optional[int] = None,
    ) -> str:
        """
        Submit a new backtest task.

        Returns:
            Task ID
        """
        task_id = f"bt_{uuid.uuid4().hex[:8]}"

        task = BacktestTask(
            task_id=task_id,
            pair=pair,
            strategy_name=strategy_name,
            timeframe=timeframe,
            days=days,
            params=params or {},
            status=TaskStatus.QUEUED,
            user_id=user_id,
            callback_chat_id=callback_chat_id,
        )

        self._tasks[task_id] = task

        # Add to queue (non-blocking)
        try:
            self._queue.put_nowait(task_id)
        except asyncio.QueueFull:
            task.status = TaskStatus.FAILED
            task.error = "Queue is full"

        return task_id

    def get_task(self, task_id: str) -> Optional[BacktestTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status as dictionary."""
        task = self._tasks.get(task_id)
        if task:
            return task.to_dict()
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        user_id: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [t.to_dict() for t in tasks[:limit]]

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        status_counts = {}
        for task in self._tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "running_count": self._running_count,
            "max_concurrent": self.max_concurrent,
            "total_tasks": len(self._tasks),
            "status_counts": status_counts,
        }

    def clear_completed(self, older_than_hours: int = 24) -> int:
        """Remove completed tasks older than specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        return len(to_remove)

    def on_complete(self, callback: Callable[[BacktestTask], Coroutine]) -> None:
        """Set completion callback."""
        self._on_complete = callback

    def on_progress(self, callback: Callable[[str, float, str], Coroutine]) -> None:
        """Set progress callback."""
        self._on_progress = callback


class BacktestQueue(TaskQueue):
    """
    Specialized queue for backtests with StrategyManager integration.
    """

    def __init__(
        self,
        manager: Any,  # StrategyManager
        max_concurrent: int = 4,
    ):
        """
        Initialize backtest queue.

        Args:
            manager: StrategyManager instance
            max_concurrent: Max concurrent backtests
        """
        super().__init__(max_concurrent=max_concurrent)
        self.manager = manager

    async def _run_backtest(self, task: BacktestTask) -> Dict[str, Any]:
        """Run backtest using StrategyManager."""
        # Update progress
        task.progress = 10.0
        task.progress_message = "Loading data..."

        if self._on_progress:
            await self._on_progress(task.task_id, task.progress, task.progress_message)

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()

        def run_sync():
            return self.manager.backtest(
                pair=task.pair,
                strategy_name=task.strategy_name,
                days=task.days,
                params=task.params if task.params else None,
            )

        task.progress = 30.0
        task.progress_message = "Running backtest..."

        if self._on_progress:
            await self._on_progress(task.task_id, task.progress, task.progress_message)

        result = await loop.run_in_executor(self._executor, run_sync)

        task.progress = 90.0
        task.progress_message = "Finalizing..."

        if self._on_progress:
            await self._on_progress(task.task_id, task.progress, task.progress_message)

        return result


class BatchBacktestQueue:
    """
    Queue for batch backtests (multiple pairs/strategies).

    Submits multiple backtests and tracks overall progress.
    """

    def __init__(self, queue: BacktestQueue):
        """Initialize with a BacktestQueue."""
        self.queue = queue
        self._batches: Dict[str, Dict[str, Any]] = {}

    def submit_batch(
        self,
        pairs: List[str],
        strategy_names: List[str],
        days: int = 30,
        user_id: Optional[int] = None,
    ) -> str:
        """
        Submit a batch of backtests.

        Returns:
            Batch ID
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"

        task_ids = []
        for pair in pairs:
            for strategy in strategy_names:
                task_id = self.queue.submit(
                    pair=pair,
                    strategy_name=strategy,
                    days=days,
                    user_id=user_id,
                )
                task_ids.append(task_id)

        self._batches[batch_id] = {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "total": len(task_ids),
            "created_at": datetime.utcnow().isoformat(),
        }

        return batch_id

    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch progress."""
        batch = self._batches.get(batch_id)
        if not batch:
            return None

        completed = 0
        failed = 0
        running = 0

        for task_id in batch["task_ids"]:
            task = self.queue.get_task(task_id)
            if task:
                if task.status == TaskStatus.COMPLETED:
                    completed += 1
                elif task.status == TaskStatus.FAILED:
                    failed += 1
                elif task.status == TaskStatus.RUNNING:
                    running += 1

        total = batch["total"]
        progress = ((completed + failed) / total * 100) if total > 0 else 0

        return {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": total - completed - failed - running,
            "progress": progress,
            "is_done": (completed + failed) >= total,
        }

    def get_batch_results(self, batch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all results for a batch."""
        batch = self._batches.get(batch_id)
        if not batch:
            return None

        results = []
        for task_id in batch["task_ids"]:
            task = self.queue.get_task(task_id)
            if task:
                results.append(task.to_dict())

        return results
