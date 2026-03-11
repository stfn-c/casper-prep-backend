import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import app.main as main


class MockAggregationTriggerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_mock_tasks = main._mock_tasks.copy()
        main._mock_tasks.clear()

    def tearDown(self):
        main._mock_tasks.clear()
        main._mock_tasks.update(self._orig_mock_tasks)

    async def test_triggers_aggregation_when_all_scenarios_completed(self):
        fake_task = MagicMock()

        def _fake_create_task(coro):
            # Prevent "coroutine was never awaited" warnings in unit tests.
            coro.close()
            return fake_task

        with patch.object(main.db, "get_scenario_attempt", AsyncMock(return_value={"mock_exam_attempt_id": 42})), \
             patch.object(main.db, "get_mock_exam_attempt", AsyncMock(return_value={"feedback_status": "processing"})), \
             patch.object(
                 main.db,
                 "get_scenario_attempts_for_mock",
                 AsyncMock(return_value=[
                     {"feedback_status": "completed"},
                     {"feedback_status": "completed"},
                 ]),
             ), \
             patch.object(main.db, "update_mock_exam_attempt_status", AsyncMock()) as update_status, \
             patch("app.main.asyncio.create_task", side_effect=_fake_create_task) as create_task, \
             patch("app.main._register_task") as register_task:
            await main._maybe_trigger_mock_aggregation_for_attempt(106)

        create_task.assert_called_once()
        register_task.assert_called_once_with(main._mock_tasks, 42, fake_task)
        update_status.assert_not_called()

    async def test_marks_mock_failed_when_any_scenario_failed(self):
        with patch.object(main.db, "get_scenario_attempt", AsyncMock(return_value={"mock_exam_attempt_id": 99})), \
             patch.object(main.db, "get_mock_exam_attempt", AsyncMock(return_value={"feedback_status": "processing"})), \
             patch.object(
                 main.db,
                 "get_scenario_attempts_for_mock",
                 AsyncMock(return_value=[
                     {"feedback_status": "completed"},
                     {"feedback_status": "failed"},
                 ]),
             ), \
             patch.object(main.db, "update_mock_exam_attempt_status", AsyncMock()) as update_status, \
             patch("app.main.asyncio.create_task") as create_task:
            await main._maybe_trigger_mock_aggregation_for_attempt(106)

        update_status.assert_called_once_with(99, "failed")
        create_task.assert_not_called()

    async def test_run_full_video_task_always_checks_mock_aggregation(self):
        with patch("app.main.run_full_video_analysis", AsyncMock(return_value={"status": "completed"})), \
             patch("app.main._maybe_trigger_mock_aggregation_for_attempt", AsyncMock()) as maybe_trigger:
            await main.run_full_video_analysis_task(106, "videos/u/a/full.webm")

        maybe_trigger.assert_awaited_once_with(106)


if __name__ == "__main__":
    unittest.main()
