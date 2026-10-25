"""Controller behaviour and joint-limit safety."""

import pytest

from doosan.controller import (
    JOINT_LIMITS,
    IdleController,
    PIDController,
    SineController,
    WaypointController,
    clamp,
    make,
)
from doosan.protocol import RobotState


def state(
    t: float = 0.0,
    qpos: list[float] | None = None,
    mode: str = "remote",
    name: str = "armA",
) -> RobotState:
    # `is None`, not `or`: an empty list is a meaningful input (no joints yet).
    q = [0.0] * 6 if qpos is None else qpos
    return RobotState(
        name=name,
        sim_time=t,
        mode=mode,  # type: ignore[arg-type]
        qpos=q,
        qvel=[0.0] * len(q),
        ctrl=list(q),
        torque=[0.0] * len(q),
    )


class TestClamp:
    def test_respects_the_tightest_joint(self):
        # Joint 3 is limited to ±2.7925 rad in the Doosan MJCF.
        assert clamp([0, 0, 99, 0, 0, 0])[2] == pytest.approx(JOINT_LIMITS[2][1])
        assert clamp([0, 0, -99, 0, 0, 0])[2] == pytest.approx(JOINT_LIMITS[2][0])

    def test_leaves_in_range_values_untouched(self):
        v = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        assert clamp(v) == v


class TestIdle:
    def test_sends_nothing(self):
        assert IdleController().step(state()) is None


class TestSine:
    def test_produces_one_target_per_joint(self):
        assert len(SineController().step(state(1.0))) == 6

    def test_output_varies_with_time(self):
        c = SineController()
        assert c.step(state(0.0)) != c.step(state(2.0))

    def test_stays_within_limits_across_a_long_run(self):
        c = SineController(amplitude=3.0)  # deliberately over-driven
        for i in range(500):
            for j, v in enumerate(c.step(state(i * 0.05))):
                lo, hi = JOINT_LIMITS[j]
                assert lo <= v <= hi

    def test_handles_a_scene_with_no_joints(self):
        assert SineController().step(state(qpos=[])) is None


class TestWaypoint:
    def test_loops_with_a_stable_period(self):
        c = WaypointController()
        first = c.step(state(0.0))
        again = c.step(state(c.duration))
        assert first == pytest.approx(again, abs=1e-9)

    def test_motion_is_continuous(self):
        # A waypoint jump would show up as a torque spike and pollute wear.
        c = WaypointController()
        prev = c.step(state(0.0))
        for i in range(1, 400):
            cur = c.step(state(i * 0.02))
            assert all(abs(a - b) < 0.05 for a, b in zip(cur, prev))
            prev = cur

    def test_anchors_to_first_state_seen(self):
        # Starting mid-simulation must still begin at the path's start.
        c = WaypointController()
        assert c.step(state(1234.5)) == pytest.approx(c.DEFAULT[0][0], abs=1e-9)

    def test_reset_reanchors(self):
        c = WaypointController()
        c.step(state(100.0))
        c.reset()
        assert c._t0 is None


class TestPID:
    def test_drives_toward_the_setpoint(self):
        c = PIDController(target=[1.0] * 6, kp=0.5)
        out = c.step(state(0.0, qpos=[0.0] * 6))
        # Below target, so the correction must push upward past feedforward.
        assert all(v > 1.0 for v in out)

    def test_settles_at_the_setpoint(self):
        c = PIDController(target=[0.5] * 6)
        out = c.step(state(0.0, qpos=[0.5] * 6))
        assert out == pytest.approx([0.5] * 6)

    def test_integral_is_bounded(self):
        c = PIDController(target=[2.0] * 6, ki=1.0, i_limit=0.3)
        for i in range(200):
            c.step(state(i * 0.02, qpos=[0.0] * 6))
        assert all(abs(v) <= 0.3 + 1e-9 for v in c._integral)

    def test_reset_clears_accumulation(self):
        c = PIDController(target=[1.0] * 6, ki=0.5)
        for i in range(10):
            c.step(state(i * 0.02))
        c.reset()
        assert c._integral == []
        assert c._last_t is None

    def test_output_is_clamped(self):
        c = PIDController(target=[99.0] * 6, kp=5.0)
        for j, v in enumerate(c.step(state(0.0))):
            lo, hi = JOINT_LIMITS[j]
            assert lo <= v <= hi


class TestRegistry:
    @pytest.mark.parametrize("name", ["idle", "sine", "waypoint", "pid"])
    def test_every_registered_controller_builds_and_steps(self, name):
        c = make(name)
        assert c.name == name
        c.step(state(1.0))  # must not raise
        c.reset()

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="unknown controller"):
            make("nope")
