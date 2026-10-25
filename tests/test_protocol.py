"""Protocol contract tests.

These pin the wire format against the frontend's ``bridge.ts``. A rename on
either side should break here rather than silently in a demo.
"""

import json

import pytest

from doosan.protocol import (
    WEAR_PARAMS,
    JointReport,
    Pose,
    StateEnvelope,
    TelemetryEnvelope,
    cmd_configure,
    cmd_ctrl,
    cmd_echo,
    cmd_mode,
    cmd_wear,
)

# Exactly as the browser emits it — camelCase, operator-facing names.
def _robot(mode: str = "remote") -> dict:
    return {
        "mode": mode,
        "qpos": [0.0, 0.3, -0.6, 0.0, 0.2, 0.0],
        "qvel": [0.1] * 6,
        "ctrl": [0.0, 0.31, -0.59, 0.0, 0.2, 0.0],
        "torque": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "ee": {"pos": [0.4, 0.0, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
    }


WIRE_STATE = {
    "type": "state",
    "seq": 3,
    "sent": 1700000000000,
    "simTime": 1.25,
    "session": "abc",
    "scene": "cell",
    "robots": {"armA": _robot("remote"), "armB": _robot("scripted")},
}

WIRE_REPORT = {
    "joint": 2,
    "frictionRatio": 1.4,
    "wearSlop": 0.01,
    "lostMotion": 0.002,
    "motorStrength": 0.8,
    "actuatorDelay": 0.03,
    "mountCompliance": 0.001,
    "hoursSinceJointReplacement": 4000.0,
    "hoursSinceRegrease": 400.0,
    "checkoutTemp": 38.0,
    "wearAcceleration": -1.5e-6,
    "highestDamage": 2.2e-5,
    "damage": 7.7e-5,
    "travel": 12.5,
    "reversals": 9,
    "work": 88.0,
    "peakTorque": 41.0,
    "duty": 0.62,
    "pos": 0.31,
    "vel": -0.05,
    "torque": 33.0,
    "trackingError": 0.004,
}


class TestStateEnvelope:
    def test_parses_browser_payload(self):
        s = StateEnvelope.from_dict(WIRE_STATE)
        assert s.names == ["armA", "armB"]
        assert s.scene == "cell"
        assert s.sim_time == 1.25
        assert s.robots["armA"].qpos[1] == 0.3

    def test_each_robot_reports_its_own_authority(self):
        # The point of per-robot modes: one arm scripted, another remote.
        s = StateEnvelope.from_dict(WIRE_STATE)
        assert s.robots["armA"].mode == "remote"
        assert s.robots["armB"].mode == "scripted"

    def test_all_arrays_agree_on_joint_count(self):
        for r in StateEnvelope.from_dict(WIRE_STATE).robots.values():
            assert len(r.qvel) == len(r.ctrl) == len(r.torque) == r.n == 6

    def test_ctrl_is_post_degradation(self):
        # ctrl differs from qpos because delay/backlash altered the command.
        r = StateEnvelope.from_dict(WIRE_STATE).robots["armA"]
        assert r.ctrl[1] != r.qpos[1]

    def test_sim_time_reaches_each_robot(self):
        # Controllers receive only a RobotState, so it must carry the clock.
        for r in StateEnvelope.from_dict(WIRE_STATE).robots.values():
            assert r.sim_time == 1.25

    def test_quaternion_is_mujoco_wxyz(self):
        r = StateEnvelope.from_dict(WIRE_STATE).robots["armA"]
        assert r.ee is not None
        # Identity rotation has w=1; guards against a silent xyzw swap.
        assert r.ee.quat[0] == 1.0
        assert r.ee.quat[1:] == (0.0, 0.0, 0.0)

    def test_tolerates_missing_end_effector(self):
        payload = {**WIRE_STATE, "robots": {"armA": {**_robot(), "ee": None}}}
        assert StateEnvelope.from_dict(payload).robots["armA"].ee is None

    def test_tolerates_empty_payload(self):
        s = StateEnvelope.from_dict({})
        assert s.robots == {}
        assert s.names == []

    def test_only_resolves_a_single_arm_scene(self):
        single = {**WIRE_STATE, "robots": {"": _robot()}}
        assert StateEnvelope.from_dict(single).only is not None
        # Ambiguous with two arms, so callers must name one.
        assert StateEnvelope.from_dict(WIRE_STATE).only is None

    def test_robot_lookup_by_name(self):
        s = StateEnvelope.from_dict(WIRE_STATE)
        assert s.robot("armB") is not None
        assert s.robot("nope") is None


class TestJointReport:
    def test_translates_camelcase_wire_names(self):
        r = JointReport.from_dict(WIRE_REPORT)
        assert r.joint == 2
        assert r.friction == 1.4       # frictionRatio
        assert r.slop == 0.01          # wearSlop
        assert r.strength == 0.8       # motorStrength
        assert r.delay == 0.03         # actuatorDelay
        assert r.compliance == 0.001   # mountCompliance
        assert r.hours_since_replacement == 4000.0

    def test_keeps_derived_fields(self):
        r = JointReport.from_dict(WIRE_REPORT)
        assert r.damage == 7.7e-5
        assert r.reversals == 9
        assert r.tracking_error == 0.004

    def test_log_record_is_json_serialisable(self):
        r = JointReport.from_dict(WIRE_REPORT)
        assert json.loads(json.dumps(r.to_dict()))["joint"] == 2

    def test_defaults_model_a_healthy_joint(self):
        r = JointReport.from_dict({"joint": 1})
        assert r.friction == 1.0
        assert r.strength == 1.0
        assert r.damage == 0.0


class TestTelemetryEnvelope:
    def test_parses_a_batch(self):
        env = TelemetryEnvelope.from_dict(
            {
                "type": "telemetry",
                "seq": 0,
                "sent": 1,
                "simTime": 2.0,
                "session": "s",
                "scene": "cell",
                "joints": [WIRE_REPORT, {**WIRE_REPORT, "joint": 3}],
            }
        )
        assert [j.joint for j in env.joints] == [2, 3]

    def test_tolerates_empty_batch(self):
        assert TelemetryEnvelope.from_dict({}).joints == []


class TestCommands:
    def test_every_command_is_json_serialisable(self):
        for msg in (
            cmd_mode("remote"),
            cmd_ctrl([0.1, 0.2]),
            cmd_wear(2, friction=3.1),
            cmd_echo(7),
            cmd_configure(state_hz=100),
        ):
            assert json.loads(json.dumps(msg))["type"] == msg["type"]

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            cmd_mode("turbo")  # type: ignore[arg-type]

    def test_rejects_unknown_wear_param(self):
        # Catching a typo here beats silently sending a field the sim ignores.
        with pytest.raises(ValueError, match="unknown wear params"):
            cmd_wear(1, frictionRatio=2.0)

    def test_accepts_every_documented_wear_param(self):
        for name in WEAR_PARAMS:
            assert cmd_wear(1, **{name: 1.0})["joints"][0][name] == 1.0

    def test_ctrl_coerces_to_float(self):
        assert cmd_ctrl([1, 2])["values"] == [1.0, 2.0]

    def test_configure_omits_unset_rates(self):
        assert "telemetryHz" not in cmd_configure(state_hz=50)


def test_pose_defaults_to_identity_rotation():
    p = Pose.from_dict({"pos": [1, 2, 3]})
    assert p is not None
    assert p.quat == (1.0, 0.0, 0.0, 0.0)
