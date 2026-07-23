"""
schemas/mission.py
==================
The single source of truth for every data structure in the pipeline.

Both the LLM planner and the validator import from here — never the reverse.
The executor only accepts a validated Mission object constructed from this module,
ensuring the LLM can never inject unvalidated data into the control loop.

Safety rationale for every constraint is called out in comments below.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Constants (edit here to change limits globally) ───────────────────────────

# Altitude limits (metres AGL)
# Lower bound: 2 m — below this, ground effect and obstacle risk are too high
#              for autonomous ops without a human with a physical override.
# Upper bound: 30 m — conservative regulatory ceiling for this demo; well below
#              the 120 m AGL limit in most national frameworks but deliberately
#              stricter to prevent runaway climbs during testing.
ALT_MIN_M: float = 2.0
ALT_MAX_M: float = 30.0

# Speed limits (m/s)
# Lower bound: 1.0 m/s — below this, the drone cannot hold attitude reliably
#              in even a light breeze, causing drift to exceed waypoint tolerance.
# Upper bound: 8.0 m/s — stays well within the ~12 m/s SITL airspeed cap and
#              leaves margin for braking manoeuvres at each waypoint.
SPEED_MIN_MPS: float = 1.0
SPEED_MAX_MPS: float = 8.0

# Loop count limit
# Upper bound: 10 — prevents an LLM hallucination from burning the battery
#              on an unbounded patrol mission.  Any mission needing more than
#              10 loops should be re-planned with a longer single loop.
LOOP_COUNT_MIN: int = 1
LOOP_COUNT_MAX: int = 10

# Maximum number of waypoints in a single mission.
# Practical limit that also bounds executor log size.
MAX_WAYPOINTS: int = 50


# ── Action type discriminator values ─────────────────────────────────────────

class ActionType(str, Enum):
    TAKEOFF = "takeoff"
    GOTO_WAYPOINT = "goto_waypoint"
    LOOP = "loop"
    RTL = "rtl"
    LAND = "land"


# ── Individual action models ──────────────────────────────────────────────────

class TakeoffAction(BaseModel):
    """Arm and climb to a target altitude, then hover."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal[ActionType.TAKEOFF] = ActionType.TAKEOFF
    altitude_m: float = Field(
        ...,
        description="Target altitude in metres AGL after takeoff.",
    )

    @field_validator("altitude_m")
    @classmethod
    def altitude_in_safe_range(cls, v: float) -> float:
        # Safety: reject values outside the globally defined safe band.
        if not (ALT_MIN_M <= v <= ALT_MAX_M):
            raise ValueError(
                f"Takeoff altitude {v} m is outside the safe range "
                f"[{ALT_MIN_M}, {ALT_MAX_M}] m."
            )
        return v


class GotoWaypointAction(BaseModel):
    """Fly to a geographic waypoint at a given altitude and speed."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal[ActionType.GOTO_WAYPOINT] = ActionType.GOTO_WAYPOINT
    lat: float = Field(..., description="Target latitude in decimal degrees.")
    lon: float = Field(..., description="Target longitude in decimal degrees.")
    alt_m: float = Field(..., description="Target altitude in metres AGL.")
    speed_mps: float = Field(..., description="Cruise speed to this waypoint (m/s).")

    @field_validator("alt_m")
    @classmethod
    def waypoint_altitude_in_safe_range(cls, v: float) -> float:
        if not (ALT_MIN_M <= v <= ALT_MAX_M):
            raise ValueError(
                f"Waypoint altitude {v} m is outside the safe range "
                f"[{ALT_MIN_M}, {ALT_MAX_M}] m."
            )
        return v

    @field_validator("speed_mps")
    @classmethod
    def speed_in_safe_range(cls, v: float) -> float:
        if not (SPEED_MIN_MPS <= v <= SPEED_MAX_MPS):
            raise ValueError(
                f"Speed {v} m/s is outside the safe range "
                f"[{SPEED_MIN_MPS}, {SPEED_MAX_MPS}] m/s."
            )
        return v

    @field_validator("lat")
    @classmethod
    def lat_in_wgs84_range(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude {v} is outside WGS-84 range [-90, 90].")
        return v

    @field_validator("lon")
    @classmethod
    def lon_in_wgs84_range(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude {v} is outside WGS-84 range [-180, 180].")
        return v


class LoopAction(BaseModel):
    """
    Repeat a sequence of previously defined waypoints N times.

    waypoint_ids: ordered list of GotoWaypointAction.action_id values.
    The validator cross-checks these IDs against actual actions in the mission.
    """

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal[ActionType.LOOP] = ActionType.LOOP
    repeat_count: int = Field(
        ...,
        description="How many times to fly the listed waypoints.",
    )
    waypoint_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of GotoWaypointAction action_ids to repeat.",
    )

    @field_validator("repeat_count")
    @classmethod
    def loop_count_in_safe_range(cls, v: int) -> int:
        # Safety: an unbounded loop is indistinguishable from a stuck mission
        # at the executor level.  Cap at 10 iterations.
        if not (LOOP_COUNT_MIN <= v <= LOOP_COUNT_MAX):
            raise ValueError(
                f"repeat_count {v} is outside the safe range "
                f"[{LOOP_COUNT_MIN}, {LOOP_COUNT_MAX}]."
            )
        return v


class RtlAction(BaseModel):
    """Return to launch position and land there."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal[ActionType.RTL] = ActionType.RTL


class LandAction(BaseModel):
    """Land immediately at the current position."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: Literal[ActionType.LAND] = ActionType.LAND


# Discriminated union — Pydantic resolves the concrete type from the 'type' field.
MissionAction = Annotated[
    Union[TakeoffAction, GotoWaypointAction, LoopAction, RtlAction, LandAction],
    Field(discriminator="type"),
]


# ── Geofence / global constraints ─────────────────────────────────────────────

class GeofenceBoundingBox(BaseModel):
    """
    Axis-aligned bounding box in WGS-84 decimal degrees.
    All waypoints in the mission must fall strictly inside this box.
    This is the primary geofence guardrail — rejection is mandatory if violated.
    """

    lat_min: float = Field(..., ge=-90.0, le=90.0)
    lat_max: float = Field(..., ge=-90.0, le=90.0)
    lon_min: float = Field(..., ge=-180.0, le=180.0)
    lon_max: float = Field(..., ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def min_less_than_max(self) -> "GeofenceBoundingBox":
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be strictly less than lat_max.")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be strictly less than lon_max.")
        return self

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if the point falls inside (or on) the geofence boundary."""
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


class GlobalConstraints(BaseModel):
    """
    Mission-level safety envelope.  Every action must respect these values
    AND the validator re-checks individual action fields against them.
    Having both field-level AND constraint-level checks provides defense-in-depth:
    even if the per-field validator is somehow relaxed, the global check still catches it.
    """

    max_altitude_m: float = Field(
        ...,
        le=ALT_MAX_M,
        description="Hard ceiling for the entire mission (metres AGL).",
    )
    max_speed_mps: float = Field(
        ...,
        le=SPEED_MAX_MPS,
        description="Maximum cruise speed anywhere in the mission (m/s).",
    )
    geofence: GeofenceBoundingBox = Field(
        ...,
        description="All waypoints must fall inside this bounding box.",
    )
    max_loop_count: int = Field(
        ...,
        le=LOOP_COUNT_MAX,
        ge=LOOP_COUNT_MIN,
        description="Maximum loop iterations allowed for any LoopAction.",
    )


# ── Top-level Mission ──────────────────────────────────────────────────────────

class Mission(BaseModel):
    """
    A fully validated, immutable mission.

    Construction rules:
      1. actions must be non-empty.
      2. actions must start with a TakeoffAction.
      3. actions must end with an RtlAction or LandAction.
      4. All GotoWaypointAction items must fall inside constraints.geofence.
      5. All LoopAction.waypoint_ids must reference existing GotoWaypointAction ids.
      6. No action may exceed constraints.max_altitude_m or constraints.max_speed_mps.

    These invariants are enforced at model_validator level so they cannot be
    bypassed after construction.
    """

    mission_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier (UUID4). Auto-generated if not provided.",
    )
    created_from_prompt: str = Field(
        ...,
        description="The original natural-language prompt, kept for audit trail.",
    )
    actions: List[MissionAction] = Field(
        ...,
        min_length=1,
        max_length=MAX_WAYPOINTS,
        description="Ordered list of actions the executor will run sequentially.",
    )
    constraints: GlobalConstraints = Field(
        ...,
        description="Mission-level safety envelope.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the validated mission was created.",
    )

    model_config = {"frozen": True}  # Immutable after construction

    @model_validator(mode="after")
    def validate_mission_structure(self) -> "Mission":
        errors: List[str] = []

        # Rule 1: Must start with takeoff.
        if not isinstance(self.actions[0], TakeoffAction):
            errors.append(
                "First action must be TakeoffAction. "
                f"Got: {self.actions[0].type}"
            )

        # Rule 2: Must end with RTL or Land.
        last = self.actions[-1]
        if not isinstance(last, (RtlAction, LandAction)):
            errors.append(
                "Last action must be RtlAction or LandAction. "
                f"Got: {last.type}"
            )

        # Build a set of valid waypoint action_ids for loop cross-checking.
        goto_ids = {
            a.action_id
            for a in self.actions
            if isinstance(a, GotoWaypointAction)
        }

        for action in self.actions:
            if isinstance(action, GotoWaypointAction):
                # Rule 3: Geofence check (defence-in-depth against field validator bypass).
                if not self.constraints.geofence.contains(action.lat, action.lon):
                    errors.append(
                        f"Action {action.action_id}: waypoint "
                        f"({action.lat}, {action.lon}) is outside the geofence "
                        f"{self.constraints.geofence}."
                    )
                # Rule 4: Global altitude ceiling.
                if action.alt_m > self.constraints.max_altitude_m:
                    errors.append(
                        f"Action {action.action_id}: altitude {action.alt_m} m "
                        f"exceeds constraint max {self.constraints.max_altitude_m} m."
                    )
                # Rule 5: Global speed ceiling.
                if action.speed_mps > self.constraints.max_speed_mps:
                    errors.append(
                        f"Action {action.action_id}: speed {action.speed_mps} m/s "
                        f"exceeds constraint max {self.constraints.max_speed_mps} m/s."
                    )

            elif isinstance(action, TakeoffAction):
                # Rule 6: Takeoff altitude vs global ceiling.
                if action.altitude_m > self.constraints.max_altitude_m:
                    errors.append(
                        f"Action {action.action_id}: takeoff altitude "
                        f"{action.altitude_m} m exceeds constraint max "
                        f"{self.constraints.max_altitude_m} m."
                    )

            elif isinstance(action, LoopAction):
                # Rule 7: Loop repeat count vs global max.
                if action.repeat_count > self.constraints.max_loop_count:
                    errors.append(
                        f"Action {action.action_id}: repeat_count "
                        f"{action.repeat_count} exceeds constraint max "
                        f"{self.constraints.max_loop_count}."
                    )
                # Rule 8: All referenced waypoint IDs must exist.
                for wid in action.waypoint_ids:
                    if wid not in goto_ids:
                        errors.append(
                            f"Action {action.action_id}: waypoint_id '{wid}' "
                            "does not reference any GotoWaypointAction in this mission."
                        )

        if errors:
            raise ValueError(
                "Mission failed structural validation:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return self
