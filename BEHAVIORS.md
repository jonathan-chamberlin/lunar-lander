# Behavior Analysis Reference

This document describes all behavior patterns detected by the Lunar Lander behavior analysis system. Multiple behaviors can be reported for each episode, providing a comprehensive description of what happened during the run.

## State Vector Reference

The LunarLander environment provides an 8-dimensional observation:

| Index | Value | Description |
|-------|-------|-------------|
| 0 | x | Horizontal position |
| 1 | y | Vertical position (altitude) |
| 2 | vx | Horizontal velocity |
| 3 | vy | Vertical velocity (negative = descending) |
| 4 | angle | Rotation angle (radians, 0 = upright) |
| 5 | angular_vel | Angular velocity |
| 6 | leg1 | Left leg contact (1.0 = touching ground) |
| 7 | leg2 | Right leg contact (1.0 = touching ground) |

---

## Episode Outcome (Mutually Exclusive)

One outcome is assigned per episode based on terminal conditions.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `LANDED_PERFECTLY` | Ideal landing with both legs, centered, slow, upright | Both legs contact, velocity < 0.5, angle < 0.1, |x| < 0.2 |
| `LANDED_SOFTLY` | Good landing with acceptable parameters | Both legs contact, velocity < 1.0, angle < 0.3 |
| `LANDED_HARD` | Landed but with excessive impact velocity | Both legs contact, velocity 1.0-2.0 |
| `LANDED_TILTED` | Landed but at significant angle | Both legs contact, angle > 0.3 |
| `LANDED_ONE_LEG` | Only achieved single leg contact | One leg contact at termination |
| `CRASHED_FAST_VERTICAL` | High-speed vertical impact while upright | Terminated, |vy| > 2.0, angle < 0.3 |
| `CRASHED_FAST_TILTED` | High-speed impact while tilted | Terminated, velocity > 2.0, angle > 0.5 |
| `CRASHED_SIDEWAYS` | Impact while nearly horizontal | Terminated, |angle| > 1.0 |
| `CRASHED_SPINNING` | Impact while rotating rapidly | Terminated, |angular_vel| > 1.5 |
| `FLEW_OFF_TOP` | Ascended beyond screen boundary | Final y > 1.5, moving upward |
| `FLEW_OFF_LEFT` | Drifted beyond left boundary | Final x < -1.0, not tilted |
| `FLEW_OFF_RIGHT` | Drifted beyond right boundary | Final x > 1.0, not tilted |
| `FLEW_OFF_LEFT_TILTED` | Tilted and flew off left side | Tilted > 0.5 rad, moving/drifting left |
| `FLEW_OFF_RIGHT_TILTED` | Tilted and flew off right side | Tilted > 0.5 rad, moving/drifting right |
| `TIMED_OUT_HOVERING` | Episode truncated while maintaining altitude | Truncated, low velocity, altitude stable |
| `TIMED_OUT_DESCENDING` | Episode truncated while actively descending | Truncated, vy < -0.1 |
| `TIMED_OUT_ASCENDING` | Episode truncated while moving upward | Truncated, vy > 0.1 |

---

## Vertical Flight Patterns

Describes the vertical movement behavior throughout the episode.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `CONTROLLED_DESCENT` | Steady, moderate-speed descent | Mean vy between -0.5 and -0.1, low vy variance |
| `SLOW_DESCENT` | Very gradual downward movement | Mean vy between -0.1 and 0, >50% time descending |
| `RAPID_DESCENT` | Fast downward movement | Mean vy < -0.5 |
| `FREEFALL` | Minimal thrust, gravity-dominated descent | Long periods matching gravity, main thrust < 0.1 |
| `HOVER_MAINTAINED` | Altitude held relatively constant | y variance < 0.1 across episode |
| `ASCENDED` | Net upward movement | Final y - initial y > 0.3 |
| `YO_YO_PATTERN` | Repeated up/down oscillation | vy changed sign > 4 times |
| `STALLED_THEN_FELL` | Hovered then dropped | Stable altitude >30% of episode, then rapid descent |
| `LATE_BRAKING` | Minimal thrust until final approach | Low thrust until final 20%, then heavy thrust |
| `CONTINUOUS_BURN` | Sustained main engine use | Main thrust > 0.3 for >80% of episode |

---

## Horizontal Movement

Describes lateral drift and corrections.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `STAYED_CENTERED` | Maintained position near landing zone | x stayed within +/-0.3 throughout |
| `DRIFTED_LEFT` | Net movement to the left | Final x < -0.3, started near center |
| `DRIFTED_RIGHT` | Net movement to the right | Final x > 0.3, started near center |
| `RETURNED_TO_CENTER` | Corrected drift back to center | Drifted but ended within +/-0.2 |
| `HORIZONTAL_OSCILLATION` | Overcorrecting side-to-side | vx changed sign > 6 times |
| `STRONG_LATERAL_VELOCITY` | Excessive horizontal speed at some point | Max |vx| > 1.0 |

---

## Orientation and Stability

Describes rotation and angular stability.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `STAYED_UPRIGHT` | Maintained vertical orientation | Angle stayed within +/-0.2 throughout |
| `SLIGHT_LEFT_LEAN` | Consistent minor leftward tilt | Mean angle between -0.5 and -0.1 |
| `SLIGHT_RIGHT_LEAN` | Consistent minor rightward tilt | Mean angle between 0.1 and 0.5 |
| `HEAVY_LEFT_TILT` | Significant leftward rotation | Angle went below -0.5 |
| `HEAVY_RIGHT_TILT` | Significant rightward rotation | Angle went above 0.5 |
| `FLIPPED_OVER` | Rotated past horizontal | |angle| exceeded pi/2 (1.57) |
| `SPINNING_UNCONTROLLED` | Rapid uncontrolled rotation | |angular_vel| exceeded 1.5 |
| `RECOVERED_FROM_TILT` | Started tilted, ended upright | Max |angle| > 0.5, final |angle| < 0.2 |
| `PROGRESSIVE_TILT` | Tilt steadily increased | |angle| trend positive |
| `WOBBLING` | Oscillating left/right | Angle changed sign > 8 times |

---

## Thruster Usage Patterns

Describes how the agent used its engines.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `MAIN_THRUST_HEAVY` | Strong sustained main engine use | Mean main action > 0.5 |
| `MAIN_THRUST_MODERATE` | Moderate main engine use | Mean main action 0.2-0.5 |
| `MAIN_THRUST_LIGHT` | Minimal main engine use | Mean main action < 0.2 |
| `MAIN_THRUST_NONE` | Main engine essentially off | Mean main action < 0.05 |
| `SIDE_THRUST_LEFT_BIAS` | Predominant left correction | Mean side action < -0.3 |
| `SIDE_THRUST_RIGHT_BIAS` | Predominant right correction | Mean side action > 0.3 |
| `SIDE_THRUST_BALANCED` | Symmetric side thruster use | |mean side action| < 0.1 |
| `ERRATIC_THRUST` | High variance in thrust commands | Action std > 0.5 |
| `SMOOTH_THRUST` | Consistent thrust commands | Action std < 0.2 |
| `FULL_THROTTLE_BURST` | Periods of maximum thrust | Main thrust = 1.0 for > 10 consecutive steps |

---

## Leg Contact Events

Describes touchdown and ground contact behavior.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `BOUNCED` | Made contact then lost it | Had leg contact, then lost contact |
| `SCRAPED_LEFT_LEG` | Left leg contacted without right | Left leg touched while right didn't |
| `SCRAPED_RIGHT_LEG` | Right leg contacted without left | Right leg touched while left didn't |
| `TOUCHED_DOWN_CLEAN` | Both legs contacted together | Both legs contacted within 5 steps |
| `PROLONGED_ONE_LEG` | Extended single-leg contact | One leg contact for > 20 steps |
| `MULTIPLE_TOUCHDOWNS` | Repeated contact/release cycles | Contact established >1 time |
| `NO_CONTACT_MADE` | Never achieved ground contact | Neither leg ever touched |

---

## Episode Efficiency

Classifies episode duration and resource usage.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `VERY_SHORT_EPISODE` | Extremely brief run | < 50 steps |
| `SHORT_EPISODE` | Quick run | 50-100 steps |
| `STANDARD_EPISODE` | Typical duration | 100-300 steps |
| `LONG_EPISODE` | Extended run | 300-600 steps |
| `VERY_LONG_EPISODE` | Prolonged run | > 600 steps |

---

## Trajectory Patterns

Describes the overall flight path shape.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `DIRECT_APPROACH` | Straight descent with minimal drift | Low x variance, steady y decrease |
| `CURVED_APPROACH` | Significant lateral movement during descent | High x variance while descending |
| `SPIRAL_DESCENT` | Rotating while descending | Descending with consistent angular velocity |
| `ZIGZAG_DESCENT` | Alternating lateral corrections | Multiple x direction changes while descending |
| `SUICIDE_BURN` | High-speed descent with late braking | High velocity until final 15%, then heavy thrust |
| `GRADUAL_SLOWDOWN` | Steadily decreasing velocity | |vy| decreased consistently |

---

## Altitude Milestones

Tracks altitude-related events.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `REACHED_LOW_ALTITUDE` | Descended near ground level | y went below 0.25 at some point |
| `STAYED_HIGH` | Never achieved low altitude | Never went below y = 0.5 |
| `PEAKED_ABOVE_START` | Ascended above initial height | Max y > initial y |
| `GROUND_APPROACH_ABORT` | Got low then ascended | y < 0.3, then y increased > 0.2 |

---

## Critical Moments

Describes phase-based behavior patterns.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `GOOD_START_BAD_FINISH` | Started well, ended poorly | First half reward > 0, second half < -50 |
| `BAD_START_RECOVERED` | Poor start, successful end | First 25% negative trend, episode succeeded |
| `OVERCORRECTED_TO_CRASH` | Increasing oscillation led to crash | Oscillation amplitude increased before crash |
| `LOST_CONTROL_LATE` | Stable flight then control loss | Stable >70% then angular velocity spiked |
| `NEVER_STABILIZED` | Never achieved stable flight | Never had 20 steps with |angle| < 0.3 |
| `CONTROLLED_THROUGHOUT` | Maintained stability entire episode | |angle| < 0.4 and |angular_vel| < 0.5 throughout |

---

## Threshold Constants Reference

| Category | Constant | Value | Unit |
|----------|----------|-------|------|
| Velocity | CRASH_VELOCITY | 2.0 | m/s |
| Velocity | HARD_LANDING_VELOCITY | 1.0 | m/s |
| Velocity | SOFT_LANDING_VELOCITY | 0.5 | m/s |
| Angle | UPRIGHT_THRESHOLD | 0.2 | rad |
| Angle | TILTED_THRESHOLD | 0.5 | rad |
| Angle | SIDEWAYS_THRESHOLD | 1.0 | rad |
| Angle | FLIPPED_THRESHOLD | 1.57 | rad |
| Position | OFF_SCREEN_Y | 1.5 | - |
| Position | OFF_SCREEN_X | 1.0 | - |
| Position | LOW_ALTITUDE | 0.25 | - |
| Position | CENTERED_THRESHOLD | 0.3 | - |
| Angular | SPINNING_THRESHOLD | 1.5 | rad/s |
| Steps | VERY_SHORT | 50 | steps |
| Steps | SHORT | 100 | steps |
| Steps | LONG | 300 | steps |
| Steps | VERY_LONG | 600 | steps |

---

## Example Output

```
Run 47 | FAILURE | Reward: -124.3 (env: -180.2, shaped: 55.9)
  Behaviors: CRASHED_FAST_TILTED | RAPID_DESCENT, DRIFTED_RIGHT, HEAVY_RIGHT_TILT,
             MAIN_THRUST_LIGHT, NO_CONTACT_MADE, SHORT_EPISODE

Run 48 | SUCCESS | Reward: 215.7 (env: 198.4, shaped: 17.3)
  Behaviors: LANDED_SOFTLY | CONTROLLED_DESCENT, STAYED_CENTERED, STAYED_UPRIGHT,
             MAIN_THRUST_MODERATE, TOUCHED_DOWN_CLEAN, STANDARD_EPISODE, REACHED_LOW_ALTITUDE

Run 49 | FAILURE | Reward: -45.2 (env: -102.1, shaped: 56.9)
  Behaviors: TIMED_OUT_HOVERING | HOVER_MAINTAINED, STAYED_CENTERED, STAYED_UPRIGHT,
             MAIN_THRUST_HEAVY, NO_CONTACT_MADE, VERY_LONG_EPISODE, STAYED_HIGH
```
