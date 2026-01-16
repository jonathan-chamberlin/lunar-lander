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

### Landing Outcomes (Safe)

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `LANDED_PERFECTLY` | Ideal landing | Both legs, velocity < 0.5, \|angle\| < 0.2, \|x\| < 0.2 |
| `LANDED_SOFTLY` | Good landing | Both legs, velocity < 1.0, \|angle\| < 0.3 |
| `LANDED_HARD` | Heavy impact landing | Both legs, velocity < 2.0 |
| `LANDED_TILTED` | Landed at angle | Both legs, \|angle\| > 0.3 |
| `TOUCHED_DOWN_ONE_LEG` | Single leg contact | One leg contact at termination |
| `LANDED_WITH_DRIFT` | Landed with horizontal drift | Both legs, \|vx\| > 0.1 |
| `TIMED_OUT_ON_GROUND` | Timeout with legs down | Truncated with both legs on ground |

### Crash Outcomes

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `CRASHED_SPINNING` | Impact while rotating | Terminated, \|angular_vel\| > 2.0 rad/s |
| `CRASHED_SIDEWAYS` | Impact nearly horizontal | Terminated, \|angle\| > 1.0 |
| `CRASHED_TILTED` | High-speed tilted impact | Terminated, \|angle\| > 0.5 |
| `CRASHED_HIGH_VELOCITY` | High-speed vertical impact | Terminated, \|angle\| <= 0.5, \|vy\| > 1.0 |
| `CRASHED_OTHER` | Other crash scenarios | Terminated, doesn't match above criteria |

### Timeout Outcomes

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `TIMED_OUT_HOVERING` | Truncated while hovering | Truncated, low velocity, altitude stable |
| `TIMED_OUT_DESCENDING` | Truncated while descending | Truncated, vy < -0.1 |
| `TIMED_OUT_ASCENDING` | Truncated while ascending | Truncated, vy > 0.1 |

### Flew Off Screen Outcomes

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `FLEW_OFF_TOP` | Ascended beyond boundary | Final y > 1.5 |
| `FLEW_OFF_LEFT` | Drifted beyond left | Final x < -1.0, not tilted |
| `FLEW_OFF_RIGHT` | Drifted beyond right | Final x > 1.0, not tilted |
| `FLEW_OFF_LEFT_TILTED` | Tilted and flew left | \|angle\| > 0.5, moving/positioned left |
| `FLEW_OFF_RIGHT_TILTED` | Tilted and flew right | \|angle\| > 0.5, moving/positioned right |

---

## Vertical Flight Patterns

Describes the vertical movement behavior throughout the episode.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `CONTROLLED_DESCENT` | Steady, moderate descent | Mean vy between -0.5 and -0.1, low variance |
| `SLOW_DESCENT` | Very gradual descent | Mean vy between -0.1 and 0, >50% descending |
| `RAPID_DESCENT` | Fast downward movement | Mean vy < -0.5 |
| `FREEFALL` | Minimal thrust descent | Mean main thrust < 0.1, mean vy < -0.3 |
| `STABLE_ALTITUDE` | Altitude held constant | y variance < 0.1 |
| `ASCENDED` | Net upward movement | Final y - initial y > 0.3 |
| `YO_YO_PATTERN` | Up/down oscillation | vy sign changed > 4 times |
| `STALLED_THEN_FELL` | Hovered then dropped | Stable first 30%, then mean vy < -0.4 |
| `LATE_BRAKING` | Late heavy thrust | Early thrust < 0.2, final 20% thrust > 0.5 |
| `CONTINUOUS_BURN` | Sustained main engine | Main thrust > 0.3 for >80% of episode |
| `HOVER_NEAR_GROUND_TIMEOUT` | Timeout hovering low | Truncated, >50% at low altitude, low velocity |

---

## Horizontal Movement

Describes lateral drift and corrections.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `STAYED_CENTERED` | Maintained center position | \|x\| < 0.3 throughout |
| `DRIFTED_LEFT` | Net leftward movement | Final x < -0.3, started centered |
| `DRIFTED_RIGHT` | Net rightward movement | Final x > 0.3, started centered |
| `RETURNED_TO_CENTER` | Corrected drift | Max \|x\| > 0.3, final \|x\| < 0.2 |
| `HORIZONTAL_OSCILLATION` | Side-to-side oscillation | vx sign changed > 6 times |
| `STRONG_LATERAL_VELOCITY` | Excessive horizontal speed | Max \|vx\| > 1.0 |

---

## Orientation and Stability

Describes rotation and angular stability.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `STAYED_UPRIGHT` | Maintained vertical | \|angle\| < 0.2 throughout |
| `SLIGHT_LEFT_LEAN` | Minor leftward tilt | Mean angle between -0.5 and -0.1 |
| `SLIGHT_RIGHT_LEAN` | Minor rightward tilt | Mean angle between 0.1 and 0.5 |
| `HEAVY_LEFT_TILT` | Significant left rotation | Min angle < -0.5 |
| `HEAVY_RIGHT_TILT` | Significant right rotation | Max angle > 0.5 |
| `FLIPPED_OVER` | Rotated past horizontal | \|angle\| > 1.57 (pi/2) |
| `EXCEEDED_SPIN_RATE` | Rapid rotation | \|angular_vel\| > 2.0 rad/s |
| `RECOVERED_FROM_TILT` | Corrected from tilt | Max \|angle\| > 0.5, final \|angle\| < 0.2 |
| `PROGRESSIVE_TILT` | Steadily increasing tilt | Late \|angle\| > early \|angle\| + 0.2 |
| `WOBBLING` | Left/right oscillation | Angle sign changed > 8 times |

---

## Thruster Usage Patterns

Describes how the agent used its engines.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `MAIN_THRUST_HEAVY` | Strong main engine | Mean main action > 0.5 |
| `MAIN_THRUST_MODERATE` | Moderate main engine | Mean main action 0.2-0.5 |
| `MAIN_THRUST_LIGHT` | Light main engine | Mean main action 0.05-0.2 |
| `MAIN_THRUST_NONE` | Main engine off | Mean main action < 0.05 |
| `SIDE_THRUST_LEFT_BIAS` | Left correction bias | Mean side action < -0.3 |
| `SIDE_THRUST_RIGHT_BIAS` | Right correction bias | Mean side action > 0.3 |
| `SIDE_THRUST_BALANCED` | Symmetric side thrust | \|mean side action\| < 0.1 |
| `ERRATIC_THRUST` | High thrust variance | Action std > 0.5 |
| `SMOOTH_THRUST` | Consistent thrust | Action std < 0.2 |
| `FULL_THROTTLE_BURST` | Maximum thrust periods | Main = 1.0 for >= 10 consecutive steps |

---

## Leg Contact Events

Describes touchdown and ground contact behavior.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `NO_CONTACT_MADE` | Never touched ground | Neither leg ever contacted |
| `SCRAPED_LEFT_LEG` | Left leg only | Left leg touched, right didn't |
| `SCRAPED_RIGHT_LEG` | Right leg only | Right leg touched, left didn't |
| `TOUCHED_DOWN_CLEAN` | Both legs together | Both legs contacted within 5 steps |
| `BOUNCED` | Contact then lost | Had contact, lost it, regained it |
| `MULTIPLE_TOUCHDOWNS` | Repeated contact cycles | Contact established multiple times |
| `PROLONGED_ONE_LEG` | Extended single-leg | One leg contact > 20 steps |

---

## Episode Efficiency

Classifies episode duration.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `VERY_SHORT_EPISODE` | Extremely brief | < 50 steps |
| `SHORT_EPISODE` | Quick run | 50-100 steps |
| `STANDARD_EPISODE` | Typical duration | 100-300 steps |
| `LONG_EPISODE` | Extended run | 300-600 steps |
| `VERY_LONG_EPISODE` | Prolonged run | > 600 steps |

---

## Trajectory Patterns

Describes the overall flight path shape.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `DIRECT_APPROACH` | Straight descent | Low x variance, descending |
| `CURVED_APPROACH` | Lateral movement descent | High x variance, descending |
| `SPIRAL_DESCENT` | Rotating while descending | Descending, mean \|angular_vel\| > 0.3 |
| `ZIGZAG_DESCENT` | Alternating corrections | Descending, x crossed mean > 4 times |
| `SUICIDE_BURN` | Late heavy braking | High velocity until 85%, then thrust > 0.6 |
| `GRADUAL_SLOWDOWN` | Steadily decreasing speed | \|vy\| decreased each third of episode |

---

## Altitude Milestones

Tracks altitude-related events.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `REACHED_LOW_ALTITUDE` | Descended near ground | y < 0.25 at some point |
| `STAYED_HIGH` | Never got low | Min y > 0.5 |
| `PEAKED_ABOVE_START` | Ascended above start | Max y > initial y + 0.1 |
| `GROUND_APPROACH_ABORT` | Got low then ascended | Min y < 0.3, then rose > 0.2 |

---

## Critical Moments

Describes phase-based behavior patterns.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `NEVER_STABILIZED` | Never achieved stability | Never had 20 steps with \|angle\| < 0.3 |
| `CONTROLLED_THROUGHOUT` | Stable entire episode | \|angle\| < 0.4 and \|angular_vel\| < 0.5 throughout |
| `LOST_CONTROL_LATE` | Late control loss | Early angular_vel < 0.3, late > 0.8 |
| `OVERCORRECTED_TO_CRASH` | Oscillation led to crash | Angle variance doubled in second half before crash |

---

## Timeout Behaviors

Special behaviors detected on truncated (timed out) episodes.

| Behavior | Description | Detection Logic |
|----------|-------------|-----------------|
| `HOVER_NEAR_GROUND_TIMEOUT` | Hovering low at timeout | >50% at low altitude, low velocity |
| `HOVERED_OVER_GOAL_TIMEOUT` | Hovering centered at timeout | >50% low altitude, >50% centered, low velocity |

---

## Threshold Constants Reference

| Category | Constant | Value | Unit |
|----------|----------|-------|------|
| Velocity | CRASH_VELOCITY | 2.0 | m/s |
| Velocity | HARD_LANDING_VELOCITY | 1.0 | m/s |
| Velocity | SOFT_LANDING_VELOCITY | 0.5 | m/s |
| Velocity | HOVER_VELOCITY | 0.1 | m/s |
| Angle | UPRIGHT | 0.2 | rad |
| Angle | SLIGHT_TILT | 0.3 | rad |
| Angle | TILTED | 0.5 | rad |
| Angle | SIDEWAYS | 1.0 | rad |
| Angle | FLIPPED | 1.57 | rad |
| Angular Vel | SPINNING | 2.0 | rad/s |
| Position | OFF_SCREEN_Y | 1.5 | - |
| Position | OFF_SCREEN_X | 1.0 | - |
| Position | LOW_ALTITUDE | 0.25 | - |
| Position | MEDIUM_ALTITUDE | 0.5 | - |
| Position | CENTERED | 0.3 | - |
| Position | WELL_CENTERED | 0.2 | - |
| Thrust | THRUST_HEAVY | 0.5 | - |
| Thrust | THRUST_MODERATE_LOW | 0.2 | - |
| Thrust | THRUST_NONE | 0.05 | - |
| Thrust | THRUST_BIAS | 0.3 | - |
| Thrust | THRUST_BALANCED | 0.1 | - |
| Variance | LOW_VARIANCE | 0.1 | - |
| Variance | HIGH_VARIANCE | 0.5 | - |
| Variance | LOW_ACTION_VARIANCE | 0.2 | - |
| Variance | HIGH_ACTION_VARIANCE | 0.5 | - |
| Steps | VERY_SHORT | 50 | steps |
| Steps | SHORT | 100 | steps |
| Steps | LONG | 300 | steps |
| Steps | VERY_LONG | 600 | steps |
| Events | SIGN_CHANGE_OSCILLATION | 6 | count |
| Events | SIGN_CHANGE_YO_YO | 4 | count |
| Events | SIGN_CHANGE_WOBBLE | 8 | count |
| Events | CONSECUTIVE_FULL_THROTTLE | 10 | steps |
| Events | STABLE_STEPS | 20 | steps |

---

## Example Output

```
Run 47 ✗ CRASHED_TILTED ❌ Didn't land safely 🥕 Reward: -124.3 (env: -180.2 / shaped: +55.9)
    ↕ RAPID_DESC
    ↔ DRIFT_R
    ↻ TILT_R
    🔥 MAIN_LIGHT
    👣 NO_CONTACT

Run 48 ✓ LANDED_SOFTLY ✅ Landed Safely 🥕 Reward: 215.7 (env: 198.4 / shaped: +17.3)
    ↕ CTRL_DESC
    ↔ CENTERED
    ↻ UPRIGHT, CONTROLLED
    🔥 MAIN_MOD, SMOOTH
    👣 CLEAN_TD, LOW_ALT

Run 49 ✗ TIMED_OUT_HOVERING ❌ Didn't land safely 🥕 Reward: -45.2 (env: -102.1 / shaped: +56.9)
    ↕ HOVER, HOVER_GROUND
    ↔ CENTERED
    ↻ UPRIGHT
    🔥 MAIN_HEAVY
    👣 NO_CONTACT, HIGH
```
