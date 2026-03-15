# Hard Example Collection

Last updated: `2026-03-15 00:10` Bangkok time

This is the shortlist for what to collect before the V2 retrain.

## Collect These First

### Goalkeeper examples

Collect frames where:

- the goalkeeper is fully visible
- the goalkeeper is partly visible
- the goalkeeper stands near the penalty area
- the goalkeeper and referee have similar shirt colors

Target:

- at least `40` to `80` strong goalkeeper frames

### Referee examples

Collect frames where:

- the referee is central
- the referee is near clustered players
- the referee is close in color to one team or the goalkeeper

Target:

- at least `40` to `80` strong referee frames

### Sideline player examples

Collect frames where:

- a player is close to the left touchline
- a player is close to the right touchline
- only part of the player is visible
- the player is cut by the image border

Target:

- at least `60` to `120` strong sideline-player frames

## Best Clip Moments To Use

From your current benchmark clip, prioritize:

- goal kicks
- long clearances
- wide midfield views
- players hugging the sideline
- moments where the referee and players overlap visually

## Minimum Useful Hard Set

If time is limited, a useful first hard set is:

- `50` goalkeeper/referee frames total
- `50` sideline-player frames

That is enough to make a meaningful first retrain attempt.

## Annotation Reminder

Keep the labels exactly as:

1. `ball`
2. `goalkeeper`
3. `player`
4. `referee`
