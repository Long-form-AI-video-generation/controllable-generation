# Synthetic pilot pose records

These tiny JSON inputs exercise [pose-pilot-v1](../../../docs/pose/PILOT_RECORD_CONTRACT.md). They contain synthetic observations and placeholder hashes; they are not dataset samples or anatomical ground truth.

| Fixture | Expected result |
|---|---|
| `valid_full.json` | Accept a selected subject with 17 valid joints, including border coordinates |
| `valid_partial.json` | Accept low-confidence coordinates, a missing location and null confidence, all invalid |
| `valid_absent.json` | Accept an absent subject with no observed joints |
| `valid_ambiguous.json` | Accept unresolved subject selection with no observed joints |
| `valid_repeated.json` | Accept the identical final observation repeated for a short interval |
| `invalid_joint_count.json` | Reject 16 joints instead of 17 |
| `invalid_coordinate.json` | Reject x=1.01 |
| `invalid_validity.json` | Reject a valid flag on a joint below the confidence threshold |
| `invalid_repeated.json` | Reject differing observations at the same source frame index |

The validator is in `src/pose_models/record_validation.py`. Its tests use these
examples by default; no Drive download or environment variable is required.
