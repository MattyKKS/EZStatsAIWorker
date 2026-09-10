"""
EZ Stats AI Worker — Test Runner
Outputs results labelled with UTC/TC identifiers matching the Test Plan.
Run from the repo root:  python run_tests.py
"""
import sys, os, unittest, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

# ── TC label map  (class_name, method_name) → (utc_label, tc_label, description) ──
TC_LABELS = {

    # ── UTC-07  run_analysis() — pipeline data model schemas ─────────────────
    ('UTC07_RunAnalysis', 'test_bbox_construction'):
        ('UTC-07', 'TC-01', 'Bounding box is created with correct corner coordinates — a result was returned with a list of detected tracks and the path to the processed video'),
    ('UTC07_RunAnalysis', 'test_bbox_cx_property'):
        ('UTC-07', 'TC-02', 'Horizontal centre of bounding box is calculated correctly'),
    ('UTC07_RunAnalysis', 'test_bbox_cy_property'):
        ('UTC-07', 'TC-03', 'Vertical centre of bounding box is calculated correctly'),
    ('UTC07_RunAnalysis', 'test_bbox_model_dump_keys'):
        ('UTC-07', 'TC-04', 'Bounding box exports all four corner values'),
    ('UTC07_RunAnalysis', 'test_video_meta_construction'):
        ('UTC-07', 'TC-05', 'Video metadata is created with correct frame rate and frame count'),
    ('UTC07_RunAnalysis', 'test_track_observation_defaults'):
        ('UTC-07', 'TC-06', 'Track observation is created with correct default values'),
    ('UTC07_RunAnalysis', 'test_event_details_default_factory'):
        ('UTC-07', 'TC-07', 'Each event gets its own separate detail dictionary'),
    ('UTC07_RunAnalysis', 'test_event_actor_target_optional'):
        ('UTC-07', 'TC-08', 'Event actor and target player IDs are optional'),
    ('UTC07_RunAnalysis', 'test_track_stats_touch_pass_shot_default_zero'):
        ('UTC-07', 'TC-09', 'Player statistics start with zero touches, passes, and shots'),
    ('UTC07_RunAnalysis', 'test_analysis_artifacts_tracks_list'):
        ('UTC-07', 'TC-10', 'Analysis result holds tracks and events as lists'),
    ('UTC07_RunAnalysis', 'test_video_meta_model_dump_json_mode_path_to_str'):
        ('UTC-07', 'TC-11', 'Video metadata exports file path as a plain text string'),
    ('UTC07_RunAnalysis', 'test_bbox_cx_cy_not_in_model_dump'):
        ('UTC-07', 'TC-12', 'Calculated centre values are not included in the exported data'),

    # ── UTC-08  Provider output contracts ─────────────────────────────────────
    ('UTC08_ProviderContracts', 'test_event_type_touch_valid'):
        ('UTC-08', 'TC-01', "Event type 'touch' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_type_pass_valid'):
        ('UTC-08', 'TC-02', "Event type 'pass' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_type_long_ball_valid'):
        ('UTC-08', 'TC-03', "Event type 'long ball' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_type_clearance_valid'):
        ('UTC-08', 'TC-04', "Event type 'clearance' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_type_interception_valid'):
        ('UTC-08', 'TC-05', "Event type 'interception' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_type_shot_attempt_valid'):
        ('UTC-08', 'TC-06', "Event type 'shot attempt' is recognised as valid"),
    ('UTC08_ProviderContracts', 'test_event_details_mutable_dict'):
        ('UTC-08', 'TC-07', 'Event detail dictionary can be updated after the event is created'),
    ('UTC08_ProviderContracts', 'test_event_all_fields_present'):
        ('UTC-08', 'TC-08', 'Event record contains all required fields'),

    # ── UTC-09  Track geometry — bounding box calculations ────────────────────
    ('UTC09_TrackGeometry', 'test_cx_center_horizontal'):
        ('UTC-09', 'TC-01', 'Horizontal centre is halfway between the left and right edges'),
    ('UTC09_TrackGeometry', 'test_cy_center_vertical'):
        ('UTC-09', 'TC-02', 'Vertical centre is halfway between the top and bottom edges'),
    ('UTC09_TrackGeometry', 'test_cx_float_precision'):
        ('UTC-09', 'TC-03', 'Horizontal centre is calculated to decimal precision'),
    ('UTC09_TrackGeometry', 'test_cy_float_precision'):
        ('UTC-09', 'TC-04', 'Vertical centre is calculated to decimal precision'),
    ('UTC09_TrackGeometry', 'test_bbox_negative_coords'):
        ('UTC-09', 'TC-05', 'Bounding box works correctly with negative coordinates'),
    ('UTC09_TrackGeometry', 'test_model_dump_contains_all_corners'):
        ('UTC-09', 'TC-06', 'Exported bounding box data contains all four corner values'),

    # ── UTC-10  Spatial layer — TrackObservation contracts ────────────────────
    ('UTC10_SpatialContracts', 'test_label_player'):
        ('UTC-10', 'TC-01', "Track label 'player' is stored correctly"),
    ('UTC10_SpatialContracts', 'test_label_ball'):
        ('UTC-10', 'TC-02', "Track label 'ball' is stored correctly"),
    ('UTC10_SpatialContracts', 'test_label_goalkeeper'):
        ('UTC-10', 'TC-03', "Track label 'goalkeeper' is stored correctly"),
    ('UTC10_SpatialContracts', 'test_confidence_range_low'):
        ('UTC-10', 'TC-04', 'A low confidence score is stored correctly'),
    ('UTC10_SpatialContracts', 'test_confidence_range_high'):
        ('UTC-10', 'TC-05', 'A high confidence score is stored correctly'),
    ('UTC10_SpatialContracts', 'test_team_id_zero_valid'):
        ('UTC-10', 'TC-06', 'Team ID of zero is stored correctly'),


    ('UTC10_SpatialContracts', 'test_team_id_one_valid'):
        ('UTC-10', 'TC-07', 'Team ID of one is stored correctly'),
    ('UTC10_SpatialContracts', 'test_bbox_accessible'):
        ('UTC-10', 'TC-08', 'Bounding box is accessible from a track observation'),
    ('UTC10_SpatialContracts', 'test_source_label_none_default'):
        ('UTC-10', 'TC-09', 'Source label defaults to empty when not provided'),
    ('UTC10_SpatialContracts', 'test_source_label_string'):
        ('UTC-10', 'TC-10', 'Source label is stored correctly when provided'),
    # ── UTC-11  Clustering output — AnalysisArtifacts ─────────────────────────
    ('UTC11_ClusterArtifacts', 'test_empty_artifacts_valid'):
        ('UTC-11', 'TC-01', 'An empty analysis result with no tracks or events is valid'),
    ('UTC11_ClusterArtifacts', 'test_processed_video_path_default_none'):
        ('UTC-11', 'TC-02', 'Processed video path defaults to empty when not set'),
    ('UTC11_ClusterArtifacts', 'test_processed_video_path_set'):
        ('UTC-11', 'TC-03', 'Processed video path is stored correctly when provided'),
    ('UTC11_ClusterArtifacts', 'test_artifacts_with_tracks_events_stats'):
        ('UTC-11', 'TC-04', 'Analysis result correctly holds tracks, events, and player statistics together'),
    ('UTC11_ClusterArtifacts', 'test_model_dump_json_mode_paths'):
        ('UTC-11', 'TC-05', 'Analysis result exports file paths as plain text strings'),

    # ── UTC-12  IO layer — write_artifacts ────────────────────────────────────
    ('UTC12_IOLayer', 'test_write_artifacts_creates_video_meta_json'):
        ('UTC-12', 'TC-01', 'Saving results creates a video metadata file'),
    ('UTC12_IOLayer', 'test_write_artifacts_creates_tracks_json'):
        ('UTC-12', 'TC-02', 'Saving results creates a tracks file'),
    ('UTC12_IOLayer', 'test_write_artifacts_creates_events_json'):
        ('UTC-12', 'TC-03', 'Saving results creates an events file'),
    ('UTC12_IOLayer', 'test_write_artifacts_creates_player_stats_json'):
        ('UTC-12', 'TC-04', 'Saving results creates a player statistics file'),
    ('UTC12_IOLayer', 'test_write_artifacts_creates_summary_json'):
        ('UTC-12', 'TC-05', 'Saving results creates a summary file'),
    ('UTC12_IOLayer', 'test_write_artifacts_creates_match_report_json'):
        ('UTC-12', 'TC-06', 'Saving results creates a match report file'),
    ('UTC12_IOLayer', 'test_match_report_has_required_keys'):
        ('UTC-12', 'TC-07', 'Match report contains all required sections'),
    ('UTC12_IOLayer', 'test_match_report_summary_keys'):
        ('UTC-12', 'TC-08', 'Match report summary section contains all required fields'),
    ('UTC12_IOLayer', 'test_match_report_duration_s'):
        ('UTC-12', 'TC-09', 'Match duration in the report is calculated correctly'),
    ('UTC12_IOLayer', 'test_events_json_is_list'):
        ('UTC-12', 'TC-10', 'Events file contains a list'),
    ('UTC12_IOLayer', 'test_tracks_json_is_list'):
        ('UTC-12', 'TC-11', 'Tracks file contains a list'),
    ('UTC12_IOLayer', 'test_player_stats_json_is_list'):
        ('UTC-12', 'TC-12', 'Player statistics file contains a list'),
    ('UTC12_IOLayer', 'test_pass_network_in_match_report'):
        ('UTC-12', 'TC-13', 'Match report includes a pass network section'),
    ('UTC12_IOLayer', 'test_write_artifacts_accepts_path_object'):
        ('UTC-12', 'TC-14', 'Results can be saved using a folder path object'),

    # ── UTC-12  IO layer — load_video_meta ────────────────────────────────────
    ('UTC12b_LoadVideoMeta', 'test_nonexistent_file_raises'):
        ('UTC-12', 'TC-15', 'Reading a video file that does not exist returns a file-not-found error'),
    ('UTC12b_LoadVideoMeta', 'test_nonexistent_path_is_not_silently_ignored'):
        ('UTC-12', 'TC-16', 'A missing video file path is not silently ignored'),

    # ── UTC-13  detect_events() state machine ─────────────────────────────────
    ('UTC13_DetectEvents', 'test_empty_tracks_returns_empty_list'):
        ('UTC-13', 'TC-01', 'No tracking data produces an empty event list'),
    ('UTC13_DetectEvents', 'test_no_ball_track_returns_empty_list'):
        ('UTC-13', 'TC-02', 'No ball in tracking data produces an empty event list'),
    ('UTC13_DetectEvents', 'test_only_ball_no_players_returns_empty'):
        ('UTC-13', 'TC-03', 'Ball present but no players produces an empty event list'),
    ('UTC13_DetectEvents', 'test_output_is_list_of_events'):
        ('UTC-13', 'TC-04', 'Event detection returns a list of event records'),
    ('UTC13_DetectEvents', 'test_all_emitted_event_types_are_valid'):
        ('UTC-13', 'TC-05', 'All detected events use one of the six recognised event types'),
    ('UTC13_DetectEvents', 'test_events_chronologically_ordered'):
        ('UTC-13', 'TC-06', 'Detected events are returned in time order, earliest first'),
    ('UTC13_DetectEvents', 'test_event_time_seconds_consistent_with_frame_and_fps'):
        ('UTC-13', 'TC-07', 'Event time in seconds matches the frame number and frame rate'),
    ('UTC13_DetectEvents', 'test_ball_far_from_all_players_produces_no_touch'):
        ('UTC-13', 'TC-08', 'Ball far from all players does not produce a touch event'),
    ('UTC13_DetectEvents', 'test_actor_track_id_in_emitted_events'):
        ('UTC-13', 'TC-09', 'Each event records the ID of the player who acted on the ball'),
    ('UTC13_DetectEvents', 'test_result_has_frame_index_int'):
        ('UTC-13', 'TC-10', 'Each event records the frame number as a whole number'),

    # ── UTC-14  build_track_stats() ───────────────────────────────────────────
    ('UTC14_BuildTrackStats', 'test_empty_tracks_returns_empty_list'):
        ('UTC-14', 'TC-01', 'No tracking data produces an empty statistics list'),
    ('UTC14_BuildTrackStats', 'test_non_player_tracks_excluded'):
        ('UTC-14', 'TC-02', 'The ball track is excluded from player statistics'),
    ('UTC14_BuildTrackStats', 'test_player_track_included'):
        ('UTC-14', 'TC-03', 'Each player track produces one statistics record'),
    ('UTC14_BuildTrackStats', 'test_frame_count_correct'):
        ('UTC-14', 'TC-04', 'Frame count in statistics matches the number of frames the player appeared in'),
    ('UTC14_BuildTrackStats', 'test_touch_count_from_events'):
        ('UTC-14', 'TC-05', 'Touch count matches the number of touch events recorded for that player'),
    ('UTC14_BuildTrackStats', 'test_pass_count_from_events'):
        ('UTC-14', 'TC-06', 'Pass count matches the number of pass events recorded for that player'),
    ('UTC14_BuildTrackStats', 'test_shot_count_from_events'):
        ('UTC-14', 'TC-07', 'Shot count matches the number of shot events recorded for that player'),
    ('UTC14_BuildTrackStats', 'test_approx_distance_zero_for_stationary'):
        ('UTC-14', 'TC-08', 'A stationary player has zero distance covered'),
    ('UTC14_BuildTrackStats', 'test_sorted_by_track_id'):
        ('UTC-14', 'TC-09', 'Statistics list is sorted by player ID from lowest to highest'),
    ('UTC14_BuildTrackStats', 'test_multiple_players_separate_stats'):
        ('UTC-14', 'TC-10', 'Each player gets their own separate statistics record'),
    ('UTC14_BuildTrackStats', 'test_avg_speed_non_negative'):
        ('UTC-14', 'TC-11', 'Average speed is always zero or greater'),
    ('UTC14_BuildTrackStats', 'test_result_items_are_track_stats'):
        ('UTC-14', 'TC-12', 'Each item in the statistics list is a valid player statistics record'),

    # ── UTC-15  _build_match_report() ─────────────────────────────────────────
    ('UTC15_MatchReport', 'test_tc01_match_report_has_required_keys'):
        ('UTC-15', 'TC-01', 'Match report contains all required top-level sections'),
    ('UTC15_MatchReport', 'test_tc02_match_report_summary_keys'):
        ('UTC-15', 'TC-02', 'Match report summary section contains all required fields'),
    ('UTC15_MatchReport', 'test_tc03_match_report_duration_s'):
        ('UTC-15', 'TC-03', 'Match duration is calculated correctly from frame count and frame rate'),
    ('UTC15_MatchReport', 'test_tc04_pass_network_in_match_report'):
        ('UTC-15', 'TC-04', 'Match report includes a pass network section'),
    ('UTC15_MatchReport', 'test_tc05_events_json_is_list'):
        ('UTC-15', 'TC-05', 'Events section in the match report is a list'),

    # ── UTC-16  fuse_events() / event_spotter ─────────────────────────────────
    ('UTC16_FuseAndSpotter', 'test_empty_inputs_returns_empty'):
        ('UTC-16', 'TC-01', 'No input events produces an empty merged list'),
    ('UTC16_FuseAndSpotter', 'test_rule_only_no_lstm_passthrough'):
        ('UTC-16', 'TC-02', 'Rule-based events are passed through unchanged when no AI events are given'),
    ('UTC16_FuseAndSpotter', 'test_lstm_goal_upgrades_shot_attempt'):
        ('UTC-16', 'TC-03', 'An AI goal detection upgrades a nearby shot attempt event'),
    ('UTC16_FuseAndSpotter', 'test_lstm_cross_upgrades_pass'):
        ('UTC-16', 'TC-04', 'An AI cross detection upgrades a nearby pass event'),
    ('UTC16_FuseAndSpotter', 'test_output_is_sorted_by_frame'):
        ('UTC-16', 'TC-05', 'Merged event list is sorted by frame number, earliest first'),
    ('UTC16_FuseAndSpotter', 'test_lstm_only_events_included'):
        ('UTC-16', 'TC-06', 'AI-only events with no matching rule-based event are included in the output'),
    ('UTC16_FuseAndSpotter', 'test_result_is_list_of_events'):
        ('UTC-16', 'TC-07', 'Merged result is a list of event records'),
    ('UTC16_FuseAndSpotter', 'test_upgrade_does_not_duplicate_original'):
        ('UTC-16', 'TC-08', 'Upgrading an event does not create a duplicate of the original'),
    ('UTC16_FuseAndSpotter', 'test_out_of_window_lstm_not_upgrade'):
        ('UTC-16', 'TC-09', 'An AI event too far from a rule-based event does not trigger an upgrade'),
}

# ── Custom result collector ───────────────────────────────────────────────────
class LabelledResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.results = []

    def _key(self, test):
        cls = type(test).__name__
        mth = test._testMethodName
        return TC_LABELS.get((cls, mth), (cls, mth, mth.replace('_', ' ')))

    def addSuccess(self, test):
        self.results.append((*self._key(test), 'PASS'))

    def addFailure(self, test, err):
        self.results.append((*self._key(test), 'FAIL'))

    def addError(self, test, err):
        self.results.append((*self._key(test), 'ERROR'))

    def addSkip(self, test, reason):
        self.results.append((*self._key(test), 'SKIP'))

# ── Run and print ─────────────────────────────────────────────────────────────
import test_all_units

loader = unittest.TestLoader()
suite  = loader.loadTestsFromModule(test_all_units)

result = LabelledResult()
t0 = time.time()
suite.run(result)
elapsed = time.time() - t0

print()
print('=' * 72)
print('  EZ Stats AI Worker — Unit Test Results')
print('=' * 72)

def sort_key(r):
    utc, tc = r[0], r[1]
    utc_num = int(utc.split('-')[1]) if utc.startswith('UTC-') else 99
    tc_num = int(tc.split('-')[1]) if tc.startswith('TC-') else 99
    return (utc_num, tc_num)

sorted_results = sorted(result.results, key=sort_key)

current_utc = None
for (utc, tc, desc, status) in sorted_results:
    if utc != current_utc:
        print(f'\n  {utc}')
        current_utc = utc
    marker = 'v' if status == 'PASS' else 'X'
    print(f'    [{marker}] {tc}: {desc} ... {status}')

total  = len(result.results)
passed = sum(1 for r in result.results if r[3] == 'PASS')
failed = total - passed

print()
print('=' * 72)
print(f'  Ran {total} tests in {elapsed:.3f}s')
if failed == 0:
    print(f'  ALL {total} TESTS PASSED')
else:
    print(f'  {passed} passed  |  {failed} FAILED')
print('=' * 72)
print()
