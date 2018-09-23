form audio_info_extraction
# TODO ricontrollare input
	text wav_file
	text global_report_file
	text pitch_file
	text pitch_tier_file
	text point_process_file
	text intensity_file
	text intensity_tier_file
	real pitch_min
	real pitch_max
	real max_period_factor
	real max_amplitude_factor
	real silence_threshold
	real voicing_thresholding
	real minimum_pitch
	text subtract_mean
endform

audio = Read from file: wav_file$

pitch = To Pitch: 0.0, pitch_min, pitch_max
Save as text file: pitch_file$

pitch_tier = Down to PitchTier
pitch_tier_tor = Down to TableOfReal: "Hertz"
pitch_tier_table = To Table: "Index"
Remove column: "Index"
Save as comma-separated file: pitch_tier_file$

selectObject: audio

intensity = To Intensity: minimum_pitch, 0.0, subtract_mean$
Save as text file: intensity_file$

intensity_tier = Down to IntensityTier
intensity_tier_tor = Down to TableOfReal
intensity_tier_table = To Table: "Index"
Remove column: "Index"
Save as comma-separated file: intensity_tier_file$

selectObject: pitch_tier

point_process = To PointProcess
Save as text file: point_process_file$

plusObject: audio_event_segment
plusObject: pitch
# TODO vedere come gestire il voice report
voice_report$ = Voice report: 0, 0, pitch_min, pitch_max, max_period_factor, max_amplitude_factor, silence_threshold, voicing_thresholding

removeObject: audio, pitch, pitch_tier, pitch_tier_tor, pitch_tier_table, point_process, intensity, intensity_tier, intensity_tier_tor, intensity_tier_table
