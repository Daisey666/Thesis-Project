form audio_info_extraction
	text wav_file
	text param_file
	text pitch_tier_file
	text intensity_tier_file
	text voice_report_file
	real window_size
	real window_shift
	real pitch_min
	real pitch_max
	real max_period_factor
	real max_amplitude_factor
	real silence_threshold
	real voicing_thresholding
endform

audio = Read from file: wav_file$

total_duration = Get total duration

pitch = To Pitch: 0.0, pitch_min, pitch_max
dx_pitch = Get time step

pitch_tier = Down to PitchTier
pitch_tier_tor = Down to TableOfReal: "Hertz"
pitch_tier_table = To Table: "Index"
Remove column: "Index"
Save as comma-separated file: pitch_tier_file$

selectObject: audio

intensity = To Intensity: pitch_min, 0.0, "no"
dx_intensity = Get time step

intensity_tier = Down to IntensityTier
intensity_tier_tor = Down to TableOfReal
intensity_tier_table = To Table: "Index"
Remove column: "Index"
Save as comma-separated file: intensity_tier_file$

selectObject: pitch_tier

point_process = To PointProcess

param_table_columns$ = "dx_pitch dx_intensity window_size_vr window_shift_vr"
param_table = Create Table with column names: "params", 1, param_table_columns$
Set numeric value: 1, "dx_pitch", dx_pitch
Set numeric value: 1, "dx_intensity", dx_intensity
Set numeric value: 1, "window_size_vr", window_size
Set numeric value: 1, "window_shift_vr", window_shift
selectObject: param_table
Save as comma-separated file: param_file$

start = 0.0
end = window_size

voice_report_table_columns$ = "start_time end_time harmonicity jitter shimmer"
voice_report_table = Create Table with column names: "voice_r", 0, voice_report_table_columns$

while end < total_duration + window_shift

	if end > total_duration
		end = total_duration
	endif

	selectObject: audio
	plusObject: pitch
	plusObject: point_process

	voice_report$ = Voice report: start, end, pitch_min, pitch_max, max_period_factor, max_amplitude_factor, silence_threshold, voicing_thresholding

	selectObject: voice_report_table
	Append row
	r = Get number of rows
	Set numeric value: r, "start_time", start
	Set numeric value: r, "end_time", end
	Set numeric value: r, "harmonicity", extractNumber (voice_report$, "Mean autocorrelation: ")
	Set numeric value: r, "jitter", extractNumber (voice_report$, "Jitter (local): ")
	Set numeric value: r, "shimmer", extractNumber (voice_report$, "Shimmer (local): ")

	start = start + window_shift
	end = end + window_shift

endwhile

selectObject: voice_report_table
Save as comma-separated file: voice_report_file$

removeObject: audio, pitch, pitch_tier, pitch_tier_tor, pitch_tier_table, point_process, intensity, intensity_tier, intensity_tier_tor, intensity_tier_table, param_table, voice_report_table
