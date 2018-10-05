form silences_info_extraction
	text wav_file
	text intensity_tier_file
	text silences_file
	real silence_threshold_db
	real minimum_silent_interval_duration
	real minimum_sounding_interval_duration
	real time_step
	positive time_decimals
endform

audio = Read from file: wav_file$
end = Get end time

intensity_tier_table = Read Table from comma-separated file: intensity_tier_file$
n_rows = Get number of rows

intensity_tier = Create IntensityTier: "intensity_tier", 0, end

for r to n_rows

	selectObject: intensity_tier_table
	t_col$ = Get column label: 1
	i_col$ = Get column label: 2
	t = Get value: r, t_col$
	i = Get value: r, i_col$

	selectObject: intensity_tier
	Add point: t, i

endfor

selectObject: intensity_tier

silences = To TextGrid (silences): silence_threshold_db, minimum_silent_interval_duration, minimum_sounding_interval_duration, "silent", "sounding", time_step
silences_table = Down to Table: "no", time_decimals, "no", "yes"
Save as comma-separated file: silences_file$

removeObject: audio, intensity_tier_table, intensity_tier, silences, silences_table
