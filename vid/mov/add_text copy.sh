#!/bin/bash

FNIN='compare_synch_v_asynch_only_explicit.mov'
FNOUT='out.mov'
# FNOUT='compare_synch_v_asynch_only_explicit_annotated.mov'
TEXTA='runtime = 22 minutes' 
TEXTB='runtime = 5 minutes'

ffmpeg -i $FNIN -filter_complex "
drawtext=times.ttf:fontsize=50:fontcolor=000000:
	x=(main_w-text_w)*(3/8-0.08):
	y=(h-text_h)*11.5/16:
	text=$TEXTA:enable='between(t,12,16)',
drawtext=times.ttf:fontsize=50:fontcolor=000000:
	x=(main_w-text_w)*0.98:
	y=(h-text_h)*11.5/16:
	text=$TEXTB:enable='between(t,12,16)',
fade=t=in:start_time=12:d=0.01:alpha=1,
fade=t=out:start_time=15.5:d=0.5:alpha=1[fg];
[0][fg]overlay=format=auto,format=yuv420p
" -c:a copy -c:v libx264 -crf 30 -r 60 -y $FNOUT
