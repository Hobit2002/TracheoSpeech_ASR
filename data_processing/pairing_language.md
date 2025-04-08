# Sequence alignment language (SAL)

## Motivation

This document specifies the manual refinement of pre-generated textual labels for shorter segments of recorded audio. 

## Labelling Environment

Once the labeling procedure starts, the developer performing the manual annotations is presented both with the textual data and with the audio. The audio is played and also displayed (together with preceding and succesive three seconds) as a spectrogram. The text is printed.

User is then asked to type in a command, wheter the audio-text pair at hand should be comitted whether some of the tokens should be modified or whether a different action should be taken. We call the language SAL and specify it in full detail bellow. 

## Commands

All commands have the form `(< sequence >)< action >`. The `< sequence >` refers either to `A` meaning that the command will affect only the audio sequence, `T` meaning that only the textual tokens will be modified or nothing which means that the both sequences will be affected. The actions that affect either only one of the sequences or only both of them are not accompanied  by the `< sequence >`.

* `< blank space >` ... both sequences
* `T` ... textual tokens
* `A` ... audio tokens


These prefixes can be used for the following commands:

* `S($s)` ... skip (move to the next element of the sequence(s): `AS` to move to the next audio, `TS` for the next piece of text or just `S` for both). The `$s` parameter specifies the hop length with the default value being 1.
* `R($s)` ... return to the previous element(s) of the sequence(s). Inverse operation to `S`. The `$s` parameter specifies the hop length with the default value being 1.
* `P` ... combine the token with the previous one. If used for the audio (i.e. `AP` also the originally filtered part between the tokens will be included)  


The commands bellow always affect both audio and the textual data. The sequence prefix is not used for these commands:

* `C` ... commit: save the elements as a pair and move on. 
* `U` ... uncommit: if the current pair is among the commited ones, remove it from this set.
* `L($s)` ... splits the string by spaces and separates the `string[:$s]` substring for positive `$s` and `string[$s:]` for the negative one. The separated string is inserted at the appropriate position in the list of textual tokens.

The following commands are applicable only for audio.The sequence prefix is not used:

* `E$s` ... extend the audio recording by `$s` following milliseconds. Negative offsets are possible.
* `B$s` ... extend the audio recording by `$s`ts are possible.
* `M($s)` ... replay the audio recording and `$s` preceding and following milliseconds. If not specified, the offset is zero. The parameter might be written also in the form `($before)s($after)` .

The following commands are applicable only for the text.The sequence prefix is not used:

* `W` ... rewrite the string (an input line where the new string should be placed appears)
* `N` ... merge the current textual command with next one

Finally, there is a command to terminate the whole process:

* `END` ... end the whole process

Multiple commands can by typed in at once, they just have to be separated by semicolons, i.e. `$command1;$command2;$command3`