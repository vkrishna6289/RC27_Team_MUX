# 5G Uplink SRS Signal based Integrated Sensing and Communication

IEEE Radar Challenge 2027 (RadarConf'27, Bengaluru), theme 5.10 (ISAC)

Team MUX, SRM Institute of Science and Technology, Chennai, India
- V Krishna (team lead)
- Tharun B
- Dr. Arumbu V N (faculty mentor)

## Summary
A live 5G NR standalone network (Open5GS core, srsRAN gNB) on two USRP N310
radios. The gNB estimates channel state information (CSI) from the UE's uplink
sounding reference signal (SRS) and forms a bistatic range-Doppler map that
locates the UE and nearby passive targets, while user data keeps flowing.

## Processing chain
SRS reception -> least-squares CSI -> direct-path calibration ->
IFFT over subcarriers (range) and FFT over SRS occasions (Doppler) ->
clutter removal and CFAR -> range-Doppler map

## Planned hardware and software
2 x USRP N310,LP0965 antennas; Open5GS, srsRAN, UHD, Python

## Status
Proposal stage. Hardware arrives January 2027. Code, configs, measurements and
results will be added here as the work progresses, and everything will be
published openly.

## Note
No credentials or SIM keys are stored in this repository.
