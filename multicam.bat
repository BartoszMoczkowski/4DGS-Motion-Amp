mkdir data\multipleview\%1
move *.mp4 data\multipleview\%1\
mkdir data\multipleview\%1\cam01 data\multipleview\%1\cam02 data\multipleview\%1\cam03 data\multipleview\%1\cam04
ffmpeg -i data\multipleview\%1\camera1.mp4 -qscale:v 1 -ss 1 -qmin 1 -vf fps=20 data\multipleview\%1\cam01\frame_%%05d.jpg
ffmpeg -i data\multipleview\%1\camera2.mp4 -qscale:v 1 -ss 1 -qmin 1 -vf fps=20 data\multipleview\%1\cam02\frame_%%05d.jpg
ffmpeg -i data\multipleview\%1\camera3.mp4 -qscale:v 1 -ss 1 -qmin 1 -vf fps=20 data\multipleview\%1\cam03\frame_%%05d.jpg
ffmpeg -i data\multipleview\%1\camera4.mp4 -qscale:v 1 -ss 1 -qmin 1 -vf fps=20 data\multipleview\%1\cam04\frame_%%05d.jpg