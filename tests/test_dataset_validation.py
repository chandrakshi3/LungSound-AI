import numpy as np
import soundfile as sf
from src.data_loader import parse_annotation,discover_dataset
def test_parse(tmp_path):
 p=tmp_path/'a.txt';p.write_text('0 1 1 0\n1 2 0 1\n');assert len(parse_annotation(p))==2
def test_invalid_annotation(tmp_path):
 p=tmp_path/'a.txt';p.write_text('0 0 1 0\n')
 try:parse_annotation(p)
 except ValueError:return
 assert False,'invalid interval should fail'
def test_missing_annotation(tmp_path):
 (tmp_path/'audio').mkdir();sf.write(tmp_path/'audio'/'101_1b1_x.wav',np.zeros(16000),16000);v,i,n=discover_dataset(tmp_path);assert n==1 and len(v)==0 and len(i)==1
def test_valid_recording(tmp_path):
 (tmp_path/'audio').mkdir();(tmp_path/'annotations').mkdir();sf.write(tmp_path/'audio'/'101_1b1_x.wav',np.zeros(32000),16000);(tmp_path/'annotations'/'101_1b1_x.txt').write_text('0 1 0 0\n1 2 1 0\n');v,i,n=discover_dataset(tmp_path);assert len(v)==1 and len(i)==0
