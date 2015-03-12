__author__ = 'v-penlu'

import os

session_cnt = 5
src_dir = r"D:\IEMOCAP_full_release"
des_dir = r"D:\IEMOCAP_full_release\emobase"
conf = r"D:\EclipseProjects\SequenceToSequence\emobase_less.conf"
openSMILE = "D:\openSMILE-2.1.0\\bin\Win32\SMILExtract_Release.exe"
if not os.path.exists(des_dir):
    os.mkdir(des_dir)

for i in range(1, session_cnt + 1):
    section_dir = os.path.join(src_dir, ("Session%d" % i), "sentences", "wav")
    des_sec_dir = os.path.join(des_dir, ("Session%d" % i))
    if not os.path.exists(des_sec_dir):
        os.mkdir(des_sec_dir)
    for wav_dir in os.listdir(section_dir):
        src_wav_dir = os.path.join(section_dir, wav_dir)
        if os.path.isdir(src_wav_dir):
            des_wav_dir = os.path.join(des_sec_dir, wav_dir)
            if not os.path.exists(des_wav_dir):
                os.mkdir(des_wav_dir)
            for sen in os.listdir(src_wav_dir):
                if sen.endswith(".wav"):
                    sen_name = os.path.splitext(sen)[0]
                    wav_name = os.path.join(src_wav_dir, sen)
                    csv_name = os.path.join(des_wav_dir, sen_name + ".csv")
                    cmd = "%s -C %s -I %s -O %s" % (openSMILE, conf, wav_name,
                                                    csv_name)
                    os.system(cmd)
