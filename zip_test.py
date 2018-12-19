import os
import time
import zipfile
# 尝试压缩文件
source_dir = 'C:\\temp_python\\zipprepare'
#
target_dir = 'C:\\temp_python\\zipprepare2'
#
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
#
source_name = os.path.split(source_dir)
#
def make_zip(source_dir, output_filename):
  zipf = zipfile.ZipFile(output_filename, 'w')
  pre_len = len(os.path.dirname(source_dir))
  for parent, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
        pathfile = os.path.join( parent, filename )
        arcname = pathfile[pre_len:].strip(os.path.sep)   #去掉路径头尾的斜杠 输出相对路径
        zipf.write(pathfile, arcname)
  zipf.close()
#
out_filename = target_dir + os.sep + source_name[1]+ time.strftime('%Y%m%d%H%M%S')+ '.zip'
#
make_zip(source_dir, out_filename)
#