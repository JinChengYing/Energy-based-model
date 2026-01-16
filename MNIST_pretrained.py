import urllib.request
from urllib.error import HTTPError
# Github URL where saved models are stored for this tutorial
#远程仓库
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial8/"
# Files to download
#等待下载的文件列表
pretrained_files = ["MNIST.ckpt", "tensorboards/events.out.tfevents.MNIST"]
#准备目录
# Create checkpoint path if it doesn't exist yet
#跳过已存在的文件夹并且跳过
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    #使用os.path.join将总目录和文件名拼成绝对路径
    file_path = os.path.join(CHECKPOINT_PATH, file_name)#os.path.join可以识别linux和windows的不同的路径的斜杠方向.
    if "/" in file_name:#按照/提取文件名
        os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)#从右边找第一个"/",把左边的路径切出来,
        # 然后在本地创建这个子目录，否则直接下载到不存在的目录会报错, 也就是先建立好目录再下载.
    if not os.path.isfile(file_path):#避免循环下载,检查该文件是否已经在本地硬盘上
        file_url = base_url + file_name #得加上目录路径,得到完整的下载链接
        print(f"Downloading {file_url}...")
        try:
            #从file_url获取数据, 写入file_path
            urllib.request.urlretrieve(file_url, file_path) #把url的内容存到path里面
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)