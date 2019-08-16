FROM ubuntu:18.04

#工作路径？
#WORKDIR $HOME
#安装必要软件

#更新软件源为阿里源
RUN sed -i "1ideb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\ndeb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\ndeb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\ndeb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\ndeb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse\ndeb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\ndeb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\ndeb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\ndeb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\ndeb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse\n" /etc/apt/sources.list
#写入文件并更新,配置vim
RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y vim wget curl cmake make g++ git 
	&& apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler \
	--no-install-recommends libboost-all-dev  \
	libgflags-dev libgoogle-glog-dev liblmdb-dev \
	libatlas-base-dev \
	python-numpy \
	libhdf5-serial-dev \
	python-dev \ 
	python-pip 
	
#下载caffe
RUN cd /opt 
RUN git clone git://github.com/BVLC/caffe.git 
RUN cp Makefile.config.example Makefile.config 
RUN cd caffe

# 编译cpu-caffe 修改Makefile.config配置文件内容
	
# expose port for ssh and start ssh service.
EXPOSE 22
# expose port for notebook.
EXPOSE 8888
# expose port for tensorboard.
EXPOSE 6666
