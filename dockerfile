FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

#docker-composeから引数を受け取る
ARG http_tmp
ARG https_tmp

#環境変数にプロキシの設定を追加
ENV http_proxy=$http_tmp
ENV https_proxy=$https_tmp

#タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#apt関連の設定
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y python3 python3-pip tmux

