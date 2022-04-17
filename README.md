# Cartoon-StyleGAN 🙃 : Fine-tuning StyleGAN2 for Cartoon Face Generation

Source: [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)

## Installation

### Install Docker
Use the [original documentation](https://docs.docker.com/get-docker/)

### Clone repo
```
git clone https://github.com/MaximovaIrina/Cartoon-StyleGAN-example.git  
cd Cartoon-StyleGAN-example
```

### Create CPU docker
Ubuntu:
```
docker run -it -v "$(pwd)":/Cartoon-StyleGAN-example ubuntu:latest bash
```
Windows:
```
docker run -it -v /YOUR/PATH/TO/Cartoon-StyleGAN-example/REPO:/Cartoon-StyleGAN-example ubuntu:latest bash
```


### Setup CPU docker 
```
apt-get update && \
apt-get install python3 && \
apt install python-is-python3 && \
apt install python3-pip && \
apt install git && \
apt-get install ffmpeg libsm6 libxext6 -y 
```

### Install requirements
```
cd Cartoon-StyleGAN-example
pip install -r requirements.txt
```

## Download pretrained models
```
python download_pretrained.py
```

## Run tests
```
python -m unittest -v test
```

## Run Inference example
```
python example.py --outdir outputs --video_name example_video --number_of_img 2 --number_of_step 7
```
Check result in `outputs` folder

## Close docker 
```
exit
docker ps -a
docker images
```

```
docker rm CONTAINER_ID
docker image rm IMAGE_ID
```
