# Cartoon-StyleGAN 🙃 : Fine-tuning StyleGAN2 for Cartoon Face Generation

Source: [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)

## 1. Prepare environment
If you don't have docker, install it using the [original documentation](https://docs.docker.com/get-docker/)

### Clone repo
```
git clone https://github.com/MaximovaIrina/Cartoon-StyleGAN-example.git  
cd Cartoon-StyleGAN-example
```

## Create docker
Ubuntu:
```
docker run -it -v "$(pwd)":/Cartoon-StyleGAN-example ubuntu:20.04 bash
```
Windows:
```
docker run -it -v /YOUR/PATH/TO/Cartoon-StyleGAN-example:/Cartoon-StyleGAN-example ubuntu:20.04 bash
```
Note: use ubuntu:20.04 (35MB) image instead of ubuntu:latest (5GB)

## Setup docker
```
apt-get update && \
apt-get install python3 && \
apt install python-is-python3 && \
apt install python3-pip && \
apt install git && \
apt-get install ffmpeg libsm6 libxext6 -y 
```

## 2. Install requirements
```
cd Cartoon-StyleGAN-example
pip install -r requirements.txt
```

## 3. Download pretrained models
```
python download_pretrained.py
```

## 4. Run tests
```
python -m unittest -v test
```

## 5. Run Inference example
```
python example.py --outdir outputs --video_name example --number_of_img 2 --number_of_step 7
```
Check result in `outputs` folder

## 6. Close docker 
```
exit
docker ps -a
docker stop CONTAINER_ID
```
