# Cartoon-StyleGAN 🙃 : Fine-tuning StyleGAN2 for Cartoon Face Generation

Source: [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)

**Install Docker** 

```none ```

``` create: docker run -i -t ubuntu:latest /bin/bash ```
``` create: apt-get update -> apt-get install python3 -> apt install python-is-python3 ```
``` show all: docker ps -a ```
``` show all: docker export ID > test.tar ```
``` show all: docker import - NAME < test.tar ```
``` show all: docker images ```
``` show all: docker run -i -t test:latest /bin/bash ```
```  ```

**Run tests**  

``` python -m unittest -v test ```

**Run Inference example**

```python exampe.py --outdir outputs --video_name example_video --number_of_img 2 --number_of_step 7```

