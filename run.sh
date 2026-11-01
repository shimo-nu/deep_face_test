#!/bin/bash

docker run -it --name face_recognition_container --rm --gpus all -p 18381:18381 -p 18382:18382 -v .:/mnt face_recognition_image bash
