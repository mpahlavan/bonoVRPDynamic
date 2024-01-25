# marpdan

How to run:
- Install docker desktop
- install vscode

open project in terminal:

docker build -t vrp_bono_image -f docker/Dockerfile  .
docker ps -a
docker image
 docker run -it --name vrp_bono_container vrp_bono_image