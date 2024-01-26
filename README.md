## How to Run
### Prerequisites
- Docker Desktop installed
- Visual Studio Code installed
### Steps
1. Open the project in Visual Studio Code.
2. Open the terminal inside VSCode and execute the following commands:

```
docker build -t vrp_bono_image -f docker/Dockerfile .
docker ps -a
docker image
docker run -it --name vrp_bono_container vrp_bono_image
```
In VSCode, press Ctrl+Shift+P, then select Dev container: Attach to Running Container... and choose /vrp_bono_container.

This will open the container, allowing you to run the code directly inside it.

To run the training phase, use the following command:

bash
Copy code
python script/train.py
Save to grepper
Now, you're set to execute the code seamlessly within the Docker container.