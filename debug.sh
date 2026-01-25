
/home/nvidia/nvsipl_multicast -m "01" -p "FileSrc=type=4:path=./FileSource_Test_Data:width=1920:height=1080,Cuda=filesink=1:width=1920:height=1080"


# 侧视
198.18.45.52:53306 -> 224.0.0.130.53304
198.18.45.52:53310 -> 224.0.0.130.53308
198.18.45.52:53314 -> 224.0.0.130.53312
198.18.45.52:53318 -> 224.0.0.130.53316

# ADC IP
198.18.45.52

# AI BOX IP
198.18.45.9

./receiving_hook 198.18.45.9 53304 265 224.0.0.130
./receiving_hook 198.18.45.9 53308 265 224.0.0.130
./receiving_hook 198.18.45.9 53312 265 224.0.0.130
./receiving_hook 198.18.45.9 53316 265 224.0.0.130

./receiving_poll
