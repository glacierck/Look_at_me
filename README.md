# Face_Recognition_System 😎

## License

The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**

## Project Structure🌲
1. Front_end
   - pyqt5 ->Windows desktop app for gate access control
   - django ->web app for guys data upload ,management and analysis
2. Back_end
   1. database
      - milvus_lite ->face_embedding 
      - sqlite3     ->person_info
    2. algorithm
       - general functions
       - insightface ->face_recognition 👩🏻‍🎓👨🏻‍🎓
       - yolo v8/Nas ->car_board_recognition 🚗

## Project Process🌈
1. insightface
2. milvus
3. sqlite3
5. flask

## Project finishing log 📝
### 2023-8-1
### tasks
1. test the new milvus server ✅
   1. refresh start
   2. no refresh start
   3. insert entries while searching 
2. sqlite connect with metabase ✅ ->docker_starter
3. flask log page
4. sqlite design


## 2023-7-31
### tasks
1.  delete the useless process of start milvus server ✅
2. try to make the milvus dynamic which means the server can be insert and delete dynamically ✅ 
3. add func check entries before insert data into milvus ✅


### 2023-7-10
## Still working on
1. maybe the way of reading data from files can be optimized by asynchronous reading ⏳
2. simplify the code of class Image ⏳
3. try to use logging to record the process of program ⏳
4. try to add method of test_videos in class FaceAnalysisTest ⏳

## Done
1. make the __init__() of class Milvus,FaceAnalysis,--Test,Image more slightly ✅
2. add the Error handling of class Milvus,FaceAnalysis,--Test,Image ⏳
3. test insert ,_create_collection,_base_config_set methods in class Milvus ✅
4. finished the construction of  **class Milvus** ✅
5. try to create collection and insert data into milvus_lite from npy files ✅
6. try to use milvus_lite to search the face data ✅

### 2023-7-8
## Still working on
1. maybe the way of reading data from files can be optimized by asynchronous reading ⏳
2. test insert ,_create_collection,_base_config_set methods in class Milvus ⏳
3. set index parameters in milvus_lite ⏳
4. finish the construction of  **class Milvus** ⏳
5. try to create collection and insert data into milvus_lite from npy files ⏳
6. try to use milvus_lite to search the face data ⏳
## Done
1. finished swap the face embedding data from npz to npy ✅
2. completed construction of the function of insert,_create_collection,_base_config_set in class Milvus ✅

### 2023-7-5
## Still working on
1. finish the construction of  **class Milvus** ⏳
2. try to create collection and insert data into milvus_lite from npy files ⏳
3. try to use milvus_lite to search the face data ⏳
4. try to swap the face embedding data from npz to npy ⏳
## Done
1. load images from mess folder with Path.rglob() ✅
2. accelerate the process of get image's ndarray or face embedding  by using npio ✅
3. try a large register of face data which is 10000+ and prepared for milvus search ✅


### 2023-7-4
## Questions
 - can Django interact with milvus_lite ?
## Done
1. finish the construction of  **class Milvus** ⏳
2. almost figure out the API of Milvus ✅


### 2023-7-3
1. get more models which have the best accuracy of recognize south Asian from insightface and try to use them ✅
2. move models folder out from project folder for easier to update git ✅

### 2021-7-2
1. stop using docker for pymilvus, use milvus **(milvus lite)** instead  ✅
   - docker is too complex to use
   - milvus lite is easy to use directly
2. got an example of **milvus lite** and start to figure out how to use it ✅
3. server need to be **stopped** (or it'll fail to w/r next time) and restarted after each time of using it
4. ***class Milvus*** in milvus_lite.py is constructing , finished the part of __init__() ✅

### 2023-7-1
1. **docker install** ✅
   - error
       - unexpected error was encountered while executing a WSL command
   - solved by
     - wsl -update
2. **milvus install** ✅
   - image from docker hub
       - milvusdb/milvus:latest
   - python sdk
     - pip install pymilvus->milvus
     - pip install milvus->milvus lite