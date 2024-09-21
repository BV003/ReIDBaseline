               PKUSketchRE-ID Dataset Description
==================================================================
The dataset contains 200 persons, each of which has one sketch and two photos.

Photos of each person were captured during daytime by two cross-view cameras. 
We cropped the raw images (or video frames) manually to make sure that every 
photo contains the one specific person.

We have a total of 5 artists to draw all persons’ sketches and 
every artist has his own painting style.

If you use this dataset, please kindly cite our paper as,
Lu Pang, Yaowei Wang, Yi-Zhe Song, Tiejun Huang, Yonghong Tian; Cross-Domain Adversarial Feature Learning for Sketch Re-identification; 
ACM Multimedia 2018
===================================================================
The package contains three folders.
1. 'sketch' folder. This folder contains 200 sketches. 
   Naming rule of the sketches:
   In sketch '1.jpg', '1' is the person ID.
2. 'photo' folder. This folder contains 400 photos(Each of person has two photos).
   Naming rule of the photos:
   In photo '1_05_356.jpg', '1' is the person ID.
3. 'styleAnnotation' folder. This folder contains five documents, which correspond to five painting styles.
   Each document contains IDs of persons who are portrayed by a specific painting style.
   
   Naming rule of the documents:
   In file 'a_46.txt', 'a' is the type of painting style and '46' denotes the number of sketches that belong to 'a' style.
   
   Note: We correct wrong statistics on the number of 'a' style and 'c' style in our paper.
