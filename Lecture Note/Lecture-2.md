# Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 2: Image Classification with Linear Classifiers

> Link: [Lecture Video](https://youtu.be/pdqofxJeBN8?si=bDjuYjE54yVT-zPl)
> Slide: [Slide](https://cs231n.stanford.edu/slides/2025/lecture_2.pdf)

---

## Syllabus

![1775572829562](image/Lecture-2/1775572829562.png)

## Image Classification Task

### Image Classification: Core task of CV

- **What to do?**
  - image -> label (one of the possible labels)
  
- **Problem**
  - Sementic Gap
    - 사람이 보는(파악하는) 이미지 vs 컴퓨터(기계)가 인식하는 이미지
    - 시각적 이미지 vs 행렬(a tensor of integers)
  
- **Challenges**
  - Illumination(조명, 광원)
  - Background Clutter(어수선한 배경)
  - Occlusion(가려짐)
  - Deformation(변형)
  - Intraclass Variation(계급(집합) 내 다양성)
  - Context(맥락)
  - etc...

### An Image Classifier

- **Basic Classifier Concept**
  ```python
  def classify_image(image):
    # Something in Here
    return class_label
  ``` 

- **Problem**
  - 정렬(Sorting)과 같은 문제와 달리, hard-code로써 이미지의 class를 분류하는 algorithm(or the steps)을 구축하는 분명한(obvious) 방법은 없음

- **Efforts?**
  - 그럼에도 불구하고, (DL을 통한 접근 이전의) hard-code algorithms을 기반으로 한 몇 시도들이 존재
    - E.g., Edge Detecting
      - 변동성(variability)이 극히 드문 경우에 항하여 제한적으로 성공을 거둠
      - 단, 알고리즘의 확장이 어렵고, 각각의 경우에 적합한 논리를 찾는 것 자체가 힘들기 때문에 그다지 좋은 접근이라고 볼 수 없었음

## Data Driven Approaches to Image Classification

- Machine Learning: What to do?

  ```python
  def train(image, labels):
    # Machine Learning
    return model

  def predict(model, test_images):
    #Use model to predict labels (to evaluate the model)
    return test_labels
  ```
  1. Collect a dataset of images and labels
  2. Use machine learning algorithms to train a classifier
  3. Evaluate the classifier on new images
   
  - Don't build a logic, build a data driven approach

### Nearest Neighbor Classifier

- The easiest form of classification

- How to build?
  ```python
  def train(image, labels): 
    # Machine Learning
    return model

  def predict(model, test_images):
    #Use model to predict labels (to evaluate the model)
    return test_labels
  ```
  - the ```train``` function needs to memorize all data and labels
  - the ```predict``` function needs to predict the label of the most similar training data

 ![1775640198910](image/Lecture-2/1775640198910.png)

### Linear Classifier
