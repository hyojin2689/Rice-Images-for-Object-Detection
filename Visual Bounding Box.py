path = '/content/drive/MyDrive/객체탐지/testrice'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.xml')] 
file_list_py.sort()
file_list_py

with open(path + '/' + file_list_py[3], 'r') as f:
  data = f.read()

root = BeautifulSoup(data, "xml")

root

path = '/content/drive/MyDrive/객체탐지/testrice'
file_list = os.listdir(path)
image_file_list_py = [file for file in file_list if file.endswith('.jpg')] 
image_file_list_py.sort()
image_file_list_py

image_file_list_py[7]

image_name = root.find('filename').text
full_image_name = os.path.join(path ,image_file_list_py[3])

img = cv2.imread(full_image_name)
#opencv의 rectangle()는 인자로 들어온 이미지 배열에 그대로 사각형을 그리기 때문에 별도의 이미지 배열을 만든다
draw_img = img.copy()
green_color=(0, 255, 0) #OpenCV는 RGB가 아니라 BGR이므로 빨간색은 (0, 0, 255)
red_color=(0, 0, 255)

#파일 내에 있는 모든 object Element=<object>를 찾음
objects_list = []
for obj in root.find_all('object'):
    xmlbox = obj.find('bndbox')
    
    left = int(xmlbox.find('xmin').text) #bbox의 좌표값
    top = int(xmlbox.find('ymin').text)
    right = int(xmlbox.find('xmax').text)
    bottom = int(xmlbox.find('ymax').text)
    class_name=obj.find('name').text #오브젝트의 이름을 가져온다
    
    #draw_img 배열의 좌상단 우하단 좌표에 초록색으로 box 표시 
    cv2.rectangle(draw_img, (left, top), (right, bottom), color=green_color, thickness=7)
    #draw_img 배열의 좌상단 좌표에 빨간색으로 클래스명(이름) 표시. 좌상단의 top보다 높은 곳에 적는다
    cv2.putText(draw_img, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.5, red_color, thickness=3)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
